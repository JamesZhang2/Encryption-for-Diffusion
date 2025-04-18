import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self):
        pass


class DownSample(nn.Module):
    def __init__(self):
        pass


class UpSample(nn.Module):
    def __init__(self):
        pass


class ResnetBlock(nn.Module):
    def __init__(self):
        pass


class AttentionBlock(nn.Module):
    def __init__(self):
        pass


class ResnetAndAttentionBlock(nn.Module):
    def __init__(self):
        pass


class UNet(nn.Module):
    """
    A U-Net architecture for diffusion.
    """

    def __init__(self, source_channel: int, unet_base_channel: int, num_groups: int):
        """
        Initialize the UNet module.

        Args:
            source_channel (int): Number of input image channels
            unet_base_channel (int): Base number of channels for the U-Net
            num_groups (int): Number of groups for group normalization
        """
        super(UNet, self).__init__()

        self.pos_enc = PositionalEncoding(
            base_dim=unet_base_channel,
            hidden_dim=unet_base_channel*2,
            output_dim=unet_base_channel*4,
        )

        c = unet_base_channel
        emb_dim = unet_base_channel * 4

        self.down_conv = nn.Conv2d(
            source_channel,
            unet_base_channel,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        self.down_blocks = nn.ModuleList([
            # Layer 1
            ResnetBlock(c, c, num_groups, emb_dim),
            ResnetBlock(c, c, num_groups, emb_dim),
            DownSample(c),
            # Layer 2
            ResnetAndAttentionBlock(c, c * 2, num_groups, emb_dim),
            ResnetAndAttentionBlock(c * 2, c * 2, num_groups, emb_dim),
            DownSample(c * 2),
            # Layer 3
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
            DownSample(c * 2),
            # Layer 4
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
        ])

        self.mid_blocks = nn.ModuleList([
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
            AttentionBlock(c * 2, num_groups),
            ResnetBlock(c * 2, c * 2, num_groups, emb_dim),
        ])

        self.up_blocks = nn.ModuleList([
            # Layer 4
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            UpSample(c * 2),
            # Layer 3
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetBlock(c * 4, c * 2, num_groups, emb_dim),
            UpSample(c * 2),
            # Layer 2
            ResnetAndAttentionBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetAndAttentionBlock(c * 4, c * 2, num_groups, emb_dim),
            ResnetAndAttentionBlock(c * 3, c * 2, num_groups, emb_dim),
            UpSample(c * 2),
            ResnetBlock(c * 3, c, num_groups, emb_dim),
            ResnetBlock(c * 2, c, num_groups, emb_dim),
            ResnetBlock(c * 2, c, num_groups, emb_dim),
        ])

        self.norm = nn.GroupNorm(
            num_groups,
            unet_base_channel,
            eps=1e-6
        )

        self.up_conv = nn.Conv2d(
            unet_base_channel,
            source_channel,
            kernel_size=3,
            stride=1,
            padding='same'
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet module.

        Args:
            x (torch.Tensor): Input images
            t (torch.Tensor): Timestep
            y_emb (torch.Tensor): Class embedding
        """

        t_embs = self.pos_enc(t)
        skip_connections = []

        out = self.down_conv(x)

        for block in self.down_blocks:
            match type(block):
                case DownSample():
                    out = block(out)
                case ResnetBlock() | ResnetAndAttentionBlock():
                    out = block(out, t_embs, y_emb)
                case _:
                    raise ValueError("Unknown block type")
            skip_connections.append(out)

        for block in self.mid_blocks:
            match type(block):
                case ResnetBlock():
                    out = block(out, t_embs, y_emb)
                case AttentionBlock():
                    out = block(out)
                case _:
                    raise ValueError("Unknown block type")

        for block in self.up_blocks:
            match type(block):
                case UpSample():
                    out = block(out)
                case ResnetBlock() | ResnetAndAttentionBlock():
                    skip = skip_connections.pop()
                    out = torch.cat((out, skip), dim=1)
                    out = block(out, t_embs, y_emb)
                    out = block(out, t_embs, y_emb)
                case _:
                    raise ValueError("Unknown block type")

        assert not skip_connections, "Not all skip connections were used"

        out = self.norm(out)
        out = F.silu(out)
        out = self.up_conv(out)

        return out

    def __call__(self, x, t, y_emb):
        return self.forward(x, t, y_emb)


if __name__ == "__main__":
    unet = UNet(source_channel=3, unet_base_channel=64, num_groups=32)
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([0])
    y_emb = torch.randn(1, 64)
    out = unet(x, t, y_emb)
    print(out.shape)
