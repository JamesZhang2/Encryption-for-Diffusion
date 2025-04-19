import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionalEncoding(nn.Module):
    """Implements fixed sinusoidal positional encodings for sequential data.

    This module generates transformer-style positional encodings using interleaved
    sine and cosine functions with geometrically increasing wavelengths:

        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    where:
    - t: timestep value
    - d: encoding dimension (base_dim)
    - i: dimension index

    Attributes:
        inv_freq (torch.Tensor): Buffer storing inverse frequency values
                                Shape: (base_dim // 2,)
    """

    def __init__(self, base_dim: int):
        """Initialize sinusoidal positional encoding module.

        Args:
            base_dim: Dimension of the encoding vectors. Must be even.
                     Typical values: 128, 256, 512

        Raises:
            AssertionError: If base_dim is not even
        """
        super().__init__()
        assert base_dim % 2 == 0, "base_dim must be even"

        self.register_buffer(
            'inv_freq',
            1.0 / (10_000 ** (torch.arange(0, base_dim,
                   2).float() / base_dim)),
            persistent=False
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate positional encodings for input timesteps.

        Args:
            timesteps: Input tensor of timestep values 
                      Shape: (batch_size,)
                      Example: torch.tensor([0, 1, 2, 3])

        Returns:
            Positional encodings with interleaved sine/cosine values
            Shape: (batch_size, base_dim)
            Example: For base_dim=4 → [sin(t), cos(t), sin(t/10000^(2/4)), cos(t/10000^(2/4))]

        Processing:
            1. Compute angle values for each timestep
            2. Generate sine and cosine components
            3. Interleave results into final encoding
        """
        t = timesteps.unsqueeze(1)  # add dim 1
        inv_freq = self.inv_freq.unsqueeze(0)  # add dim 0
        theta = t * inv_freq  # t / (10_000)^(2i/d)
        sines = torch.sin(theta)
        cosines = torch.cos(theta)
        out = torch.empty((sines.shape[0], 2*sines.shape[1]))
        out[:, 0::2] = sines
        out[:, 1::2] = cosines
        return out


class PositionalEncoding(SinusoidalPositionalEncoding):
    """Learned positional encoding with nonlinear transformation.

    Extends the base sinusoidal encoding with:
    1. A two-layer feedforward network
    2. SiLU nonlinear activation
    3. Dimension projection

    Inherits the sinusoidal pattern from parent class:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Attributes:
        linear1 (nn.Linear): First transformation layer (base_dim → hidden_dim)
        linear2 (nn.Linear): Second transformation layer (hidden_dim → output_dim)
    """

    def __init__(self, base_dim: int, hidden_dim: int, output_dim: int):
        """Initialize enhanced positional encoding module.

        Args:
            base_dim: Input dimension for sinusoidal encoding
            hidden_dim: Hidden layer dimension in feedforward network
            output_dim: Final output dimension of transformed encodings

        Raises:
            AssertionError: If base_dim is not even
        """
        super().__init__(base_dim)
        self.linear1 = nn.Linear(base_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate and transform positional encodings.

        Args:
            timesteps: Input tensor of timestep values
                       Shape: (batch_size,)

        Returns:
            Transformed positional encodings
            Shape: (batch_size, output_dim)

        Processing:
            1. Generate base sinusoidal encodings (via parent class)
            2. Apply first linear transform + SiLU activation
            3. Apply second linear transform
        """
        x = super().forward(timesteps).to(timesteps.device)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x


class ResnetBlock(nn.Module):
    """A residual block with adaptive conditioning on timestep and class embeddings.

    Implements a modified ResNet block architecture that:
    1. Applies group normalization and nonlinear activations
    2. Incorporates timestep (t) and class label (y) embeddings via projection
    3. Uses residual connection with channel matching

    The block follows this computation flow:
    h = norm1(x) → silu → conv1 → multiply(t_emb) → add(y_emb)
    h = norm2(h) → silu → dropout → conv2
    out = h + residual(x)

    Attributes:
        norm1 (nn.GroupNorm): First group normalization layer
        norm2 (nn.GroupNorm): Second group normalization layer
        conv1 (nn.Conv2d): First 3x3 convolution layer
        conv2 (nn.Conv2d): Second 3x3 convolution layer
        linear_t (nn.Linear): Timestep embedding projection
        linear_y (nn.Linear): Class embedding projection
        dropout (nn.Dropout): Optional dropout layer
        linear_res (nn.Module): Residual connection (1x1 conv or identity)
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_norm_groups: int,  # 32
        embedding_dim: int,  # 512
        dropout: float = 0.1,
    ):
        """Initialize the ResnetBlock module.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            num_norm_groups: Number of groups for GroupNorm (must divide channels)
            embedding_dim: Dimension of timestep/class embeddings
            dropout: Dropout probability (default: 0.1)

        Raises:
            AssertionError: If num_norm_groups doesn't evenly divide in_channel or out_channel
        """
        super().__init__()
        assert (in_channel % num_norm_groups == 0)
        assert (out_channel % num_norm_groups == 0)

        self.norm1 = nn.GroupNorm(num_norm_groups, in_channel, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_norm_groups, out_channel, eps=1e-6)

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding="same")
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding="same")

        self.linear_t = nn.Linear(embedding_dim, out_channel)
        self.linear_y = nn.Linear(embedding_dim, out_channel)

        self.dropout = nn.Dropout(p=dropout)

        if in_channel != out_channel:
            # same as linear over channels
            self.linear_res = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.linear_res = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        y_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch_size, in_channel, height, width)
            t_emb: Timestep embeddings of shape (batch_size, embedding_dim)
            y_emb: Class embeddings of shape (batch_size, embedding_dim)

        Returns:
            Output tensor of shape (batch_size, out_channel, height, width)

        Processing steps:
            1. Project and reshape timestep/class embeddings
            2. Apply first normalization → activation → convolution
            3. Scale by timestep and shift by class embedding
            4. Apply second normalization → activation → dropout → convolution
            5. Add residual connection (with channel projection if needed)
        """
        t = F.silu(t_emb)
        t = self.linear_t(t)
        t = rearrange(t, 'b c -> b c 1 1')

        y = F.silu(y_emb)
        y = self.linear_y(y)
        y = rearrange(y, 'b c -> b c 1 1')

        z = self.norm1(x)  # preserve input x
        z = F.silu(z)
        z = self.conv1(z)

        z = t * z + y

        z = self.norm2(z)
        z = F.silu(z)
        z = self.dropout(z)
        z = self.conv2(z)

        x = self.linear_res(x)
        z = z + x
        return z


class AttentionBlock(nn.Module):
    """Implements a self-attention mechanism with skip connection for 2D feature maps.

    This block performs the following operations:
    1. Applies GroupNorm normalization
    2. Flattens spatial dimensions and computes query, key, value projections
    3. Calculates scaled dot-product attention
    4. Applies final linear transformation
    5. Adds skip connection to original input

    The attention mechanism follows the standard transformer architecture:
    Attention(Q, K, V) = softmax(QK^T/√d)V

    Attributes:
        norm (nn.GroupNorm): Group normalization layer
        w_q (nn.Linear): Query projection layer
        w_k (nn.Linear): Key projection layer
        w_v (nn.Linear): Value projection layer
        linear (nn.Linear): Final output projection (no bias)
    """

    def __init__(self,
                 channel: int,
                 num_norm_groups: int  # 32
                 ):
        """Initializes the AttentionBlock module.

        Args:
            channel: Number of input/output channels
            num_norm_groups: Number of groups for GroupNorm (must divide channel)

        Raises:
            AssertionError: If channel is not divisible by num_norm_groups
        """
        super().__init__()
        assert channel % num_norm_groups == 0, \
            f"Channel dimension {channel} must be divisible by num_norm_groups {num_norm_groups}"

        self.norm = nn.GroupNorm(num_norm_groups, channel)
        self.w_q = nn.Linear(channel, channel)
        self.w_k = nn.Linear(channel, channel)
        self.w_v = nn.Linear(channel, channel)
        self.linear = nn.Linear(channel, channel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention block.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of same shape as input with attention features added

        Processing steps:
            1. Apply group normalization
            2. Rearrange to flattened spatial dimensions (channel last)
            3. Compute query, key, value projections
            4. Calculate scaled dot-product attention weights
            5. Apply attention to values
            6. Project back to original channel dimension
            7. Rearrange back to spatial dimensions
            8. Add skip connection to original input
        """
        z = self.norm(x)
        z = rearrange(z, 'b c h w -> b (h w) c')  # switch channels with pixels
        Q = self.w_q(z)
        K = self.w_k(z)
        V = self.w_v(z)

        attention = torch.einsum('bic,bjc->bij', Q, K)  # QK^T
        attention = attention / x.size(1)**0.5  # scale by sqrt(d_k)
        attention = F.softmax(attention, dim=-1)  # attention weights

        z = torch.einsum('bij,bjc->bic', attention, V)
        z = self.linear(z)

        z = rearrange(z, 'b (h w) c -> b c h w', h=x.size(2),
                      w=x.size(3))  # restore channels vs pixels
        z = z + x
        return z


class ResnetAndAttention(nn.Module):
    """A combined ResNet and Attention block module.

    This module sequentially applies:
    1. A ResNet block with timestep and class conditioning
    2. A self-attention mechanism

    The architecture enables both local feature processing (via ResNet) 
    and global context modeling (via attention). The ResNet block handles
    timestep and class embeddings, while the attention block captures
    long-range spatial dependencies.

    Attributes:
        resnet (ResnetBlock): The ResNet block component
        attention (AttentionBlock): The attention block component
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_norm_groups: int,  # 32
        embedding_dim: int,  # 512
    ):
        """Initializes the ResnetAndAttention module.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            num_norm_groups: Number of groups for GroupNorm layers (must divide channels)
            embedding_dim: Dimension of timestep and class embeddings

        Note:
            Typical values:
            - num_norm_groups: 32 (default)
            - embedding_dim: 512 (default)
        """
        super().__init__()
        self.resnet = ResnetBlock(
            in_channel,
            out_channel,
            num_norm_groups,
            embedding_dim,
        )
        self.attention = AttentionBlock(
            out_channel,
            num_norm_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        y_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the combined ResNet and Attention blocks.

        Args:
            x: Input feature map
                Shape: (batch_size, in_channel, height, width)
            t_emb: Timestep embeddings 
                Shape: (batch_size, embedding_dim)
            y_emb: Class embeddings
                Shape: (batch_size, embedding_dim)

        Returns:
            Processed feature map with same spatial dimensions as input
            Shape: (batch_size, out_channel, height, width)

        Processing steps:
            1. Apply ResNet block with timestep and class conditioning
            2. Apply self-attention to capture global dependencies
            3. Return processed features
        """
        out = self.resnet(x, t_emb, y_emb)
        out = self.attention(out)
        return out


class DownSample(nn.Module):
    """
    A down sampling module.

    This module downsamples the input tensor by a factor of 2 using a 
    convolution-based method.
    """

    def __init__(self, channels: int):
        super().__init__()
        """Initializes the DownSample module.

        Args:
            channels: Number of channels
        """

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    """
    A up sampling module.

    This module upsamples the input tensor by a factor of 2 using nearest 
    neighbor interpolation. It then applies a convolution to the upsampled 
    tensor.
    """

    def __init__(self, channels: int):
        super().__init__()
        """Initializes the UpSample module.

        Args:
            channel: Number of input channels
        """

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.conv(out)

        return out


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
            ResnetAndAttention(c, c * 2, num_groups, emb_dim),
            ResnetAndAttention(c * 2, c * 2, num_groups, emb_dim),
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
            ResnetAndAttention(c * 4, c * 2, num_groups, emb_dim),
            ResnetAndAttention(c * 4, c * 2, num_groups, emb_dim),
            ResnetAndAttention(c * 3, c * 2, num_groups, emb_dim),
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
        skip_connections.append(out)

        for block in self.down_blocks:
            match block:
                case DownSample():
                    out = block(out)
                case ResnetBlock() | ResnetAndAttention():
                    out = block(out, t_embs, y_emb)
                case _:
                    raise ValueError("Unknown block type")
            skip_connections.append(out)

        for block in self.mid_blocks:
            match block:
                case ResnetBlock():
                    out = block(out, t_embs, y_emb)
                case AttentionBlock():
                    out = block(out)
                case _:
                    raise ValueError("Unknown block type")

        for block in self.up_blocks:
            match block:
                case UpSample():
                    out = block(out)
                case ResnetBlock() | ResnetAndAttention():
                    skip = skip_connections.pop()
                    out = torch.cat((out, skip), dim=1)
                    out = block(out, t_embs, y_emb)
                case _:
                    raise ValueError("Unknown block type")

        assert not skip_connections, "Not all skip connections were used"

        out = self.norm(out)
        out = F.silu(out)
        out = self.up_conv(out)

        return out


if __name__ == "__main__":
    unet = UNet(source_channel=3, unet_base_channel=64,
                num_groups=32).to('mps')
    x = torch.randn(1, 3, 256, 256).to('mps')
    t = torch.tensor([0]).to('mps')
    y_emb = torch.randn(1, 256).to('mps')
    out = unet(x, t, y_emb)
    print(out.shape)
