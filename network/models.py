import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fhe import *


def scale_round(x, scale):
    return torch.round(x * scale).long()


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        return x


class FHELinear(nn.Module):
    def __init__(self, linear: nn.Linear, scale=100.0):
        super().__init__()
        self.in_dim = linear.in_dim
        self.out_dim = linear.out_dim
        self.scale = scale
        self.true_bias = linear.linear.bias
        self.weight = enc_mat(scale_round(linear.linear.weight, scale))

    def forward(self, c_x, in_scale):  # assume input is encrypted
        if len(c_x.shape) == 1:
            c_x = rearrange(c_x, "i -> 1 i")
        assert (len(c_x.shape) == 2)
        out = matrix_product(self.weight, c_x.T).T
        bias = enc_vec(scale_round(
            self.true_bias, in_scale * self.scale))
        out = add_matrix_vector(out, bias)
        return out, in_scale * self.scale


class ResidualBlock(nn.Module):
    "A residual block of 2 linear layers, connected by quadratic non-linearities"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = dropout
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        y = self.linear1(x)**2
        y = self.linear2(y)**2
        y = F.dropout(y, p=self.p, training=self.training)
        return y + x


if __name__ == "__main__":
    # block = ResidualBlock(10, 20)
    block = FHELinear(Linear(10, 20))
