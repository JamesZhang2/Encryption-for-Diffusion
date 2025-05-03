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
    def __init__(self, linear: nn.Linear, fhe, scale=100.0):
        super().__init__()
        self.in_dim = linear.in_dim
        self.out_dim = linear.out_dim
        self.scale = scale
        self.true_bias = linear.linear.bias
        self.fhe = fhe
        self.weight = scale_round(linear.linear.weight, scale)

    def forward(self, c_x, in_scale):  # assume input is encrypted
        if len(c_x.shape) == 1:
            c_x = rearrange(c_x, "i -> 1 i")
        assert (len(c_x.shape) == 2)
        out = self.fhe.matrix_product(self.weight, c_x.T).T
        bias = self.fhe.enc_vec(scale_round(
            self.true_bias, in_scale * self.scale))
        out = self.fhe.add_matrix_vector(out, bias)
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
        self.residual = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.non_lin = lambda x: x**2 + x

    def forward(self, x):
        y = self.non_lin(self.linear1(x))
        print(torch.max(y), torch.min(y))
        y = self.non_lin(self.linear2(y))
        y = F.dropout(y, p=self.p, training=self.training)
        return y + self.residual(x)


class MultiLayerNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Define Layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # block = ResidualBlock(10, 20)
    block = FHELinear(Linear(10, 20))
