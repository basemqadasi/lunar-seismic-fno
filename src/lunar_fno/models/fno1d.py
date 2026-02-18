import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, inp, weights):
        return torch.einsum("bix,iox->box", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        x = torch.fft.irfftn(out_ft, s=x.size(-1))
        return x


class SimpleBlock1d(nn.Module):
    def __init__(self, modes1: int, width: int):
        super().__init__()
        self.modes1 = modes1
        self.width = width

        self.fc0 = nn.Linear(1, self.width)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.width)

        self.fc10 = nn.Linear(self.width, 64 * 2)
        self.dropout10 = nn.Dropout(0.5)
        self.fc20 = nn.Linear(64 * 2, 64 * 2)
        self.dropout20 = nn.Dropout(0.5)
        self.fc30 = nn.Linear(64 * 2, 1)

        self.bn0 = nn.BatchNorm1d(self.width)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x)
        x = self.bn0(x1 + x2)
        x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))

        x = x.view(x.size(0), -1, x.size(-1)).max(dim=1).values
        x = self.dropout10(F.relu(self.fc10(x)))
        x = self.dropout20(F.relu(self.fc20(x)))
        x = self.fc30(x)
        return torch.sigmoid(x)
