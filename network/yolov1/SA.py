import math
import torch
import torch.nn as nn


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v


class Channel_Attention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(channels, channels // 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 16)
        self.conv_2 = nn.Conv1d(channels // 16, channels // 16, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(channels // 16, channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c, h, w = x.size()
        v = self.avg_pool(x).view(bs, c, 1, 1)
        v = self.bn1(self.conv_1(v)).view(bs // 8, 8, c // 16)  # [batch, segment, channel//16]
        v = self.relu(self.conv_2(v.transpose(-1, -2))).transpose(-1, -2).contiguous().view(bs, c // 16, 1, 1)
        v = self.bn3(self.conv_3(v))

        v = self.sigmoid(v) - 0.1
        return x * v + x


class SpatioTemporal(nn.Module):
    def __init__(self, channels, n_segment):
        super(SpatioTemporal, self).__init__()
        self.in_channels = channels
        self.n_segment = n_segment

        self.channel_pool = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1))

        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv_diff = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 3, 3), padding=(0, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return self._diff(x) * x

    def _diff(self, x):
        bs, c, h, w = x.size()
        x_ = self.channel_pool(x).view(bs // self.n_segment, self.n_segment, h, w)

        x_left = torch.zeros_like(x_)
        x_left[:, 1:] = x_[:, :-1]
        x_left = (x_ - x_left).view(bs, 1, h, w)

        x_right = torch.zeros_like(x_)
        x_right[:, :-1] = x_[:, 1:]
        x_right = (x_right - x_).view(bs, 1, h, w)

        x_diff = torch.cat([x_left, x_right], dim=1).view(bs, 1, 2, h, w)

        x_ = self.conv3d(x_.unsqueeze(1)).view(bs, 1, h, w)
        x_ = self.sigmoid(x_)

        x_diff = self.conv_diff(x_diff).view(bs, 1, h, w)
        x_diff = self.sigmoid(x_diff)

        return x_ * x_diff