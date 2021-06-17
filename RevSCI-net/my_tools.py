import torch
import torch.nn as nn


def split_feature(x):
    l = x.shape[1]
    x1 = x[:, 0:l // 2, ::]
    x2 = x[:, l // 2:, ::]
    return x1, x2


def split_n_features(x, n):
    x_list = list(torch.chunk(x, n, dim=1))
    return x_list


class rev_part(nn.Module):

    def __init__(self, in_ch):
        super(rev_part, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.g1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = split_feature(x)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.g1(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def reverse(self, y):
        y1, y2 = split_feature(y)
        x2 = y2 - self.g1(y1)
        x1 = y1 - self.f1(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class f_g_layer(nn.Module):
    def __init__(self, ch):
        super(f_g_layer, self).__init__()
        self.nn_layer = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.nn_layer(x)
        return x


class rev_3d_part1(nn.Module):

    def __init__(self, in_ch, n):
        super(rev_3d_part1, self).__init__()
        self.f = nn.ModuleList()
        self.n = n
        self.ch = in_ch
        for i in range(n):
            self.f.append(f_g_layer(in_ch // n))

    def forward(self, x):
        x = split_n_features(x, self.n)
        y1 = x[-1] + self.f[0](x[0])
        y = y1
        for i in range(1, self.n):
            y1 = x[(self.n - 1 - i)] + self.f[i](y1)
            y = torch.cat([y, y1], dim=1)
        return y

    def reverse(self, y):
        y = split_n_features(y, self.n)
        for i in range(1, self.n):
            x1 = y[self.n - i] - self.f[self.n - i](y[self.n - i - 1])
            if i == 1:
                x = x1
            else:
                x = torch.cat([x, x1], dim=1)
        x1 = y[0] - self.f[0](x[:, 0:(self.ch // self.n), ::])
        x = torch.cat([x, x1], dim=1)
        return x


class rev_3d_part(nn.Module):

    def __init__(self, in_ch):
        super(rev_3d_part, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )
        self.g1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = split_feature(x)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.g1(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def reverse(self, y):
        y1, y2 = split_feature(y)
        x2 = y2 - self.g1(y1)
        x1 = y1 - self.f1(x2)
        x = torch.cat([x1, x2], dim=1)
        return x
