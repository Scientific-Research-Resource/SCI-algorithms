"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:
    This is the code of specific parts for Snapshot Compressive Imaging reconstruction in recurrent convolution neural network

Citation:
    The code prepares for ECCV 2020

Contact:
    Ziheng Cheng
    zhcheng@stu.xidian.edu.cn
    Xidian University, Xi'an, China

    Bo Chen
    bchen@mail.xidian.edu.cn
    Xidian University, Xi'an, China

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Copyright (c), 2020, Ziheng Cheng
zhcheng@stu.xidian.edu.cn

"""

import torch
import torch.nn as nn


class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x


class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            # nn.Conv2d(20, 40, 5, stride=2, padding=2),
            # nn.Conv2d(40, out_ch, 5, stride=2, padding=2),
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            # nn.ConvTranspose2d(in_ch, 40, 3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(40, 20, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class first_frame_5(nn.Module):
    # 输入meas concat mask
    # 3 下采样

    def __init__(self):
        super(first_frame_5, self).__init__()
        self.conv1 = nn.Conv2d(11, 32, kernel_size=5, stride=1, padding=2)                        #  nn.Conv2d((fn+1), 32,...
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.conv51 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.LeakyReLU(inplace=True)
        self.conv52 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52 = nn.LeakyReLU(inplace=True)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.res_part1 = res_part(128, 128)
        self.res_part2 = res_part(128, 128)
        self.res_part3 = res_part(128, 128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(inplace=True)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        # self.attention1 = self_attention(128)

    def forward(self, x, mask, batch_size, meas_re):
        # x = torch.unsqueeze(x, 1)

        maskt = mask.expand([batch_size, 10, 256, 256])                          # [batch_size, fn, 256, 256]
        maskt = maskt.mul(meas_re)

        # x1 = x.repeat([1, maskt.shape[1], 1, 1])
        # xt = torch.mul(x1, maskt)
        # xt = torch.cat([x, xt], dim=1)

        # xt = torch.cat([x, maskt], dim=1)
        xt = torch.cat([meas_re, maskt], dim=1)
        data = xt
        # data = maskt
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        # out = self.attention1(out, batch_size, 64)
        out = self.res_part1(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.res_part2(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.res_part3(out)
        # out = self.attention1(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.relu52(out)
        out = self.conv6(out)

        return out
