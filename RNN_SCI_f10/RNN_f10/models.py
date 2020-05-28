"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:
    This is the model class code for Snapshot Compressive Imaging reconstruction in recurrent convolution neural network

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

from my_tools import *


class forward_rnn(nn.Module):

    def __init__(self):
        super(forward_rnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(50, 1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(2, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 80, 3, stride=2, padding=1),
            nn.Conv2d(80, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 40, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(40, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(50, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(50, 50)
        self.res_part2 = res_part(50, 50)

    def forward(self, xt1, meas, mask, batch_size, h, mode, meas_re):
        ht = h
        xt = xt1

        out = xt1
        for i in range(9):                                                                 # range(fn-1):
            d1 = torch.zeros(batch_size, 256, 256).cuda()
            d2 = torch.zeros(batch_size, 256, 256).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[ii, :, :], out[:, ii, :, :])
            for ii in range(i + 2, 10):                                                    # range(i + 2, fn):
                d2 = d2 + torch.mul(mask[ii, :, :], torch.squeeze(meas_re))
            x1 = self.conv_x(torch.cat([meas_re, torch.unsqueeze(meas - d1 - d2, 1)], dim=1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x1, x2], dim=1)

            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)
            out = torch.cat([out, xt], dim=1)

        return out, ht


class cnn1(nn.Module):
    # 输入meas concat mask
    # 3 下采样

    def __init__(self):
        super(cnn1, self).__init__()
        self.conv1 = nn.Conv2d(11, 32, kernel_size=5, stride=1, padding=2)                   # nn.Conv2d( (fn+1), 32, ...  
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

    def forward(self, x, mask, batch_size, meas_re):
        maskt = mask.expand([batch_size, 10, 256, 256])
        maskt = maskt.mul(meas_re)
        xt = torch.cat([meas_re, maskt], dim=1)
        data = xt
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.res_part1(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.res_part2(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.res_part3(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.relu52(out)
        out = self.conv6(out)

        return out


class backrnn(nn.Module):

    def __init__(self):
        super(backrnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(50, 1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(2, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 80, 3, stride=2, padding=1),
            nn.Conv2d(80, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 40, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(40, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(50, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(50, 50)
        self.res_part2 = res_part(50, 50)
        # self.res_part3 = res_part(80, 80)

        # self.first = first_frame()

    def forward(self, xt8, meas, mask, batch_size, h, mode, meas_re):
        ht = h

        xt = xt8
        xt = torch.unsqueeze(xt, 1)

        out = torch.zeros(batch_size, 10, 256, 256).cuda()              # (batch_size, fn, 256, 256)
        out[:, 9, :, :] = xt[:, 0, :, :]                                # out[:, fn-1, :, :] = xt[:, 0, :, :]
        for i in range(9):                                              # range(fn-1):
            d1 = torch.zeros(batch_size, 256, 256).cuda()
            d2 = torch.zeros(batch_size, 256, 256).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[9 - ii, :, :], out[:, 9 - ii, :, :].clone())                # mask[(fn-1) - ii, :, :]  out...
            for ii in range(i + 2, 10):
                d2 = d2 + torch.mul(mask[9 - ii, :, :], torch.squeeze(meas_re))                      # mask[(fn-1) - ii, :, :]
            x1 = self.conv_x(torch.cat([meas_re, torch.unsqueeze(meas - d1 - d2, 1)], dim=1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x1, x2], dim=1)

            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)

            out[:, 8 - i, :, :] = xt[:, 0, :, :]                         # out[:, (fn-2) - i, :, :]

        return out
