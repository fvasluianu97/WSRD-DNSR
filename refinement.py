import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, x_in, x_out, k):
        super(Residual, self).__init__()

        block = [
            nn.ReflectionPad2d(k // 2),
            nn.Conv2d(x_in, x_out, k),
            nn.InstanceNorm2d(x_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(k // 2),
            nn.Conv2d(x_in, x_in, k),
            nn.InstanceNorm2d(x_in)
        ]

        self.block = nn.Sequential(*block)
        self.res_conv = nn.Conv2d(x_in, x_out, kernel_size=1, bias=True)

    def forward(self, x):
        return self.res_conv(x) + self.block(x)


class RefinementBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz=3):
        super(RefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 2, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_ch * 2, out_ch, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)

    def forward(self, x):
        x = self.op1(self.conv1(x))
        x = self.conv2(x)
        return x
