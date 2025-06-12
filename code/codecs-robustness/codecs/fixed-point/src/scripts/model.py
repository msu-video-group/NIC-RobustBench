import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, a_l=0):
        super(ResBlock,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)


    def forward(self,x):
        x1 = self.conv2(F.relu(self.conv1(x)))
        out = x + x1
        return out


class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.SU_Res1 = ResBlock(64, 64, 3, 1, 1)
        self.SU_Res2 = ResBlock(64, 64, 3, 1, 1)
        self.SU_Res3 = ResBlock(64, 64, 3, 1, 1)

        self.conv2 = nn.Conv2d(64, 288, 3, 1, 1)
        self.PixelShuffle = nn.PixelShuffle(3)
        self.conv3 = nn.Conv2d(32, 3, 3, 2, 1)

    def forward(self,x):
        out1 = F.relu(self.conv1(x))
        out2 = self.SU_Res1(out1)
        out3 = self.SU_Res2(out2)
        out4 = self.SU_Res3(out3)

        out5 = F.relu(self.conv2(out1 + out4))
        out6 = self.PixelShuffle(out5)
        out7 = self.conv3(out6)
        return out7
 
