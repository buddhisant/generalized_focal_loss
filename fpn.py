import torch
import config as cfg
import torch.nn.functional as F

class PyramidFeatures(torch.nn.Module):
    """fpn网络"""
    def __init__(self):
        super(PyramidFeatures, self).__init__()

        self.c5_channels = cfg.c5_channels
        self.c4_channels = cfg.c4_channels
        self.c3_channels = cfg.c3_channels
        self.fpn_channels = cfg.fpn_channels

        self.P5_1 = torch.nn.Conv2d(self.c5_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.P5_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.P4_1 = torch.nn.Conv2d(self.c4_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.P4_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.P3_1 = torch.nn.Conv2d(self.c3_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.P3_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.P6 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.P7 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=2, padding=1, bias=True)

        for m in self.modules():
            if(isinstance(m,torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight,gain=1)
                torch.nn.init.constant_(m.bias,0)

    def forward(self, x):
        c3, c4, c5 = x

        c3_shape=c3.shape[2:]
        c4_shape=c4.shape[2:]

        p5=self.P5_1(c5)
        p5_upsampled=F.interpolate(p5,size=c4_shape,mode="nearest")
        p5=self.P5_2(p5)

        p4=self.P4_1(c4)
        p4=p4+p5_upsampled
        p4_upsampled=F.interpolate(p4,size=c3_shape,mode="nearest")
        p4=self.P4_2(p4)

        p3=self.P3_1(c3)
        p3=p3+p4_upsampled
        p3=self.P3_2(p3)

        p6=self.P6(p5)

        p7=self.P7(p6)

        return [p3,p4,p5,p6,p7]
