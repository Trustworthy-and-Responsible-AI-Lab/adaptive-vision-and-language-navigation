import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Modules from Sim to Real VLN paper
"""
class RadialPad(nn.Module):
    """Circular padding of heading, replication padding of range."""

    def __init__(self, padding):
        """
        Args:
            padding (int): the size of the padding."""
        super(RadialPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        # x has shape [batch, channels, range_bins, heading_bins]
        x1 = F.pad(x, [0, 0, self.padding, self.padding], mode="replicate")
        x2 = F.pad(x1, [self.padding, self.padding, 0, 0], mode="circular")
        return x2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            RadialPad(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            RadialPad(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Radial padding
        x1 = F.pad(
            x1, [0, 0, diffY // 2, diffY - diffY // 2], mode="replicate"
        )
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 0, 0], mode="circular")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class ScanOnlyNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, ch=64, bilinear=True, dropout=0.2
    ):
        super(ScanOnlyNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, ch)
        self.down1 = Down(ch, 2 * ch)
        self.down2 = Down(2 * ch, 4 * ch)
        self.drop = nn.Dropout(p=dropout)
        # self.down3 = Down(8 * ch, 8 * ch)
        self.down3 = Down(4 * ch, 8 * ch)
        self.down4 = Down(8 * ch, 8 * ch)
        self.up1 = Up(16 * ch, 4 * ch, bilinear)
        # self.up2 = Up(12 * ch, 2 * ch, bilinear)
        self.up2 = Up(8 * ch, 2 * ch, bilinear)
        self.up3 = Up(4 * ch, ch, bilinear)
        self.up4 = Up(2 * ch, ch, bilinear)
        self.outc = OutConv(ch, n_classes)

    def forward(self, scans):
        # scans [batch_size, 2, 24, 48] - channels range and return type

        x1 = self.inc(scans)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits