"""The submitted U-Net, preserving layer names and checkpoint mathematics."""

import torch
from torch import nn


def CBR2d(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, n_classes: int = 13, dropout_rate: float = 0.5):
        super().__init__()
        if n_classes < 1:
            raise ValueError("n_classes must be positive")
        self.enc1_1 = CBR2d(3, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout_rate)
        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout_rate)
        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(dropout_rate)
        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(dropout_rate)
        self.enc5_1 = CBR2d(512, 1024)
        self.dec5_1 = CBR2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, stride=2, output_padding=1)
        self.dec4_2 = CBR2d(1024, 512)
        self.dec4_1 = CBR2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, stride=2, output_padding=1)
        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, stride=2, output_padding=1)
        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, stride=2, output_padding=1)
        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)
        self.fc = nn.Conv2d(64, n_classes, 1)

    @staticmethod
    def _join(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Crop the decoder result exactly as in the original implementation."""
        height, width = skip.shape[-2:]
        top = (upsampled.shape[-2] - height) // 2
        left = (upsampled.shape[-1] - width) // 2
        cropped = upsampled[:, :, top:top + height, left:left + width]
        return torch.cat((cropped, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 16:
            raise ValueError("Expected [N, 3, H, W] input with H and W at least 16")
        e1 = self.enc1_2(self.enc1_1(x))
        e2 = self.enc2_2(self.enc2_1(self.drop1(self.pool1(e1))))
        e3 = self.enc3_2(self.enc3_1(self.drop2(self.pool2(e2))))
        e4 = self.enc4_2(self.enc4_1(self.drop3(self.pool3(e3))))
        bottleneck = self.enc5_1(self.drop4(self.pool4(e4)))
        d5 = self.dec5_1(bottleneck)
        d4 = self.dec4_1(self.dec4_2(self._join(self.unpool4(d5), e4)))
        d3 = self.dec3_1(self.dec3_2(self._join(self.unpool3(d4), e3)))
        d2 = self.dec2_1(self.dec2_2(self._join(self.unpool2(d3), e2)))
        d1 = self.dec1_1(self.dec1_2(self._join(self.unpool1(d2), e1)))
        return self.fc(d1)
