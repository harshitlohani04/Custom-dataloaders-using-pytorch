import torch
import torch.nn as nn
import torch.nn.functional as F
from index.IdxPool import IdxPool


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.pool = IdxPool(kernel_size = 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self.pool,

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool,

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            self.pool,

            nn.Conv2d(4096, 8192, kernel_size=3, padding=1),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True),
            nn.Conv2d(8192, 8192, kernel_size=3, padding=1),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True),
            self.pool,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(8192*4, 8192*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8192*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8192*8, 8192*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8192*8),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.ConvTranspose2d(8192*8, 8192*4, kernel_size=3, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, x):
        print("Running the forward method")
        enc1 = self.encoder[0:4](x)
        x = self.encoder[4](enc1)
        print("enc 1 layer executed successfully")
        print(enc1.shape)
        enc2 = self.encoder[5:9](x)
        x = self.encoder[9](enc2)
        print("enc 2 layer executed successfully")
        print(enc2.shape)
        enc3 = self.encoder[10:14](x)
        x = self.encoder[14](enc3)
        print("enc 3 layer executed successfully")
        print(enc3.shape)
        enc4 = self.encoder[15:19](x)
        x = self.encoder[19](enc4)
        print("enc 4 layer executed successfully")
        print(enc4.shape)

        bottleneck = self.bottleneck(x)
        print("Bottleneck successfully executed")
        dec4 = self.upconv4(bottleneck)
        print("UPCONV4 block executed successfully")
        print(dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        x = self.decoder4(dec4)

        dec3 = self.upconv3(x)
        print("UPCONV3 block executed successfully")
        print(dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        x = self.decoder3(dec3)

        dec2 = self.upconv2(x)
        print("UPCONV2 block executed successfully")
        print(dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        x = self.decoder2(dec2)

        dec1 = self.upconv1(x)
        print("UPCONV1 block executed successfully")
        print(dec1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        x = self.decoder1(dec1)

        return x

