import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_tensor(decoder_tensor, encoder_tensor):
    """
    Crops the decoder tensor to match the spatial dimensions of the encoder tensor.

    Args:
        decoder_tensor (torch.Tensor): The tensor from the decoder (output of upconv).
        encoder_tensor (torch.Tensor): The tensor from the encoder (before pooling).
    
    Returns:
        torch.Tensor: Cropped decoder tensor to match the encoder tensor.
    """
    _, _, h_decoder, w_decoder = decoder_tensor.size()
    _, _, h_encoder, w_encoder = encoder_tensor.size()
    
    # Compute the difference in height and width
    diff_h = h_encoder - h_decoder
    diff_w = w_encoder - w_decoder
    
    # If there's a difference, crop the decoder tensor
    if diff_h > 0:
        encoder_tensor = encoder_tensor[:, :, :h_decoder, :]
    if diff_w > 0:
        encoder_tensor = encoder_tensor[:, :, :, :w_decoder]
    
    return encoder_tensor


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2)
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

