"""
U-Net архитектура для семантической сегментации
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Двойная свёртка: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d -> Concatenate -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding если размеры не совпадают
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Выходной слой
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net архитектура
    
    Args:
        n_channels: количество входных каналов (3 для RGB)
        n_classes: количество выходных классов (1 для бинарной сегментации)
        bilinear: использовать билинейную интерполяцию вместо ConvTranspose
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder с skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) - улучшенная версия U-Net
    с дополнительными skip connections
    """
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.conv0_0 = DoubleConv(n_channels, 64)
        self.conv1_0 = Down(64, 128)
        self.conv2_0 = Down(128, 256)
        self.conv3_0 = Down(256, 512)
        self.conv4_0 = Down(512, 1024)
        
        # Nested skip pathways
        self.conv0_1 = DoubleConv(64 + 128, 64)
        self.conv1_1 = DoubleConv(128 + 256, 128)
        self.conv2_1 = DoubleConv(256 + 512, 256)
        self.conv3_1 = DoubleConv(512 + 1024, 512)
        
        self.conv0_2 = DoubleConv(64 * 2 + 128, 64)
        self.conv1_2 = DoubleConv(128 * 2 + 256, 128)
        self.conv2_2 = DoubleConv(256 * 2 + 512, 256)
        
        self.conv0_3 = DoubleConv(64 * 3 + 128, 64)
        self.conv1_3 = DoubleConv(128 * 3 + 256, 128)
        
        self.conv0_4 = DoubleConv(64 * 4 + 128, 64)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolution
        if self.deep_supervision:
            self.final1 = OutConv(64, n_classes)
            self.final2 = OutConv(64, n_classes)
            self.final3 = OutConv(64, n_classes)
            self.final4 = OutConv(64, n_classes)
        else:
            self.final = OutConv(64, n_classes)
    
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


def get_unet_model(model_type='unet', n_channels=3, n_classes=1, pretrained=False):
    """
    Фабрика для создания U-Net моделей
    
    Args:
        model_type: 'unet', 'unet++', 'unet_bilinear'
        n_channels: количество входных каналов
        n_classes: количество классов
        pretrained: использовать ли предобученные веса (для encoder)
    """
    if model_type == 'unet':
        model = UNet(n_channels, n_classes, bilinear=False)
    elif model_type == 'unet_bilinear':
        model = UNet(n_channels, n_classes, bilinear=True)
    elif model_type == 'unet++':
        model = UNetPlusPlus(n_channels, n_classes, deep_supervision=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
