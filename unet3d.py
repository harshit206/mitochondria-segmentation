import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, second_conv_output):
        super(DoubleConvBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, second_conv_output, kernel_size=3, padding=1),
            nn.BatchNorm3d(second_conv_output),
            nn.ReLU(inplace=True)        
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, second_conv_output):
        super(DownBlock, self).__init__()

        self.maxpool_3d = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels, second_conv_output)
        )
    
    def forward(self, x):
        return self.maxpool_3d(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.up_sample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self,x):
        return self.up_sample(x)


class Unet3d(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(Unet3d, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes


        self.double_conv = DoubleConvBlock(n_channels, 32, 64)
        self.down1 = DownBlock(64,64,128)
        self.down2 = DownBlock(128,128,256)
        self.down3 = DownBlock(256,256,512)

        self.up1= UpBlock(512,512)   
        self.up_conv1 = DoubleConvBlock(768,256,256)
        self.up2 = UpBlock(256,256)  
        self.up_conv2 = DoubleConvBlock(384,128,128)
        self.up3 = UpBlock(128,128)  
        self.up_conv3 = DoubleConvBlock(192,64,64)

        
        self.final_conv = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):

        # encoder 
        x1 = self.double_conv(x)  
        
        x2 = self.down1(x1)       
        
        x3 = self.down2(x2)      
        
        x4 = self.down3(x3)       
        

        #decoder
        y1 = self.up1(x4)
        c1 = torch.cat([y1,x3], dim=1)  

        y2 = self.up_conv1(c1)

        y3 = self.up2(y2)
        c2 = torch.cat([y3,x2], dim=1)  

        y4 = self.up_conv2(c2)

        y5 = self.up3(y4)
        c3 = torch.cat([y5,x1], dim=1)

        y6 = self.up_conv3(c3)


        
        output = self.final_conv(y6)
        return output




