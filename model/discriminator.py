from model.blocks import CDkBLOCK, CkBLOCK, CkBLOCK_Transpose
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Discriminator, self).__init__()
        self.c64 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      bias=False,
                      padding = 1,
                      padding_mode='reflect'
                     ),
            nn.LeakyReLU(0.2))
        self.c128 = CkBLOCK(64, 128, padding = 1)
        self.c256 = CkBLOCK(128, 256, padding = 1)
        self.c512 = CkBLOCK(256, 512, padding = 1, stride = 1)
        self.fin_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding = 1, padding_mode='reflect')
        
    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.c64(x)
        x = self.c128(x)
        x = self.c256(x)
        x = self.c512(x)
        x = self.fin_conv(x)
        return x
        