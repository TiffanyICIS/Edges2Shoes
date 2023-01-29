import torch.nn as nn

class CkBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, padding = 0, stride = 2):
        super(CkBLOCK, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding = padding,
                      padding_mode='reflect',
                      bias=False
                     ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x


class CkBLOCK_Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, padding = 0, stride = 2):
        super(CkBLOCK_Transpose, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x


class CDkBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, padding = 0, stride = 2):
        super(CDkBLOCK, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout2d(p = 0.5)
    def forward(self, x):
        x = self.block(x)
        return self.dropout(x)

