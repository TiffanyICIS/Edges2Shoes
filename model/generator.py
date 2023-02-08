from model.blocks import CDkBLOCK, CkBLOCK, CkBLOCK_Transpose
import torch
import torch.nn as nn
from model.config import DEVICE


# class Generator(nn.Module):
#     def __init__(self, in_channels = 3):
#         super(Generator, self).__init__()
#         # encoder        
#         self.c64_enc = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=64,
#                       kernel_size=4,
#                       stride=2,
#                       padding = 1,
#                       padding_mode='reflect',
#                       bias=False
#                      ),
#             nn.LeakyReLU(0.2)
#         ) # 256 -> 128
#         self.c128_enc = CkBLOCK(64, 128, padding = 1) # 128 -> 64
#         self.c256_enc = CkBLOCK(128, 256, padding = 1) # 64 -> 32
#         self.c512_1_enc = CkBLOCK(256, 512, padding = 1) # 32 -> 16
#         self.c512_2_enc = CkBLOCK(512, 512, padding = 1) # 16 -> 8
#         self.c512_3_enc = CkBLOCK(512, 512, padding = 1) # 8 -> 4
#         self.c512_4_enc = CkBLOCK(512, 512, padding = 1) # 4 -> 2
#         #bottleneck
#         self.bottleneck = nn.Sequential(
#             CkBLOCK(512, 1024, kernel_size = 2, padding = 1),
#             CDkBLOCK(1024, 512, kernel_size = 2, padding = 1)
#         )
#         #decoder
#         self.c1024_4_dec = CDkBLOCK(1024, 512, padding = 1) # 2 -> 4
#         self.c1024_3_dec = CDkBLOCK(1024, 512, padding = 1) # 4 -> 8
#         self.c1024_2_dec = CkBLOCK_Transpose(1024, 512, padding = 1) # 8 -> 16
#         self.c1024_1_dec = CkBLOCK_Transpose(1024, 256, padding = 1) # 16 -> 32
#         self.c512_dec = CkBLOCK_Transpose(512, 128, padding = 1) # 32 -> 64
#         self.c256_dec = CkBLOCK_Transpose(256, 64, padding = 1) # 64 -> 128
#         self.c128_dec = CkBLOCK_Transpose(128, in_channels, padding = 1) # 128 -> 256
#         self.tanh = nn.Tanh()
#     def forward(self, x):
#         d1 = self.c64_enc(x)
#         d2 = self.c128_enc(d1)
#         d3 = self.c256_enc(d2)
#         d4 = self.c512_1_enc(d3)
#         d5 = self.c512_2_enc(d4)
#         d6 = self.c512_3_enc(d5)
#         d7 = self.c512_4_enc(d6)
#         bottleneck = self.bottleneck(d7)
#         u2 = self.c1024_4_dec(torch.cat([bottleneck, d7], dim=1))
#         u3 = self.c1024_3_dec(torch.cat([u2, d6], dim=1))
#         u4 = self.c1024_2_dec(torch.cat([u3, d5], dim=1))
#         u5 = self.c1024_1_dec(torch.cat([u4, d4], dim=1))
#         u6 = self.c512_dec(torch.cat([u5, d3], dim=1))
#         u7 = self.c256_dec(torch.cat([u6, d2], dim=1))
#         u8 = self.c128_dec(torch.cat([u7, d1], dim=1))
#         x = self.tanh(u8)
#         return x

class Generator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Generator, self).__init__()
        # encoder        
        self.c64_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding = 1,
                      padding_mode='reflect',
                      bias=False
                     ),
            nn.LeakyReLU(0.2)
        ) # 256 -> 128
        self.c128_enc = CkBLOCK(64, 128, padding = 1) # 128 -> 64
        self.c256_enc = CkBLOCK(128, 256, padding = 1) # 64 -> 32
        self.c512_1_enc = CkBLOCK(256, 512, padding = 1) # 32 -> 16
        self.c512_2_enc = CkBLOCK(512, 512, padding = 1) # 16 -> 8
        self.c512_3_enc = CkBLOCK(512, 512, padding = 1) # 8 -> 4
        self.c512_4_enc = CkBLOCK(512, 512, padding = 1) # 4 -> 2
        
        self.embedding = nn.Linear(8 * 3, 64)
        
        #bottleneck
        self.bottleneck = nn.Sequential(
            CkBLOCK(512, 512, kernel_size = 4, padding = 1)
        )
            
        self.c1024_5_dec = CDkBLOCK(576, 512, padding = 1)
        #decoder
        self.c1024_4_dec = CDkBLOCK(1024, 512, padding = 1) # 2 -> 4
        self.c1024_3_dec = CDkBLOCK(1024, 512, padding = 1) # 4 -> 8
        self.c1024_2_dec = CkBLOCK_Transpose(1024, 512, padding = 1) # 8 -> 16
        self.c1024_1_dec = CkBLOCK_Transpose(1024, 256, padding = 1) # 16 -> 32
        self.c512_dec = CkBLOCK_Transpose(512, 128, padding = 1) # 32 -> 64
        self.c256_dec = CkBLOCK_Transpose(256, 64, padding = 1) # 64 -> 128
        self.c128_dec = CkBLOCK_Transpose(128, in_channels, padding = 1) # 128 -> 256
        self.tanh = nn.Tanh()
    def forward(self, x):
        colors = torch.rand(8 * 3).to(DEVICE).repeat(x.shape[0],1)
        # colors = torch.rand(8 * 3).repeat(x.shape[0],1)
        colors = colors - 0.5
        d1 = self.c64_enc(x)
        d2 = self.c128_enc(d1)
        d3 = self.c256_enc(d2)
        d4 = self.c512_1_enc(d3)
        d5 = self.c512_2_enc(d4)
        d6 = self.c512_3_enc(d5)
        d7 = self.c512_4_enc(d6)
        bottleneck = self.bottleneck(d7)
        embed = self.embedding(colors)
        embed = embed.unflatten(1, torch.Size(([64, 1, 1])))
        latent = torch.cat([bottleneck, embed], dim=1)
        u1 = self.c1024_5_dec(latent)
        u2 = self.c1024_4_dec(torch.cat([u1, d7], dim=1))
        u3 = self.c1024_3_dec(torch.cat([u2, d6], dim=1))
        u4 = self.c1024_2_dec(torch.cat([u3, d5], dim=1))
        u5 = self.c1024_1_dec(torch.cat([u4, d4], dim=1))
        u6 = self.c512_dec(torch.cat([u5, d3], dim=1))
        u7 = self.c256_dec(torch.cat([u6, d2], dim=1))
        u8 = self.c128_dec(torch.cat([u7, d1], dim=1))
        return self.tanh(u8)