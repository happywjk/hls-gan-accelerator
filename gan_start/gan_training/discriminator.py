# %% 
import torch
import torch.nn as nn
 # %% 
class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DisBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        ) 
    
    def forward(self, x):
        return self.layers(x)
 # %% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential( #input 64x64
            DisBlock(3, 32),      # output 32x32
            DisBlock(32, 64),   # output 8x8
            DisBlock(64, 128),   # output 4x14
             DisBlock(128, 256), 
            nn.Conv2d(256, 1, 4, 2, 0)  # output 1x1
        )

    def forward(self, x):
        return self.layers(x)

