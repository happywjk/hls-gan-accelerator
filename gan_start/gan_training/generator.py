 # %% 
import torch
import torch.nn as nn
 # %% 
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels,padding):
        super(GenBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(  #input 1x1
            GenBlock(128,256,0),     #4x4
            GenBlock(256, 128,1),  # output 8x8 
            GenBlock(128, 64,1), # output 16x16
            GenBlock(64, 32,1), # output 32x32
            nn.ConvTranspose2d(32, 3,4,2,1),  # output 64x64
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)