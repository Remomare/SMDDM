import torch
import torch.nn as nn
import torchvision.transforms as transforms

import layers

class Unet_encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
class Unet_decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Guidance(nn.Module):
    def __init__(self) -> None:
        super(Guidance, self).__init__()
        
        in_channel = 1
        base_channel = 32
        
        self.to_grayscale = transforms.Grayscale()
        self.downsamlping
        self.conv_start = layers.BasicConv(in_channel, base_channel, kernel_size=3, stride=1, activation_fun=None)
        self.ResBlock_1 = layers.ResBlock()
        self.ResBlock_2 = layers.ResBlock()
        self.ResBlock_3 = layers.ResBlock()
        self.ResBlock_4 = layers.ResBlock()
        self.conv_end = layers.BasicConv(in_channel, base_channel, kernel_size=3, stride=1, activation_fun=None)
    
    def forward(self, input):
        x = self.to_grayscale(input)
        x = self.conv_start(x)
        
        return x