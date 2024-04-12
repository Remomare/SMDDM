import torch
import math
import torch.nn as nn

import utility 

class BasicConv(nn.Module):
    def __init__(self, 
                 in_channle :int, 
                 out_channel :int, 
                 kernel_size :int, 
                 stride :int, 
                 bias=True, 
                 norm=False, 
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        
        padding = kernel_size // 2
        layers = list()

        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channle, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channle, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
            
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)

class NoiseBlock(nn.Module):
    def __init__(self, activation_function="swish") -> None:
        super(NoiseBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)

        
        self.main = nn.Sequential(
            #Position embeding
            self.activation_fn
            #MLP
        )

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="swish",norm=False) -> None:
        super(ResBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)
        
        self.main = nn.Sequential(
            self.activation_fn,
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm),
            self.activation_fn,
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm)
        )
        
    def forward(self, x):
        return self.main(x) + x   

class Diffusion_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="swish",norm=False) -> None:
        super(Diffusion_ResBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)
        
        self.main = nn.Sequential(
            self.activation_fn,
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm),
            #Noise block
            self.activation_fn,
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm)
        )
        
    def forward(self, x):
        return self.main(x) + x

class Guided_Diffusion_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="swish", norm=False) -> None:
        super(Guided_Diffusion_ResBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)

        self.main = nn.Sequential(
            #concat guidance
            self.activation_fn,
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm),
            #concat duidance + Noise block
            self.activation_fn,
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm)
          
        )
    def forward(self, x):
        return self.main(x) + x