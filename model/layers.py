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
                 activation_fun :str, 
                 bias=True, 
                 norm=False, 
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        
        padding = kernel_size // 2
        layers = list()
        if activation_fun != None:
            act_fn = utility.get_activation_function(activation_fun)
            layers.append(act_fn)
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
        
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="relu",norm=False) -> None:
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, activation_fun=activation_function, norm=norm),
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, activation_fun=activation_function, norm=norm)
        )
        
    def forward(self, x):
        return self.main(x)

class Guided_ResBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Guided_ResBlock, self).__init__()
        