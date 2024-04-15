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
        self.conv_start = layers.BasicConv(in_channel, base_channel, kernel_size=3, stride=1) #conv 3x3
        self.ResBlock_1 = layers.ResBlock()
        self.ResBlock_2 = layers.ResBlock()
        self.ResBlock_3 = layers.ResBlock()
        self.ResBlock_4 = layers.ResBlock()
        self.conv_end = layers.BasicConv(in_channel, base_channel, kernel_size=3, stride=1) #conv 3x3
    
    def downsampling(self, dim, dim_out):
        return layers.Downsample(dim, dim_out)
    
    def forward(self, input):
        x = self.to_grayscale(input)
        x = self.downsampling()
        x = self.conv_start(x)
        
        return x
    
    
class XYDeblur(nn.Module): #baseline1
    def __init__(self):
        super(XYDeblur, self).__init__()

        in_channel = 3
        base_channel = 32
        
        self.relu = nn.ReLU(inplace=True)
        
        num_res_ENC = 6

        self.Encoder1 = layers.XYDeblur_EBlock(in_channel, base_channel, num_res_ENC, first=True)
        self.Encoder2 = layers.XYDeblur_EBlock(base_channel, base_channel*2, num_res_ENC, norm=False)
        self.Encoder3 = layers.XYDeblur_EBlock(base_channel*2, base_channel*4, num_res_ENC, norm=False)

        self.Convs1_1 = layers.BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, stride=1)
        self.Convs1_2 = layers.BasicConv(base_channel * 2, base_channel, kernel_size=1, stride=1)

        num_res_DEC = 6

        self.Decoder1_1 = layers.XYDeblur_DBlock(base_channel * 4, num_res_DEC, norm=False)
        self.Decoder1_2 = layers.XYDeblur_DBlock(base_channel * 2, num_res_DEC, norm=False)
        self.Decoder1_3 = layers.XYDeblur_DBlock(base_channel, num_res_DEC, last=True, feature_ensemble=True)
        self.Decoder1_4 = layers.BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)


    def forward(self, x):
        output = list()
        
        # Common encoder
        x_e1 = self.Encoder1(x)
        x_e2 = self.Encoder2(x_e1)
        x_decomp = self.Encoder3(x_e2)

        # Resultant image reconstruction
        x_decomp1 = self.Decoder1_1(x_decomp)
        x_decomp1 = self.Convs1_1(torch.cat([x_decomp1, x_e2], dim=1))
        x_decomp1 = self.relu(x_decomp1)
        x_decomp1 = self.Decoder1_2(x_decomp1)
        x_decomp1 = self.Convs1_2(torch.cat([x_decomp1, x_e1], dim=1))
        x_decomp1 = self.relu(x_decomp1)
        x_decomp1 = self.Decoder1_3(x_decomp1)
        x_decomp1 = self.Decoder1_4(x_decomp1)

        x_decomp_rot = x_decomp.transpose(2, 3).flip(2)
        x_e1_rot = x_e1.transpose(2, 3).flip(2)
        x_e2_rot = x_e2.transpose(2, 3).flip(2)

        x_decomp2 = self.Decoder1_1(x_decomp_rot)
        x_decomp2 = self.Convs1_1(torch.cat([x_decomp2, x_e2_rot], dim=1))
        x_decomp2 = self.relu(x_decomp2)
        x_decomp2 = self.Decoder1_2(x_decomp2)
        x_decomp2 = self.Convs1_2(torch.cat([x_decomp2, x_e1_rot], dim=1))
        x_decomp2 = self.relu(x_decomp2)
        x_decomp2 = self.Decoder1_3(x_decomp2)
        x_decomp2 = self.Decoder1_4(x_decomp2)

        x_decomp2 = x_decomp2.transpose(2, 3).flip(3)
        
        x_final = x_decomp1 + x_decomp2 + x

        output.append(x_decomp1)
        output.append(x_decomp2)
        output.append(x_final)
        
        return output

def build_net():
    return XYDeblur()