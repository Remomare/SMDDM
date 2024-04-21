import torch
import torch.nn as nn
import torchvision.transforms as transforms

from functools import partial

from model import layers
from model import utility

class Unet_DDPM(nn.Module):
    def __init__(self,
                 dim,
                 init_dim = None,
                 out_dim = None,
                 dim_mults = (1,2,4,8),
                 channels = 3,
                 self_condition = False,
                 resnet_block_groups = 8,
                 learned_variance = False,
                 learned_sinusoidal_cond = False,
                 random_fourier_features = False,
                 learned_sinusoidal_dim = 16,
                 sinusoidal_pos_emb_theta = 10000,
                 attn_dim_head = 32,
                 attn_heads = 4,
                 full_attn = None,    # defaults to full attention only for inner most layer
                 flash_attn = False
                 ) -> None:
        super(Unet_DDPM, self).__init__()
        
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else  1)
        
        int_dim = utility.default(init_dim, dim)
        self.init_conv = layers.BasicConv(input_channels, init_dim, 7)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = utility.partial(layers.ResBlock ,groups = resnet_block_groups)
        
        time_dim = dim * 4
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = layers.Random_or_Learned_Sinusoidal_Positional_Embed(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = layers.Sinusoidal_Positional_Embed(dim, theta= sinusoidal_pos_emb_theta)
            fourier_dim = dim
        
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)
            
        num_stages = len(dim_mults)
        full_attn = utility.cast_tuple(full_attn, num_stages)
        attn_heads = utility.cast_tuple(attn_heads, num_stages)
        attn_dim_head = utility.cast_tuple(attn_dim_head, num_stages)
        
        assert len(full_attn) == len(dim_mults)
        
        FullAttention = partial(layers.Attention, flash = flash_attn)
        
        self.down = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else layers.LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else layers.LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = utility.default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


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
        self.Decoder1_4 = layers.BasicConv(base_channel, 3, kernel_size=3, stride=1)


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
