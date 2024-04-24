import torch
import math
import torch.nn as nn

from packaging import version
from functools import partial
from collections import namedtuple

from einops import rearrange, reduce, repeat
from einops.layers import torch as einops_torch

from . import utility 

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, utility.default(dim_out, dim), 3, padding = 1 )
    )
    
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        einops_torch.Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim*4, utility.default(dim_out, dim), 1)    
    )

class RMSNorm(nn.Module):
    def __init__(self, dim) -> None:
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        
    def forward(self, x):
        return nn.functional.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

#pos emb

class Sinusoidal_Positional_Embed(nn.Module):
    def __init__(self, dim, theta = 10000):
        super(Sinusoidal_Positional_Embed, self).__init__()
        assert utility.divisible_by(dim, 2)
        self.dim = dim
        self.theta = theta
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
        
class Random_or_Learned_Sinusoidal_Positional_Embed(nn.Module):
    def __init__(self, dim, is_random = False):
        super(Random_or_Learned_Sinusoidal_Positional_Embed, self).__init__()
        assert utility.divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)
        
    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weight, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered       

# attention

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if utility.exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = utility.default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
  
        
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super(LinearAttention, self).__init__()       
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.norm = RMSNorm(dim)
        
        self.mem_kv = nn.Parameter(torch.rand(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t,  'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))
        
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
        
        
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

        
class BasicConv(nn.Module):
    def __init__(self, 
                 in_channel :int, 
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
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
            
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)



class NoiseBlock(nn.Module):
    def __init__(self, image_size, dim, dim_out, time_emb_dim = None, activation_function="swish"):
        super(NoiseBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)
        
        #gaussian noise 
        
        self.main = nn.Sequential(
            #Position embedding
            self.activation_fn, #swish
            nn.Linear(time_emb_dim, dim_out * 2)#MLP
        )

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="swish",norm=False):
        super(ResBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)
        
        self.main = nn.Sequential(
            self.activation_fn, #swish
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm), #conv 3x3
            self.activation_fn, #swish
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm) #conv 3x3
        )
        
    def forward(self, x):
        return self.main(x) + x   


class Diffusion_ResBlock(nn.Module):
    def __init__(self, dim, dim_out, base_dim = 32, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if utility.exists(time_emb_dim) else None

        guidance_base_channel = 32
        self.base_channel = base_dim

        self.block = BasicConv(dim, dim_out, kernel_size=3, stride=1)
        self.activation = nn.SiLU(inplace=True)
        
        self.guidance_conv = BasicConv(guidance_base_channel*4, self.base_channel, kernel_size=1, stride=1)

    def forward(self, x, guidance = None , time_emb = None):

        scale_shift = None
        if utility.exists(self.mlp) and utility.exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        if guidance is not None:
            guide = self.guidance_conv(guidance)
            x = x + guide
        
        x = self.activation(x)
        
        h = self.block(x)
        if utility.exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        return h


class Guided_Diffusion_ResBlock(nn.Module):
    def __init__(self, 
                 in_channel, out_channel, attn_channel, activation_function="swish", norm=False):
        super(Guided_Diffusion_ResBlock, self).__init__()
        self.activation_fn = utility.get_activation_function(activation_function)

        base_channel = 32
        
        self.guidance_conv = BasicConv(base_channel*4, base_channel, kernel_size=1, stride=1)
        self.conv_1 = BasicConv(in_channel + base_channel, out_channel, kernel_size=3, stride=1, norm=norm),
        self.conv_2 = BasicConv(out_channel + base_channel + attn_channel, out_channel, kernel_size=3, stride=1, norm=norm)
          
    def forward(self, input, guidance):
        guidances = self.guidance_conv(guidance)
        x = input + guidances
        x = self.activation_fn(x)
        x = self.conv_1(x)
        x + x
        return 
    

class XYDeblur_Resblock(nn.Module):
    def __init__(self, in_channel, out_channel, norm=False) -> None:
        super(XYDeblur_Resblock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm),
            nn.ReLU(inplace=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm)
        )

    def forward(self, x):
        return self.main(x) + x
   
    
class XYDeblur_EBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8, norm=False, first=False):
        super(XYDeblur_EBlock, self).__init__()
        if first:
            layers = [BasicConv(in_channel, out_channel, kernel_size=3,norm=norm, stride=1),
                      nn.ReLU(inplace=True)]
        else:
            layers = [BasicConv(in_channel,out_channel,kernel_size=3,stride=2),
                      nn.ReLU(inplace=True)]
        
        layers += [XYDeblur_Resblock(out_channel, out_channel, norm) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
class XYDeblur_DBlock(nn.Module):
    def __init__(self, channel, num_res=8, norm=False, last=False, feature_ensemble=False):
        super(XYDeblur_DBlock, self).__init__()

        layers = [XYDeblur_Resblock(channel, channel, norm) for _ in range(num_res)]

        if last:
            if feature_ensemble == False:
                layers.append(BasicConv(channel, 3, kernel_size=3, norm=norm, relu=False, stride=1))
        else:
            layers.append(BasicConv(channel, channel // 2, kernel_size=4, norm=norm, stride=2, transpose=True))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)