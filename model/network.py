import torch
import torch.nn as nn
import torchvision.transforms as transforms

import math

from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from random import random
from functools import partial

from einops import reduce

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
        
        block_klass = partial(layers.ResBlock ,groups = resnet_block_groups)
        
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
        
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([utility.divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        if self.self_condition:
            x_self_cond = utility.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = utility.linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = utility.cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = utility.sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = utility.default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = utility.normalize_to_neg_one_to_one if auto_normalize else utility.identity
        self.unnormalize = utility.unnormalize_to_zero_to_one if auto_normalize else utility.identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            utility.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            utility.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (utility.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            utility.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            utility.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            utility.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            utility.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            utility.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            utility.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            utility.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = utility.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = utility.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else utility.identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return utility.ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = utility.default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = utility.default(noise, lambda: torch.randn_like(x_start))

        return (
            utility.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            utility.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = utility.default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = utility.default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * utility.rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x_start, x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = nn.functional.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * utility.extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

class Guidance(nn.Module):
    def __init__(self) -> None:
        super(Guidance, self).__init__()
        
        self.dim = 1
        
        in_channel = 1
        base_channel = 32
        
        self.to_grayscale = transforms.Grayscale()
        self.conv_start = layers.BasicConv(in_channel, base_channel, kernel_size=3, stride=1) #conv 3x3
        self.ResBlock_1 = layers.ResBlock(base_channel, base_channel)
        self.ResBlock_2 = layers.ResBlock(base_channel, base_channel*2)
        self.ResBlock_3 = layers.ResBlock(base_channel*2, base_channel*3)
        self.ResBlock_4 = layers.ResBlock(base_channel*3, base_channel*4)
        self.conv_end = layers.BasicConv(base_channel*4, in_channel, kernel_size=3, stride=1) #conv 3x3
    

        self.downsampling = layers.Downsample(self.dim)
        self.upsampling = layers.Upsample(base_channel*4)
            
    def forward(self, input, downsample_factor):
        x = self.to_grayscale(input)
        for i in range(int(math.log2(downsample_factor))):
            x = self.downsampling(x)
        x = self.conv_start(x)
        x = self.ResBlock_1(x)
        x = self.ResBlock_2(x)
        x = self.ResBlock_3(x)
        guidance = self.ResBlock_4(x)
        for i in range(int(math.log2(downsample_factor))):
            y = self.upsampling(guidance)
        y = self.conv_end(y)
        return guidance, y
    
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

class Unet_with_Guidance(nn.Module):
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
        
        self.guidance_net = Guidance()
        
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else  1)
        
        init_dim = utility.default(init_dim, dim)
        self.init_conv = layers.BasicConv(input_channels, init_dim, 7)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
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
                layers.Diffusion_ResBlock(dim_out, dim_out, dim_in, time_emb_dim = time_dim),
                layers.Diffusion_ResBlock(dim_out, dim_out, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
                 # 여기부터 수정 
        mid_dim = dims[-1]
        self.mid_block1 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_block3 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block4 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else layers.LinearAttention
            
            self.ups.append(nn.ModuleList([
                layers.Diffusion_ResBlock(dim_out + dim_in, dim_in, time_emb_dim = time_dim),
                layers.Diffusion_ResBlock(dim_out + dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = utility.default(out_dim, default_out_dim)

        self.final_res_block = layers.Diffusion_ResBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, input, x, time, x_self_cond = None):
        assert all([utility.divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        guidance, y = self.guidance_net(input, self.downsample_factor)
        
        if self.self_condition:
            x_self_cond = utility.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = downsample(x)
            
            x = block1(x, guidance, t)
            h.append(x)

            x = attn(x) + x
            x = block2(x, guidance,t)
            
            h.append(x)

        x = self.mid_block1(x, guidance, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, guidance, t)
        x = self.mid_block3(x, guidance, t)
        x = self.mid_attn(x) + x
        x = self.mid_block4(x, guidance, t)

        for block1, block2, attn, upsample in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            
            x = attn(x) + x
            
            x = block2(x, t)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class XYUnet_with_Guidance(nn.Module):
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
        
        self.guidance_net = Guidance()
        
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else  1)
        
        init_dim = utility.default(init_dim, dim)
        self.init_conv = layers.BasicConv(input_channels, init_dim, 7)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
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
                layers.Diffusion_ResBlock(dim_out, dim_out, dim_in, time_emb_dim = time_dim),
                layers.Diffusion_ResBlock(dim_out, dim_out, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
                 # 여기부터 수정 
        mid_dim = dims[-1]
        self.mid_block1 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_block3 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block4 = layers.Diffusion_ResBlock(mid_dim, mid_dim, dim_in, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else layers.LinearAttention
            
            self.ups.append(nn.ModuleList([
                layers.Diffusion_ResBlock(dim_out + dim_in, dim_in, time_emb_dim = time_dim),
                layers.Diffusion_ResBlock(dim_out + dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                layers.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = utility.default(out_dim, default_out_dim)

        self.final_res_block = layers.Diffusion_ResBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, input, x, time, x_self_cond = None):
        assert all([utility.divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        guidance, y = self.guidance_net(input, self.downsample_factor)
        
        if self.self_condition:
            x_self_cond = utility.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = downsample(x)
            
            x = block1(x, guidance, t)
            h.append(x)

            x = attn(x) + x
            x = block2(x, guidance,t)
            
            h.append(x)

        x = self.mid_block1(x, guidance, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, guidance, t)
        
        x_trans = x.transpose(2,3).flip(2)
        
        x = self.mid_block3(x, guidance, t)
        x = self.mid_attn(x) + x
        x = self.mid_block4(x, guidance, t)


        x_trans = self.mid_block3(x_trans, guidance, t)
        x_trans = self.mid_attn(x_trans) + x_trans
        x_trans = self.mid_block4(x_trans, guidance, t)

        
        for block1, block2, attn, upsample in self.ups:
            x = upsample(x)
            x_trans = upsample(x_trans)
            residual = h.pop()
            x = torch.cat((x, residual), dim = 1)
            x_trans = torch.cat((x_trans, residual.transpose(2,3).flip(2)), dim = 1)
            x = block1(x, t)
            
            x = attn(x) + x
            
            x = block2(x, t)
            
            x_trans = block1(x_trans, t)
            
            x_trans = attn(x_trans) + x
            
            x_trans = block2(x_trans, t)

        x = torch.cat((x, r), dim = 1)
        
        x_trans = torch.cat((x_trans, r.transpose(2,3).flip(2)), dim=1)

        x = self.final_res_block(x, t)
        x_trans = self.final_res_block(x_trans, t)
        
        x = self.final_conv(x)
        x_trans = self.final_conv(x_trans)
        
        return x + x_trans
