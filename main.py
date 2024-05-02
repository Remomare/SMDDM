import os
import torch
import argparse
from torch.backends import cudnn
from model.network import XYDeblur
from train import XYDeblur_train, Diffusion_Trainer
from eval import XYDeblur_eval
from model import network, layers

def main(config):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(config.model_save_dir)
    if not os.path.exists('results/' + config.model_name + '/'):
        os.makedirs('results/' + config.model_name + '/')
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    if config.model_name == 'XYDeblur':
        model = XYDeblur()
        if torch.cuda.is_available():
            model.cuda()
        if config.mode == 'train':
            XYDeblur_train(model, config)

        elif config.mode == 'test':
            XYDeblur_eval(model, config)

    if config.model_name == 'DiffusionDeblurGuidance':
        nets = network.Unet_with_Guidance(dim = config.dimension,
                                        dim_mults = (1, 2, 4),
                                        flash_attn=True
                                        )
        model = network.GaussianDiffusion(nets, 
                                        image_size=128, 
                                        timesteps=1000,
                                        sampling_timesteps= 250
                                        )
        trainer = Diffusion_Trainer(config,
                                    model,
                                    train_batch_size = 32,
                                    train_lr = 8e-5,
                                    train_num_steps = 700000,         # total training steps
                                    gradient_accumulate_every = 2,    # gradient accumulation steps
                                    ema_decay = 0.995,                # exponential moving average decay
                                    amp = True,                       # turn on mixed precision
                                    calculate_fid = True   
                                    )
        trainer.train()
        
    if config.model_name == 'XYDiffusionDeblurGuidance':
        nets = network.XYUnet_with_Guidance(dim = config.dimension,
                                        dim_mults = (2, 3, 4),
                                        flash_attn=True
                                        )
        model = layers.GaussianDiffusion(nets, 
                                        image_size=128, 
                                        timesteps=1000,
                                        sampling_timesteps= 250
                                        )
        trainer = Diffusion_Trainer(
            config,
            model,
            results_folder=config.result_dir,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = True   
        )
        trainer.train()
        
    if config.model_name == 'XYDiffusionDeblur':
        nets = network.XYUnet_without_Guidance(dim = config.dimension,
                                        dim_mults = (2, 3, 4),
                                        flash_attn=True
                                        )
        model = layers.GaussianDiffusion(nets, 
                                        image_size=128, 
                                        timesteps=1000,
                                        sampling_timesteps= 250
                                        )
        trainer = Diffusion_Trainer(
            config,
            model,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = True   
        )
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', type=str, default='XYDeblur')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    
    # Train
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(10000//500)])

    parser.add_argument('--store_opt', type=bool, default=True)
    parser.add_argument('--num_subband', type=int, default=2)
    parser.add_argument('--store_freq', type=int, default=300)

    # Test
    parser.add_argument('--test_model', type=str, default='pretrained_model.pkl')
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()
    config.model_save_dir = os.path.join('results/', config.model_name, 'weights/')
    config.result_dir = os.path.join('results/', config.model_name, 'eval/')
    print(config)
    main(config)