import torch
from data import data_load
from model import network
import argparse
import logging
"""
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
"""

from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path

from ema_pytorch import EMA
from utils import Adder, Timer, check_lr, cycle
from model import utility
from val import diffusion_valid

def main(args):
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Initialize WandbLogger
    """
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    """
    # dataset

    if args.mode == 'train':
        train_load = data_load.train_dataloader(args.data_dir, 128, args.batch_size, args.num_worker)
    print('Initial Dataset Finished')

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # model
    nets = network.Unet_with_Guidance(dim = args.dimension,
                                      dim_mults = (1, 2, 4),
                                      flash_attn=True
                                    ).to(device)
    
    diffusion = network.GaussianDiffusion(nets, 
                                        image_size=128, 
                                        timesteps=1000,
                                        sampling_timesteps= 250
                                        ).to(device)
    print('Initial Model Finished')

    optimizer = torch.optim.Adam(diffusion.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    
    max_iter = len(train_load)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    epoch_adder = Adder()
    iter_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    writer = SummaryWriter(os.path.join('runs',args.model_name))
    
    model_save_overwrite = os.path.join(args.model_save_dir, 'model_overwrite.pkl')

    channels = diffusion.channels
    is_ddim_sampling = diffusion.is_ddim_sampling
    
    if not utility.exists(args.convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels)
    
    num_samples = 16
    assert utility.has_int_squareroot(num_samples), 'number of samples must have an integer square root'
    train_batch_size = 16
    gradient_accumulate_every = 1
    assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

    train_num_steps = 100000
    image_size = diffusion.image_size
    max_grad_norm = 1

    dataloader = cycle(train_load)

    if os.path.isfile(model_save_overwrite):
        state_dict = torch.load(model_save_overwrite)
        diffusion.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        start_epoch = state_dict['epoch']
        lr = check_lr(optimizer)
        print("\n Model Restored, epoch = %4d\n" % (start_epoch + 1))
        print("\n                 current lr = %10f\n" % (lr))
    else:
        print("No previous data... Started from scratch ... \n")
        start_epoch = 0
  

    best_psnr=-1
    for epoch_idx in range(start_epoch + 1, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        total_loss = 0
        
        for iter_idx, batch_data in enumerate(tqdm(train_load)):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            
            loss = diffusion(input_img)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
            iter_adder(loss.item())
            epoch_adder(loss.item())
            
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (iter_timer.toc(), epoch_idx,
                                                                             iter_idx + 1, max_iter, lr,
                                                                             iter_adder.average()))
                writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_adder.reset()

        if epoch_idx % args.save_freq == 0:
            torch.save({'model': diffusion.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, model_save_overwrite)
            
            
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Loss: %7.4f" % (
        epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        epoch_adder.reset()
        scheduler.step()
        
        
        if epoch_idx % args.valid_freq == 0:
            
              
            val = diffusion_valid(diffusion, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            

            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': diffusion.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
    
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': diffusion.state_dict()}, save_name)

    print('End of training.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    
    parser.add_argument('--model_name', type=str, default='DiffusionDeblurGuidance')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    
    # Train
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
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

    parser.add_argument('--convert_image_to', type=bool, default=None)
    
    
    # Test
    parser.add_argument('--test_model', type=str, default='pretrained_model.pkl')
    parser.add_argument('--mode', type=str, default='train')
    
    args = parser.parse_args()
    
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'eval/')

    # parse args
    
    
    main(args)
    
    
    