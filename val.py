import torch
from torchvision.transforms import functional as F
from data.data_load import valid_dataloader, test_dataloader
from utils import Adder, calculate_psnr
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import sys, scipy.io
from model import utility

def XYDeblur_valid(model, config, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(config.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(config.result_dir, '%d' % (ep)))
            pred = model.sample(input_img)

            p_numpy = pred.squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)

            if config.store_opt:
                if ep % config.store_freq == 0:
                    if idx % 20 == 0:
                        save_name = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '.png')
                        save_name_R = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_Result.png')
                        save_name_I = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_Input.png')
                        save_name_R_resize = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_Result_resize.png')


                        label = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
                        label.save(save_name)

                        input_i = F.to_pil_image(input_img.squeeze(0).cpu(), 'RGB')
                        input_i.save(save_name_I)

                        pred = torch.clamp(pred, 0, 1)
                        result = F.to_pil_image(pred.squeeze(0).cpu(), 'RGB')
                        result.save(save_name_R)
                        
                        result_resize = F.to_pil_image(F.resize(pred.squeeze(0).cpu(), (1280,720)), 'RGB')
                        result.save(save_name_R_resize)
                        
                        for num_sub in range(config.num_subband):
                            tmp_save_name = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_' + str(num_sub) + '.mat')
                            tmp_result = pred[num_sub].squeeze(0).cpu().numpy()
                            scipy.io.savemat(tmp_save_name, mdict={'data': tmp_result})

            psnr_adder(psnr)
            if idx % 100 == 0:
                print('\r%03d'%idx, end=' ')
    print('\n')
    model.train()
    return psnr_adder.average()

def diffusion_valid(diffusion, args, epoch_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, 128, batch_size=1, num_workers=0)
    diffusion.eval()
    psnr_adder = Adder()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (epoch_index))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (epoch_index)))
            pred = diffusion.sample(input_img)
            
            p_numpy = pred.squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()
            
            #psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            
            if args.store_opt:
                if epoch_index % args.store_freq == 0:
                    if idx % 20 == 0:
                        save_name = os.path.join(args.result_dir, '%d' %epoch_index, '%d' % (idx) + '.png')
                        save_name_R = os.path.join(args.result_dir, '%d' %epoch_index, '%d' % (idx) + '_Result.png')
                        save_name_I = os.path.join(args.result_dir, '%d' %epoch_index, '%d' % (idx) + '_Input.png')
                        save_name_R_resize = os.path.join(args.result_dir, '%d' %epoch_index, '%d' % (idx) + '_Result_resize.png')
                        
                        #batches = utility.num_to_groups(args.num_samples, args.batch_size)
                        #all_images_list = list(map(lambda n: diffusion.sample(label = input_img, batch_size=n), batches))
            
                        label = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
                        label.save(save_name)
                        
                        input_i = F.to_pil_image(input_img.squeeze(0).cpu(), 'RGB')
                        input_i.save(save_name_I)

                        pred= torch.clamp(pred, 0, 1)
                        result = F.to_pil_image(pred.squeeze(0).cpu(), 'RGB')
                        result.save(save_name_R)
            
                        result_resize = F.to_pil_image(F.resize(pred.squeeze(0).cpu(), (1280,720)), 'RGB')
                        result_resize.save(save_name_R_resize)
                        
            #psnr_adder(psnr)
            if idx % 100 == 0:
                print('\r%03d'%idx, end=' ')
    print('\n')
    diffusion.train()
    return psnr_adder.average()