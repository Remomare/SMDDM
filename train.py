import os
import torch
import math

from pathlib import Path
from data.data_load import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from val import XYDeblur_valid
import torch.nn.functional as F
from tqdm import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator

import utils
from matrix import fid_score
from model import network, utility

def XYDeblur_train(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    dataloader = train_dataloader(config.data_dir, config.batch_size, config.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_steps, config.gamma)

    writer = SummaryWriter(os.path.join('runs',config.model_name))
    epoch_adder = Adder()
    iter_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    model_save_overwrite = os.path.join(config.model_save_dir, 'model_overwrite.pkl')

    if os.path.isfile(model_save_overwrite):
        state_dict = torch.load(model_save_overwrite)
        model.load_state_dict(state_dict['model'])
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
    for epoch_idx in range(start_epoch + 1, config.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(tqdm(dataloader)):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)

            l_tot = criterion(pred_img[config.num_subband], label_img)

            loss = l_tot
            
            loss.backward()
            optimizer.step()

            iter_adder(loss.item())
            epoch_adder(loss.item())

            if (iter_idx + 1) % config.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (iter_timer.toc(), epoch_idx,
                                                                             iter_idx + 1, max_iter, lr,
                                                                             iter_adder.average()))
                writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_adder.reset()

        if epoch_idx % config.save_freq == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, model_save_overwrite)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Loss: %7.4f" % (
        epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        epoch_adder.reset()
        scheduler.step()
        if epoch_idx % config.valid_freq == 0:
            val = XYDeblur_valid(model, config, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)

            if val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(config.model_save_dir, 'Best.pkl'))

            save_name = os.path.join(config.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
    
    save_name = os.path.join(config.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
    

class Trainer(object):
    def __init__(
        self,
        config,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super(Trainer, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('torch device is %s' %device)
        
        self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = mixed_precision_type if amp else 'no'
            )

            # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not utility.exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert utility.has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        dataloader = train_dataloader(config.data_dir, config.batch_size, config.num_worker)

        dataloader = self.accelerator.prepare(dataloader)
        self.dataloader = utils.cycle(dataloader)

        # optimizer

        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = fid_score.FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if utility.exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if utility.exists(self.accelerator.scaler) and utility.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and utility.divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = utility.num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')