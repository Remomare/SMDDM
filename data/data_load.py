import os
import torch
import numpy as np
from PIL import Image as Image
from data import data_augment 
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def train_dataloader(path, model_image_size, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')
    # TODO: Add transform
    transform = None
    if use_transform:
        data_aug = data_augment.PairCompose(
            [
                data_augment.PairRandomCrop(model_image_size),
                data_augment.PairRandomHorizontalFlip(),
                data_augment.PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, data_aug=data_aug),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, model_image_size, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    transform = transforms.Compose(
            [
                transforms.Resize(size = (model_image_size, model_image_size)),
                transforms.ToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'valid')
    dataloader = DataLoader(
        DeblurDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


class DeblurDataset(Dataset):

    def __init__(self, image_dir, transform=None, data_aug = None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.data_aug = data_aug

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        elif self.data_aug:
            image, label = self.data_aug(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            _, ext = x.split('.')
            if ext not in ['png', 'jpg', 'jpeg']:
                raise ValueError