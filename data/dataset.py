from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import torch
import sys
import data.simulation as simulation
from data.data_utils import *


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.defect_image, self.clear_image, self.defect_gt, self.gt_rot, self.gt_trans = simulation.generate_random_data(256, 256, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.defect_image)

    def __getitem__(self, idx):
        defect_image = self.defect_image[idx]
        clear_image = self.clear_image[idx]
        defect_gt = self.defect_gt[idx]
        gt_rot = self.gt_rot[idx]
        gt_trans = self.gt_trans[idx]
        gt_scale = torch.tensor(1.)
        
        if self.transform:
            defect_image = self.transform(defect_image)
            clear_image = self.transform(clear_image)
            defect_gt = self.transform(defect_gt)
            gt_rots = torch.tensor(gt_rot)
            gt_scale = torch.tensor(1.)
            gt_trans = torch.tensor(gt_trans)
        # print("gt = ", gt)
        # print("gt tensor = ", gt_tensor)

        return [defect_image, clear_image, defect_gt, gt_rots, gt_trans]

def generate_dataloader(batch_size):
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(1000, transform = trans)
    val_set = SimDataset(100, transform = trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    return dataloaders