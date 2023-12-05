from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
import numpy as np
import shutil
from utils.utils import *
import kornia
from data.dataset import *
from utils.validate_utils import *
import argparse


def val_model(model, writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch):

    # for the use of visualizing the validation properly on the tensorboard
    iters -= 500
    phase = "val"
    loss_list = []
    rot_list = []
    model.eval()   # Set model to evaluate mode
    acc_x = np.zeros(20)
    acc_y = np.zeros(20)
    acc = 0.

    with torch.no_grad():

        for defected, clear, defect_gt, rotation_gt, translation_gt in dataloader(batch_size_val)[phase]:
            defected = defected.to(device)
            clear = clear.to(device)
            defect_gt = defect_gt.to(device)
            iters += 1    
            print("rotation_gt", rotation_gt)
            print("translation_gt", translation_gt)            
            loss_defect, defected_unet, clear_unet, clear_transformed, defection_map, defect_gt_blur= validate_defect(defected, clear, defect_gt, model, phase, device )

            loss_list.append(loss_defect.tolist())
            writer_val.add_scalar('LOSS DEFECT', loss_defect.detach().cpu().numpy(), iters)
            writer_val.add_image("defection_map", defection_map[0,:,:].cpu(), iters)
            writer_val.add_image("gt", defect_gt[0,:,:].cpu(), iters)
            writer_val.add_image("defected_unet", defected_unet[0,:,:].cpu(), iters)
            writer_val.add_image("clear_unet", clear_unet[0,:,:].cpu(), iters)
           
    return loss_list

if __name__ == "__main__":
    # Passing a bunch of parameters
    parser_val = argparse.ArgumentParser(description="DPCN Network Validation")
    parser_val.add_argument('--only_valid', action='store_true', default=False)
    parser_val.add_argument('--cpu', action='store_true', default=False)
    parser_val.add_argument('--load_path', type=str, default="./checkpoints/checkpoint.pt")
    parser_val.add_argument('--use_dsnt', action='store_true', default=False)
    parser_val.add_argument('--batch_size_val', type=int, default=2)
    parser_val.add_argument('--val_writer_path', type=str, default="./checkpoints/log/val/")
    args_val = parser_val.parse_args()

    if args_val.only_valid:
        epoch = 1
        checkpoint_path = args_val.load_path
        device = torch.device("cuda:0" if not args_val.cpu else "cpu")
        print("The devices that the code is running on:", device)
        writer_val = SummaryWriter(log_dir=args_val.val_writer_path)
        batch_size_val = args_val.batch_size_val
        dataloader = generate_dataloader
        dsnt = args_val.use_dsnt


        num_class = 1
        start_epoch = 0
        iters = 0


    # create a shell model for checkpoint loader to load into
        model = UNet(num_class).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-3)



    # load checkpoint
        model, optimizer, start_epoch = load_checkpoint(\
                                                    checkpoint_path, model, optimizer, device)        

    # Entering the mean loop of Validation
        loss_list = val_model(model, writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch)