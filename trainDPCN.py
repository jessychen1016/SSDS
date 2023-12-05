
import torch
import kornia
import time
import copy
import shutil
import numpy as np
import torch.nn as nn
from graphviz import Digraph
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from data.dataset import *
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.train_utils import *
from validate import val_model
import argparse
import os

#create a set of folders for intermediate results
if not os.path.exists("./checkpoints/log"):
    os.mkdir("./checkpoints/log")
if not os.path.exists("./checkpoints/log/train"):
    os.mkdir("./checkpoints/log/train")
if not os.path.exists("./checkpoints/log/val"):
    os.mkdir("./checkpoints/log/val")

# adding a bunch of parameters for an easy access
parser = argparse.ArgumentParser(description="DPCN Network Training")

parser.add_argument('--cpu', action='store_true', default=False, help="The Program will use cpu for the training")
parser.add_argument('--save_path', type=str, default="./checkpoints/", help="The path to save the checkpoint of every epoch")
parser.add_argument('--load_pretrained', action='store_true', default=False, help="Choose whether to use a pretrained model to fine tune")
parser.add_argument('--load_path', type=str, default="./checkpoints/checkpoint.pt", help="The path to load a pretrained checkpoint")
parser.add_argument('--load_optimizer', action='store_true', default=False, help="When using a pretrained model, options of loading it's optimizer")
parser.add_argument('--pretrained_mode', type=str, default="all", help="...")
parser.add_argument('--use_dsnt', action='store_true', default=False, help="When enabled, the loss will be calculated via DSNT and MSELoss, or it will use a CELoss")
parser.add_argument('--batch_size_train', type=int, default=2, help="The batch size of training")
parser.add_argument('--batch_size_val', type=int, default=2, help="The batch size of validation")
parser.add_argument('--train_writer_path', type=str, default="./checkpoints/log/train/", help="Where to write the Log of training")
parser.add_argument('--val_writer_path', type=str, default="./checkpoints/log/val/", help="Where to write the Log of validation")
args = parser.parse_args()

writer = SummaryWriter(log_dir=args.train_writer_path)
writer_val = SummaryWriter(log_dir=args.val_writer_path)
np.set_printoptions(threshold=np.inf)


def train_model(model, optimizer, scheduler, save_path, start_epoch, num_epochs=25):
    best_loss = 1e10
    iters = 0

    for epoch in range(start_epoch , start_epoch + num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            if phase == 'train':
                for defected, clear, defect_gt, rotation_gt, translation_gt in dataloader(batch_size)[phase]:
                    iters = iters + 1
                    defected = defected.to(device)
                    clear = clear.to(device)
                    defect_gt = defect_gt.to(device)
                    torch.autograd.set_detect_anomaly(True)

                    # defected.requires_grad = True
                    # clear.requires_grad = True
                    # defect_gt.requires_grad = True


                    # zero the parameter gradients
                    optimizer.zero_grad()

        # forward
                    loss_defect, defected_unet, clear_unet, clear_transformed, defection_map, defect_gt_blur, rot_cal = train_defect(defected, clear, defect_gt, model, phase, device )

        # backward + optimize only if in training phase:
                    if phase == 'train':
                        # if dataloader == "DPCNdataloader":
                            
                        if 90.00 in rot_cal or 0.00 in rot_cal:
                            print("bad image input")
                            del defected, clear, defect_gt, defected_unet, clear_unet, clear_transformed, defection_map, defect_gt_blur
                            torch.cuda.empty_cache()
                            continue
                        
                        loss_defect.backward(retain_graph=False)
                        optimizer.step()

                        
                        writer.add_scalar('LOSS DEFECT', loss_defect.detach().cpu().numpy(), iters)
                        # writer.add_scalar('LOSS ROTATION', loss_rot.detach().cpu().numpy(), iters)
                        # writer.add_scalar('LOSS ROTATION L1', loss_l1_rot.item(), iters)
                        writer.add_image("defected_input", defected[0,:,:].cpu(), iters)
                        writer.add_image("clear_input", clear[0,:,:].cpu(), iters)
                        writer.add_image("defection_map", defection_map[0,:,:].cpu(), iters)
                        writer.add_image("gt", defect_gt_blur[0,:,:].cpu(), iters)
                        writer.add_image("defected_unet", defected_unet[0,:,:].cpu(), iters)
                        writer.add_image("clear_unet", clear_unet[0,:,:].cpu(), iters)


                # statistics


            checkpoint = {'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}

            if phase == 'val':
                print("in val")
                loss_list = val_model(model, writer_val, iters, dsnt, dataloader, batch_size_val, device, epoch)
                epoch_loss = np.mean(loss_list)
                print("epoch_loss", epoch_loss)
                print("best_loss", best_loss)
                # print("accuracy = ", acc)
                if epoch_loss < best_loss:
                    is_best = True
                    best_loss = epoch_loss
                else:
                    is_best = False
                save_checkpoint(checkpoint, is_best, save_path)

               
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


        scheduler.step()
        
    print('Best val loss: {:4f}'.format(best_loss))

    return model



save_path = args.save_path
checkpoint_path = args.load_path
load_pretrained = args.load_pretrained
load_optimizer = args.load_optimizer
dsnt = args.use_dsnt
load_pretrained_mode = args.pretrained_mode
batch_size = args.batch_size_train
batch_size_val = args.batch_size_val
dataloader = generate_dataloader
device = torch.device("cuda:0" if not args.cpu else "cpu")
print("The devices that the code is running on:", device)
print("batch size is ",batch_size)


# to create models for rotations and translations for source images and template images
num_class = 1
start_epoch = 0
model = UNet(num_class)
model.to(device)
print(111)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-3)



exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)



# load pretrained model based on the input pretrained mode
if load_pretrained:
    if load_pretrained_mode == 'all':
        print("Load Optimizer?", load_optimizer)
        if load_optimizer:
            model, optimizer, start_epoch = load_checkpoint(\
                                            checkpoint_path, model, optimizer, device)
        else:
            model, _, start_epoch = load_checkpoint(\
                                            checkpoint_path, model, optimizer, device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-3)

model_template = train_model(model, optimizer, exp_lr_scheduler, save_path, start_epoch, num_epochs=700)
