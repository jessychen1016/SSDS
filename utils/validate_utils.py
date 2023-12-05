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
from unet.loss import NCC_score
import torch.optim as optim
from data.dataset import *
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, PoseCalculator
from tensorboardX import SummaryWriter
from utils.utils import *
import matplotlib.pyplot as plt


def validate_defect(defected, clear, defect_gt, model, phase, device):
    print("                             ")
    print("                             Validating DR")
    print("                             ")
    
    with torch.no_grad():
    # to calculate the pose and transform the clear image 
        calculate_pose = PoseCalculator(device, trans = True, align = True)
        clear_transformed, rot_cal, scale_cal, trans_y_cal, trans_x_cal = calculate_pose(defected,clear)
    # to obtain a defection map by a new model
        # to concatinate the two images in the 'channel' channel
        synthesis_image = torch.cat((defected,clear_transformed),1)
        synthesis_unet = model(synthesis_image)
        _, rot_cal_unet, scale_cal, trans_y_cal, trans_x_cal = calculate_pose(synthesis_unet[:,0,:].unsqueeze(1),synthesis_unet[:,1,:].unsqueeze(1))
        defection_map = model.defect_recog(synthesis_unet)

        gauss_blur = kornia.filters.GaussianBlur2d((5, 5), (5, 5))
        defect_gt_blur = gauss_blur(defect_gt)

        # imshow(defected_unet[0,:,:])
        # plt.show()
        # imshow(clear_unet[0,:,:])
        # plt.show()        
        # imshow(clear_transformed[0,:,:])
        # plt.show()  
        # imshow(defect_gt_blur[0,:,:])
        # plt.show()

    # set the loss function:
        # compute_loss = torch.nn.KLDivLoss(reduction="sum").to(device)
        # compute_loss = torch.nn.BCEWithLogitsLoss(reduction="sum").to(device)
        # compute_loss_rot = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
        # compute_loss_scale = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
        compute_mse = torch.nn.MSELoss(reduction="sum").to(device)
        compute_l1=torch.nn.L1Loss().to(device)
        # compute_loss = torch.nn.MSELoss()
        # compute_loss=torch.nn.L1Loss()

        # loss_rot = compute_loss_rot(corr_final_rot,gt_angle)
        # loss_scale = compute_loss_scale(corr_final_scale,gt_scale)
        # loss_l1_rot = compute_l1(angle_softargmax, groundTruth_rot)
        # loss_l1_scale = compute_l1(scale_cal_argmax, groundTruth_scale)
        loss_mse = compute_mse(defection_map, defect_gt_blur)
        loss_angle = 1/compute_l1(rot_cal_unet,torch.zeros(rot_cal_unet.shape).to(device)+0.0001)
        loss_total = loss_mse + 0.001*loss_angle
        print("loss",loss_mse,"\n")


        return  loss_total, synthesis_unet[:,0,:].unsqueeze(1), synthesis_unet[:,1,:].unsqueeze(1), clear_transformed, defection_map, defect_gt_blur
