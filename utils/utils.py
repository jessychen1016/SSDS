import sys
import os
sys.path.append(os.path.abspath("../unet"))
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from unet.loss import NCC_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
import shutil
import math
from PIL import Image
import kornia
import cv2


def logpolar_filter(shape, device):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy ** 2 + xx ** 2)
    filt = 1.0 - np.cos(rads) ** 2
    #  This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    filt = torch.from_numpy(filt).to(device)
    return filt

def roll_n(X, axis, n):

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift = n_shift+1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def soft_argmax(input_tensor, device, alpha = 1000):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert input_tensor.dim()==5
    # alpha is the temperature tuning parameter, it should be pretty large but keep within
    # arange
    B,C,H,W,D = input_tensor.shape

    # to softmax the input tensor with softmax2d
    input_softmax = nn.functional.softmax(input_tensor.view(B,C,-1)/alpha,dim=2)
    # recover the shape
    input_softmax = input_softmax.view(input_tensor.shape)

    indices_kernel = torch.arange(start=0,end=H*W*D,step=1).unsqueeze(0)
    indices_kernel = indices_kernel.to(device)
    indices_kernel = indices_kernel.view((H,W,D))
    conv = input_softmax*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    Coords_3d = torch.stack([x,y,z],dim=2)
    return Coords_3d

def softmax2d(input, device, beta=10000):
    *_, h, w = input.shape
    
    input_orig = input.reshape(*_, h * w)
    beta_t = 100. / torch.max(input_orig).to(device)
    input_d = nn.functional.softmax(1000 * input_orig, dim=-1)
    soft_r = input_d.reshape(*_,h,w)
    # soft_r.retain_grad()
    # print("softmax grad =======", soft_r.grad)
    return soft_r

def GT_angle_convert(this_gt,size):
    for batch_num in range(this_gt.shape[0]):
        if this_gt[batch_num] > 90:
            this_gt[batch_num] = this_gt[batch_num].clone() - 90
        else:
            this_gt[batch_num] = this_gt[batch_num].clone() + 90
        this_gt[batch_num] = this_gt[batch_num].clone()*size/180
        this_gt[batch_num] = this_gt[batch_num].clone()//1 + (this_gt[batch_num].clone()%1+0.5)//1
        if this_gt[batch_num].long() == size:
            this_gt[batch_num] = this_gt[batch_num] - 1
    return this_gt.long()
def GT_scale_convert(scale_gt,logbase,size):
    for batch_num in range(scale_gt.shape[0]):
            scale_gt[batch_num] = torch.log10(1/scale_gt[batch_num].clone())/torch.log10(logbase)+128.
    return scale_gt.long()

def GT_trans_convert(this_trans, size):
    this_trans = (this_trans.clone() + size[0] // 2)
    # gt_converted = this_trans[:,1] * size[0] + this_trans[:,0]
    # gt_converted = this_trans[:,0]
    gt_converted = this_trans
# # create a gt for kldivloss
    # kldiv_gt = torch.zeros(this_trans.clone().shape[0],size[0],size[1])
    # gauss_blur = kornia.filters.GaussianBlur2d((5, 5), (5, 5))
    # for batch_num in range(this_trans.clone().shape[0]):
    #     kldiv_gt[batch_num, this_trans.clone()[batch_num,0].long(), this_trans.clone()[batch_num,1].long()] = 1

    # kldiv_gt = torch.unsqueeze(kldiv_gt.clone(), dim = 0)
    # kldiv_gt = kldiv_gt.permute(1,0,2,3)
    # kldiv_gt = gauss_blur(kldiv_gt.clone())
    # kldiv_gt = kldiv_gt.permute(1,0,2,3)
    # kldiv_gt = torch.squeeze(kldiv_gt.clone(), dim = 0)
    # (b, h, w) = kldiv_gt.shape
    # kldiv_gt = kldiv_gt.clone().reshape(b, h*w)
# # Create GT for Pooling data
#     gt_pooling = torch.floor(this_trans.clone()/4)
    return gt_converted.long()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] = metrics['bce']+bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] = metrics['dice']+dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] = metrics['loss']+loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
def imshow(tensor, title=None):
    image = tensor.cpu().detach().numpy()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    plt.tight_layout()
    plt.imshow(image, cmap="gray")
    plt.imshow(image, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
def heatmap_imshow(tensor, title=None):
    image = tensor.cpu().detach().numpy()  # we clone the tensor to not do changes on it
    image = gaussian_filter(image, sigma = 5, mode = 'nearest')
    plt.imshow(image, cmap="jet", interpolation="hamming")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
def align_image(template, source):
    template = template.cpu().detach().numpy()
    source = source.cpu().detach().numpy()
    template = template.squeeze(0)  # remove the fake batch dimension
    source = source.squeeze(0)  # remove the fake batch dimension
    dst = cv2.addWeighted(template, 1, source, 0.6, 0)
    # plt.imshow(dst, cmap="gray")
    # plt.show()  # pause a bit so that plots are updated
    return dst

def plot_and_save_result(template, source, rotated, dst):
    template = template.cpu().detach().numpy()
    source = source.cpu().detach().numpy()
    template = template.squeeze(0)  # remove the fake batch dimension
    source = source.squeeze(0)  # remove the fake batch dimension
    rotated = rotated.cpu().detach().numpy()
    rotated = rotated.squeeze(0)


    result = plt.figure()
    result_t = result.add_subplot(1,4,1)
    result_t.set_title("Template")
    result_t.imshow(template, cmap="gray").axes.get_xaxis().set_visible(False)
    result_t.imshow(template, cmap="gray").axes.get_yaxis().set_visible(False)

    result_s = result.add_subplot(1,4,2)
    result_s.set_title("Source")
    result_s.imshow(source, cmap="gray").axes.get_xaxis().set_visible(False)
    result_s.imshow(source, cmap="gray").axes.get_yaxis().set_visible(False)

    result_r = result.add_subplot(1,4,3)
    result_r.set_title("Rotated Source")
    result_r.imshow(rotated, cmap="gray").axes.get_xaxis().set_visible(False)
    result_r.imshow(rotated, cmap="gray").axes.get_yaxis().set_visible(False)

    result_d = result.add_subplot(1,4,4)
    result_d.set_title("Destination")
    result_d.imshow(dst, cmap="gray").axes.get_xaxis().set_visible(False)
    result_d.imshow(dst, cmap="gray").axes.get_yaxis().set_visible(False)
    plt.savefig("Result.png")
    plt.show()
   


def save_checkpoint(state, is_best, checkpoint_dir):
    file_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, file_path)
    if is_best:
        best_fpath = checkpoint_dir + 'best_model.pt'
        shutil.copyfile(file_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer, device):

    if (device == torch.device('cpu')):
        print("using cpu")
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath, map_location=device)


    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])



    return model, optimizer, checkpoint['epoch']
