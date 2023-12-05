import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# this is a ncc calculator embedded with dot product 
class NCC_score(nn.Module):

    def __init__(self,if_sum=False):
        super(NCC_score, self).__init__()
        self.normalization_image = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
        self.sum = if_sum

    def forward(self, image1, image2):

    	# image1 and image2 are supposed to be of the same shape [B,C,H,W]
        # print(image1)
        b, c, h, w = image1.shape
        image1_norm = self.normalization_image(image1)
        image2_norm = self.normalization_image(image2)
        # print(image1_norm)
        correlation_score = torch.matmul(image1_norm.view(b, 1, c*h*w), image2_norm.view(b, c*h*w, 1))
        correlation_score /= (h*w*c)

        # the range of correlation_score is [-1,1] from totally unrelevant to the same
        # turn the range of correlation_score from [-1,1] to (infinite,0]
        loss_ncc = 2/(correlation_score+1)-1
        if self.sum:
            return loss_ncc.sum()
        else:
            return loss_ncc