import sys
import os
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
import numpy as np
from phase_correlation.phase_corr import phase_corr
from log_polar.log_polar import polar_transformer
from utils.utils import *
print("sys path", sys.path)
from utils.utils import *
import kornia

class LogPolar(nn.Module):
    def __init__(self, out_size, device):
        super(LogPolar, self).__init__()
        self.out_size = out_size
        self.device = device

    def forward(self, input):
        return polar_transformer(input, self.out_size, self.device) 


class PhaseCorr(nn.Module):
    def __init__(self, device, logbase, trans=False):
        super(PhaseCorr, self).__init__()
        self.device = device
        self.logbase = logbase
        self.trans = trans

    def forward(self, template, source):
        return phase_corr(template, source, self.device, self.logbase, trans=self.trans)

class FFT2(nn.Module):
    def __init__(self, device):
        super(FFT2, self).__init__()
        self.device = device

    def forward(self, input):

        median_output = torch.rfft(input, 2, onesided=False)
        median_output_r = median_output[:, :, :, 0]
        median_output_i = median_output[:, :, :, 1]
        # print("median_output r", median_output_r)
        # print("median_output i", median_output_i)
        output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2 + 1e-15)
        # output = median_outputW_r
        output = fftshift2d(output)
        # h = logpolar_filter((output.shape[1],output.shape[2]), self.device)
        # output = output.squeeze(0) * h
        # output = output.unsqueeze(-1)
        output = output.unsqueeze(-1)
        return output

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
    # params for Feature Extractor
        self.dconv_down1_defect = double_conv(2, 64)
        self.dconv_down1 = double_conv(2, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 2, 1)

    # params for Defection Recognitation
        self.dconv_down1_defect = double_conv(2, 64)
        self.dconv_down2_defect = double_conv(64, 128)
        self.dconv_down3_defect = double_conv(128, 256)
        self.dconv_down4_defect = double_conv(256, 512)

        self.maxpool_defect = nn.MaxPool2d(2)
        self.upsample_defect = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        

        self.dconv_up3_defect = double_conv(256 + 512, 256)
        self.dconv_up2_defect = double_conv(128 + 256, 128)
        self.dconv_up1_defect = double_conv(128 + 64, 64)
        
        self.conv_last_defect = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)   
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    def defect_recog(self, x):
        conv1 = self.dconv_down1_defect(x)
        x = self.maxpool_defect(conv1)

        conv2 = self.dconv_down2_defect(x)
        x = self.maxpool_defect(conv2)
        
        conv3 = self.dconv_down3_defect(x)
        x = self.maxpool_defect(conv3)   
        
        x = self.dconv_down4_defect(x)
        
        x = self.upsample_defect(x)   
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3_defect(x)
        x = self.upsample_defect(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2_defect(x)
        x = self.upsample_defect(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1_defect(x)
        
        out = self.conv_last_defect(x)
        
        return out        

class Corr2Softmax(nn.Module):

    def __init__(self, weight, bias):
        super(Corr2Softmax, self).__init__()
        softmax_w = torch.tensor((weight), requires_grad=True)
        softmax_b = torch.tensor((bias), requires_grad=True)
        self.softmax_w = torch.nn.Parameter(softmax_w)
        self.softmax_b = torch.nn.Parameter(softmax_b)
        self.register_parameter("softmax_w",self.softmax_w)
        self.register_parameter("softmax_b",self.softmax_b)
    def forward(self, x):
        x1 = self.softmax_w*x + self.softmax_b
        # print("w = ",self.softmax_w, "b = ",self.softmax_b)
        # x1 = 1000. * x
        return x1

class PoseCalculator(nn.Module):
    def __init__(self, device, trans=True, align = False):
        super(PoseCalculator, self).__init__()
        self.trans = trans
        self.device = device
        self.align = align
    def forward(self, defected, clear):

        defected_rot = defected.permute(0,2,3,1)
        clear_rot = clear.permute(0,2,3,1)

        
        defected_rot = defected_rot.squeeze(-1)
        clear_rot = clear_rot.squeeze(-1)

        fft_layer = FFT2(self.device)
        template_fft = fft_layer(defected_rot)
        source_fft = fft_layer(clear_rot) # [B,H,W,1]

        h = logpolar_filter((source_fft.shape[1],source_fft.shape[2]), self.device)#highpass((source.shape[1],source.shape[2])) # [H,W]
        template_fft = template_fft.squeeze(-1) * h
        source_fft = source_fft.squeeze(-1) * h
        
        template_fft = template_fft.unsqueeze(-1)
        source_fft = source_fft.unsqueeze(-1)

        # for tensorboard visualize
        template_fft_visual = template_fft.permute(0,3,1,2)
        source_fft_visual = source_fft.permute(0,3,1,2)
        logpolar_layer = LogPolar((template_fft.shape[1], template_fft.shape[2]), self.device)
        template_logpolar, logbase_rot = logpolar_layer(template_fft)
        source_logpolar, logbase_rot = logpolar_layer(source_fft)

        # for tensorboard visualize
        template_logpolar_visual = template_logpolar.permute(0,3,1,2)
        source_logpolar_visual = source_logpolar.permute(0,3,1,2)

        template_logpolar = template_logpolar.squeeze(-1)
        source_logpolar = source_logpolar.squeeze(-1)
        phase_corr_layer_rs = PhaseCorr(self.device, logbase_rot)
        rotation_argmax, scale_cal_argmax, softmax_result_rot, corr_result_rot = phase_corr_layer_rs(template_logpolar, source_logpolar)


        corr_final_rot = corr_result_rot.clone()
        # corr_visual_rot = corr_final_rot.unsqueeze(-1)
        # corr_visual_rot = corr_visual_rot.permute(0,3,1,2)
        corr_final_rot_raw = torch.sum(corr_final_rot, 2, keepdim=False)
        # corr_final_rot = model_corr2softmax_rot(corr_final_rot_raw)


        corr_final_scale = corr_result_rot.clone()
        corr_final_scale_raw = torch.sum(corr_final_scale,1,keepdim=False)
        # corr_final_scale = model_corr2softmax_rot(corr_final_scale_raw)
        # # print("corr_rot shape",corr_final_scale.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).shape)

        angle_softargmax = soft_argmax(corr_final_rot_raw.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), self.device, alpha=1e-2)[:,0,0]
        scale_softargmax = soft_argmax(corr_final_scale_raw.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), self.device, alpha=1e-2)[:,0,0]

        # in the board checking case, scale is set to a fix number "1"
        scale_softargmax = scale_softargmax *0. + 1.
        print("scale_softargmax",scale_softargmax)

        angle_softargmax = angle_softargmax*180.00/corr_result_rot.shape[1]
        for batch_num in range(angle_softargmax.shape[0]):
            if angle_softargmax[batch_num].item() >= 90:
                angle_softargmax[batch_num] -= 90.00
            else:
                angle_softargmax[batch_num] += 90.00

        print("angle softargmax",angle_softargmax)  


        if not self.trans:

            return _,angle_softargmax, scale_softargmax, _, _

        else:
           
            b, c, h, w = clear.shape
            center = torch.ones(b,2).to(self.device)
            center[:, 0] = h // 2
            center[:, 1] = w // 2
            angle_rot = torch.ones(b).to(self.device) * (-angle_softargmax.to(self.device))
            scale_rot = torch.ones(b).to(self.device)
            rot_mat = kornia.get_rotation_matrix2d(center, angle_rot, scale_rot)
            clear = kornia.warp_affine(clear.to(self.device), rot_mat, dsize=(h, w))

            template_unet_trans = defected.permute(0,2,3,1)
            source_unet_trans = clear.permute(0,2,3,1)

            template_unet_trans = template_unet_trans.squeeze(-1)
            source_unet_trans = source_unet_trans.squeeze(-1)

            (b, h, w) = template_unet_trans.shape
            logbase_trans = torch.tensor(1.)
            phase_corr_layer_xy = PhaseCorr(self.device, logbase_trans, trans=True)
            t0, t1, softmax_result_trans, corr_result_trans = phase_corr_layer_xy(template_unet_trans.to(self.device), source_unet_trans.to(self.device))


            corr_final_trans = corr_result_trans.clone()
            corr_y_raw = torch.sum(corr_final_trans.clone(), 2, keepdim=False)
            # corr_2d = corr_final_trans.clone().reshape(b, h*w)
            # corr_2d = model_corr2softmax(corr_2d)
            # corr_y = model_corr2softmax_trans(corr_y_raw)
            transformation_y = soft_argmax(corr_y_raw.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), self.device, alpha=1e-2)[:,0,0]
            transformation_y = transformation_y - h/2
            # transformation_y = torch.argmax(corr_y, dim=-1)

            corr_x_raw = torch.sum(corr_final_trans.clone(), 1, keepdim=False)
            # corr_final_trans = corr_final_trans.reshape(b, h*w)
            # corr_x = model_corr2softmax_trans(corr_x)
            transformation_x = soft_argmax(corr_x_raw.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), self.device, alpha=1e-2)[:,0,0]
            transformation_x = transformation_x - w/2

            print("trans x", transformation_x)

            print("trans y", transformation_y)

            if not self.align:
                return _, angle_softargmax, scale_softargmax, transformation_y, transformation_x

            else:
                rotation_mat = torch.Tensor([[1,0],[0,1]]).unsqueeze(0).repeat(b,1,1).to(self.device)
                translation_mat = torch.cat((-transformation_x.unsqueeze(-1), -transformation_y.unsqueeze(-1)),1).unsqueeze(-1)
                affine_mat = torch.cat((rotation_mat,translation_mat),-1)
                clear_transformed = kornia.warp_affine(clear.to(self.device), affine_mat, dsize=(h, w))

                return clear_transformed, angle_softargmax, scale_softargmax, transformation_y, transformation_x