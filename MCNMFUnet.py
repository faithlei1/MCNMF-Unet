import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from maxim_component import *

__all__ = ['UNet', 'NestedUNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
import einops


class MCNMF_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])  # 3--32
        self.convf1_0 = VGGBlock(nb_filter[0], nb_filter[1],nb_filter[1])  # 32--64
        self.conv0_1 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.CGB0 = CrossGatingBlock(x_in=32, y_in=32, out_features=32,
                                     patch_size=[256, 256],
                                     block_size=(8, 8), grid_size=(8, 8),
                                     dropout_rate=0.0, input_proj_factor=2,
                                     upsample_y=False, use_bias=True)

        self.conv1_0 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.convf2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.CGB1 = CrossGatingBlock(x_in=64, y_in=64, out_features=64,
                                     patch_size=[128, 128],
                                     block_size=(8, 8), grid_size=(8, 8),
                                     dropout_rate=0.0, input_proj_factor=2,
                                     upsample_y=False, use_bias=True)

        self.conv2_0 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.convf3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.CGB2 = CrossGatingBlock(x_in=128, y_in=128, out_features=128,
                                     patch_size=[64, 64],
                                     block_size=(4, 4), grid_size=(4, 4),
                                     dropout_rate=0.0, input_proj_factor=2,
                                     upsample_y=False, use_bias=True)

        self.conv3_0 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[3], nb_filter[3])
        self.convf4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.CGB3 = CrossGatingBlock(x_in=256, y_in=256, out_features=256,
                                     patch_size=[32, 32],
                                     block_size=(4, 4), grid_size=(4, 4),
                                     dropout_rate=0.0, input_proj_factor=2,
                                     upsample_y=False, use_bias=True)

        self.conv4_0 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.deconvf3_0 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.deconv4_1 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.deCGB4 = CrossGatingBlock(x_in=512, y_in=512, out_features=512,
                                       patch_size=[16, 16],
                                       block_size=(2, 2), grid_size=(2, 2),
                                       dropout_rate=0.0, input_proj_factor=2,
                                       upsample_y=False, use_bias=True)

        self.deconv3_0 = VGGBlock(2*nb_filter[3]+ nb_filter[4], nb_filter[3], nb_filter[3])
        self.deconvf2_0 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.deconv3_1 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.deCGB3 = CrossGatingBlock(x_in=256, y_in=256, out_features=256,
                                       patch_size=[32, 32],
                                       block_size=(4, 4), grid_size=(4, 4),
                                       dropout_rate=0.0, input_proj_factor=2,
                                       upsample_y=False, use_bias=True)

        self.deconv2_0 = VGGBlock(2*nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.deconvf1_0 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.deconv2_1 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.deCGB2 = CrossGatingBlock(x_in=128, y_in=128, out_features=128,
                                       patch_size=[64, 64],
                                       block_size=(4, 4), grid_size=(4, 4),
                                       dropout_rate=0.0, input_proj_factor=2,
                                       upsample_y=False, use_bias=True)

        self.deconv1_0 = VGGBlock(2*nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.deconvf0_0 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.deconv1_1 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.deCGB1 = CrossGatingBlock(x_in=64, y_in=64, out_features=64,
                                       patch_size=[128, 128],
                                       block_size=(8, 8), grid_size=(8, 8),
                                       dropout_rate=0.0, input_proj_factor=2,
                                       upsample_y=False, use_bias=True)

        self.deconv0_0 = VGGBlock(2*nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


        self.Conv_down0 = Conv_down(32, 64, use_bias=True)
        self.Conv_down1 = Conv_down(64, 128, use_bias=True)
        self.Conv_down2 = Conv_down(128, 256, use_bias=True)
        self.Conv_down3 = Conv_down(256, 512, use_bias=True)

        self.ConvT_up3 = ConvT_up(512, 256, use_bias=True)
        self.ConvT_up2 = ConvT_up(256, 128, use_bias=True)
        self.ConvT_up1 = ConvT_up(128, 64, use_bias=True)
        self.ConvT_up0 = ConvT_up(64, 32, use_bias=True)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        fx1_0 = self.convf1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(fx1_0))
        CGBx0, _ = self.CGB0(x0_0, x0_1)

        x1_0 = self.conv1_0(torch.cat([self.pool(CGBx0), fx1_0], 1))
        fx2_0 = self.convf2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(fx2_0))
        CGBx1, _ = self.CGB1(x1_0, x1_1)

        x2_0 = self.conv2_0(torch.cat([self.pool(CGBx1), fx2_0], 1))
        fx3_0 = self.convf3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(fx3_0))
        CGBx2, _ = self.CGB2(x2_0, x2_1)

        x3_0 = self.conv3_0(torch.cat([self.pool(CGBx2), fx3_0], 1))
        fx4_0 = self.convf4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(fx4_0))
        CGBx3, _ = self.CGB3(x3_0, x3_1)

        x4_0 = self.conv4_0(torch.cat([self.pool(CGBx3), fx4_0], 1))
        fy3_0 = self.deconvf3_0(torch.cat([self.up(x4_0), CGBx3], 1))
        y4_1 = self.deconv4_1(self.pool(fy3_0))
        CGBy4, _ = self.deCGB4(x4_0, y4_1)

        y3_0 = self.deconv3_0(torch.cat([self.up(CGBy4), CGBx3, fy3_0], 1))
        fy2_0 = self.deconvf2_0(torch.cat([self.up(y3_0), CGBx2], 1))
        y3_1 = self.deconv3_1(self.pool(fy2_0))
        CGBy3, _ = self.deCGB3(y3_0, y3_1)

        y2_0 = self.deconv2_0(torch.cat([self.up(CGBy3), CGBx2, fy2_0], 1))
        fy1_0 = self.deconvf1_0(torch.cat([self.up(y2_0), CGBx1], 1))
        y2_1 = self.deconv2_1(self.pool(fy1_0))
        CGBy2, _ = self.deCGB2(y2_0, y2_1)

        y1_0 = self.deconv1_0(torch.cat([self.up(CGBy2), CGBx1, fy1_0], 1))
        fy0_0 = self.deconvf0_0(torch.cat([self.up(y1_0), CGBx0], 1))
        y1_1 = self.deconv1_1(self.pool(fy0_0))
        CGBy1, _ = self.deCGB1(y1_0, y1_1)

        y0_0 = self.deconv0_0(torch.cat([self.up(CGBy1), CGBx0, fy0_0], 1))
        output = self.final(y0_0)
        return output




class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


#TODO CGB
class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self,x_in:int,y_in:int,out_features:int,patch_size:[int,int],block_size:[int,int],grid_size:[int,int],dropout_rate:float=0.0,input_proj_factor:int=2,upsample_y:bool=True,use_bias:bool=True):
        super(CrossGatingBlock, self).__init__()
        self.IN_x = x_in
        self.IN_y = y_in
        self._h = patch_size[0]
        self._w = patch_size[1]
        self.features = out_features
        self.block_size=block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.Conv1X1_y = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y = nn.Linear(self.features,self.features,bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)

        self.ConvT = conv_T_y_2_x(self.IN_y,self.IN_x)

        self.fun_gx = GetSpatialGatingWeights_2D_4axi(nIn=self.features,Nout=self.features,H_size=self._h,W_size=self._w,
                                                         block_size=self.block_size, grid_size=self.grid_size,
                                                                          input_proj_factor=2,dropout_rate=0.0,use_bias=True)

        self.fun_gy = GetSpatialGatingWeights_2D_4axi(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                         block_size=self.block_size, grid_size=self.grid_size,
                                                                          input_proj_factor=2, dropout_rate=0.0, use_bias=True)
    def forward(self, x, y):
        if self.upsample_y:
            y = self.ConvT(x,y)
        x = self.Conv1X1_x(x)
        n,num_channels,h,w = x.shape
        y = self.Conv1X1_y(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n x h x w x c
        y = y.permute(0, 2, 3, 1)
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)
        #__init__(self,nIn:int,Nout:int,H_size:int=128,W_size:int=128,block_size:[int,...]=[2,2],grid_size:[int,...]=[2,2],input_proj_factor:int=2,dropout_rate:float=0.0,use_bias:bool=True):

        gx = self.fun_gx(x)
        # n x h x w x c
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.Linear_y(y)
        y = self.Gelu_y(y)

        gy = self.fun_gy(y)
        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.Linear_y_end(y)
        y = self.dropout_y(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.Linear_y_end(x)
        x = self.dropout_x(x)
        x = x + y + shortcut_x
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        return x, y



class conv_T_y_2_x(nn.Module):
    """ Unified y Dimensional to x """
    def __init__(self,y_nIn,x_nOut):
        super(conv_T_y_2_x, self).__init__()
        self.x_c = x_nOut
        self.y_c = y_nIn
        self.convT = nn.ConvTranspose2d(in_channels=self.y_c, out_channels=self.x_c, kernel_size=(3,3),
                                        stride=(2,2))
    def forward(self,x,y):
        y = self.convT(y)
        _, _, h, w, = x.shape
        y = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        return y


class GetSpatialGatingWeights_2D_4axi(nn.Module):
    """Get gating weights for cross-gating MLP block."""
    def __init__(self,nIn:int,Nout:int,H_size:int=128,W_size:int=128,block_size:[int,int]=[2,2],grid_size:[int,int]=[2,2],input_proj_factor:int=2,dropout_rate:float=0.0,use_bias:bool=True):
        super(GetSpatialGatingWeights_2D_4axi,self).__init__()
        self.H = H_size
        self.W = W_size
        self.IN  = nIn
        self.OUT = Nout
        self.block_size = block_size
        self.grid_size = grid_size
        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_head = nn.Linear(self.IN,self.IN*input_proj_factor,bias=use_bias) # nn.Linear（）是用于设置网络中的全连接层
        self.Linear_end  = nn.Linear(self.IN*self.input_proj_factor,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP1 = nn.Linear((self.grid_size[0]*self.grid_size[1]),(self.grid_size[0]*self.grid_size[1]),bias=use_bias)
        self.Linear_grid_MLP2 = nn.Linear(((self.grid_size[0]//2) * (self.grid_size[1]//2)),((self.grid_size[0]//2) * (self.grid_size[1]//2)), bias=use_bias)
        self.Linear_Block_MLP1 = nn.Linear((self.block_size[0]*self.block_size[1]),(self.block_size[0]*self.block_size[1]),bias=use_bias)
        self.Linear_Block_MLP2 = nn.Linear(((self.block_size[0]//2) * (self.block_size[1]//2)),((self.block_size[0]//2) * (self.block_size[1]//2)), bias=use_bias)

    def forward(self, x):
        n, h, w,num_channels = x.shape
        # n x h x w x c
        # input projection
        x = self.LayerNorm(x.float())
        x = self.Linear_head(x) # 2C
        x = self.Gelu(x)
        # u, v = np.split(x, 2, axis=-1)
        u1, u2, v1, v2 = torch.chunk(x,4,dim =-1)

        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        u1 = block_images_einops(u1, patch_size=(fh, fw))

        u1 = u1.permute(0,3,2,1)
        u1 = self.Linear_grid_MLP1(u1)
        u1 = u1.permute(0,3,2,1)
        u1 = unblock_images_einops(u1, grid_size=(gh, gw), patch_size=(fh, fw))

        gh, gw = self.grid_size[0]//2 ,self.grid_size[1]//2
        fh, fw = h // gh, w // gw
        u2 = block_images_einops(u2, patch_size=(fh, fw))

        u2 = u2.permute(0, 3, 2, 1)
        u2 = self.Linear_grid_MLP2(u2)
        u2 = u2.permute(0, 3, 2, 1)
        u2 = unblock_images_einops(u2, grid_size=(gh, gw), patch_size=(fh, fw))

        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        v1 = block_images_einops(v1, patch_size=(fh, fw))
        dim_v1 = v1.shape[-2]
        v1 = v1.permute(0, 1, 3, 2)
        v1 = self.Linear_Block_MLP1(v1)
        v1 = v1.permute(0, 1, 3, 2)
        v1 = unblock_images_einops(v1, grid_size=(gh, gw), patch_size=(fh, fw))

        fh, fw = self.grid_size[0]//2 ,self.grid_size[1]//2
        gh, gw = h // fh, w // fw
        v2 = block_images_einops(v2, patch_size=(fh, fw))
        dim_v2 = v2.shape[-2]
        v2 = v2.permute(0, 1, 3, 2)
        v2 = self.Linear_Block_MLP2(v2)
        v2 = v2.permute(0, 1, 3, 2)
        v2 = unblock_images_einops(v2, grid_size=(gh, gw), patch_size=(fh, fw))



        x = torch.cat((u1,u2,v1,v2),dim=-1)
        x = self.Linear_end(x)
        x = self.dropout(x)
        return x

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c", # 现将基于 fh,fw 窗口 大小图片拉成一维，共有 ghxgw 个
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1]) # 将维度变为 n (gh gw) (fh fw) c
  return x

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x