from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from truenet_tumseg.utils import truenet_tumseg_model_utils

#=========================================================================================
# Triplanar U-Net ensemble network (TrUE-Net) model
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================

class TrUENetTumSeg(nn.Module):
    '''
    TrUE-Net model definition
    '''
    def __init__(self, n_channels, n_classes, init_channels, plane='axial', bilinear=False):
        super(TrUENetTumSeg, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = truenet_tumseg_model_utils.OutConv(n_channels, 3)
        if plane == 'axial' or plane == 'tc':
            self.convfirst = truenet_tumseg_model_utils.DoubleConv(3, init_channels, 3)
        else:
            self.convfirst = truenet_tumseg_model_utils.DoubleConv(3, init_channels, 5)
        self.down1 = truenet_tumseg_model_utils.Down(init_channels, init_channels*2, 3)
        self.down2 = truenet_tumseg_model_utils.Down(init_channels*2, init_channels*4, 3)
        self.down3 = truenet_tumseg_model_utils.Down(init_channels*4, init_channels*8, 3)
        self.up3 = truenet_tumseg_model_utils.Up(init_channels*8, init_channels*4, 3, bilinear)
        self.up2 = truenet_tumseg_model_utils.Up(init_channels*4, init_channels*2, 3, bilinear)
        self.up1 = truenet_tumseg_model_utils.Up(init_channels*2, init_channels, 3, bilinear)
        self.outconv = truenet_tumseg_model_utils.OutConv(init_channels, n_classes)

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        return logits


