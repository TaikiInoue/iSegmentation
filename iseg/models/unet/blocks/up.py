import torch
import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class Up(nn.Module, Builder):

    up: T.Module
    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - up: torch.nn - Upsample (torch.nn - ConvTranspose2d)
                - conv_bn_relu_0: iseg.models.unet.blocks - ConvNormReLU
                - conv_bn_relu_1: iseg.models.unet.blocks - ConvNormReLU
        """

        super(Up, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x_0: T.Tensor, x_1: T.Tensor) -> T.Tensor:

        x_0 = self.up(x_0)
        x_2 = torch.cat([x_1, x_0], dim=1)
        x_2 = self.conv_bn_relu_0(x_2)
        x_2 = self.conv_bn_relu_1(x_2)
        return x_2
