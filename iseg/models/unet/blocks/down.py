import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class Down(nn.Module, Builder):

    maxpool: T.Module
    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - maxpool: torch.nn - MaxPool2d
                - conv_bn_relu_0: iseg.models.unet.blocks - ConvNormReLU
                - conv_bn_relu_1: iseg.models.unet.blocks - ConvNormReLU
        """

        super(Down, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.maxpool(x)
        x = self.conv_bn_relu_0(x)
        x = self.conv_bn_relu_1(x)
        return x
