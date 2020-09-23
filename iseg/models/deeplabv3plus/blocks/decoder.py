import torch
import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class Decoder(nn.Module, Builder):

    interpolate_0: T.Module
    interpolate_1: T.Module
    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module
    conv: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - interpolate_0: iseg.blocks - Interpolate
                - interpolate_1: iseg.blocks - Interpolate
                - conv_bn_relu_0: iseg.blocks - ConvBnReLU
                - conv_bn_relu_1: iseg.blocks - ConvBnReLU
                - conv: torch.nn - Conv2d
        """

        super(Decoder, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor, low_feature: T.Tensor) -> T.Tensor:

        x = self.interpolate_0(x)
        low_feature = self.conv_bn_relu_0(low_feature)
        out = torch.cat([x, low_feature], dim=1)
        out = self.conv_bn_relu_1(out)
        out = self.conv(out)
        out = self.interpolate_1(out)
        return out
