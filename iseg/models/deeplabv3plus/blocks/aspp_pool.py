import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class ASPPPool(nn.Module, Builder):

    avgpool: T.Module
    conv_bn_relu: T.Module
    interpolate: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - avgpool: nn.torch - AdaptiveAvgPool2d
                - conv_bn_relu: iseg.blocks - ConvBnReLU
                - interpolate: iseg.blocks - Interpolate
        """

        super(ASPPPool, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.avgpool(x)
        x = self.conv_bn_relu(x)
        x = self.interpolate(x)
        return x
