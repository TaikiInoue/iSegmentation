import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class Bottleneck(nn.Module, Builder):

    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module
    conv_bn: T.Module
    downsample: T.Module
    relu: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu_0: iseg.blocks - ConvBnReLU
                - conv_bn_relu_1: iseg.blocks - ConvBnReLU
                - conv_bn: iseg.blocks - ConvBn
                - downsample: iseg.blocks - ConvBn
                - relu: torch.nn - ReLU
        """

        super(Bottleneck, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        identity = x
        out = self.conv_bn_relu_0(x)
        out = self.conv_bn_relu_1(out)
        out = self.conv_bn(out)

        if hasattr(self, "downsample"):
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out
