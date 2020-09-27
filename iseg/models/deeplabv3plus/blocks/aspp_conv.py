import iseg.types as T
import torch.nn as nn
from iseg.models import Builder


class ASPPConv(nn.Module, Builder):

    depthwise_conv: T.Module
    pointwise_conv: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - depthwise_conv: iseg.blocks - ConvBnReLU
                - pointwise_conv: iseg.blocks - ConvBnReLU
        """

        super(ASPPConv, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
