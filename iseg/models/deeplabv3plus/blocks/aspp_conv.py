import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class ASPPConv(nn.Module, Builder):

    conv_bn_relu: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu: iseg.blocks - ConvBnReLU
        """

        super(ASPPConv, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.conv_bn_relu(x)
        return x
