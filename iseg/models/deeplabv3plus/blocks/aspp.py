import torch
import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class ASPP(nn.Module, Builder):

    aspp_conv_0: T.Module
    aspp_conv_1: T.Module
    aspp_conv_2: T.Module
    aspp_conv_3: T.Module
    aspp_pool: T.Module
    conv_bn_relu: T.Module
    dropout: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - aspp_conv_0: iseg.blocks - ConvBnReLU
                - aspp_conv_1: iseg.blocks - ConvBnReLU
                - aspp_conv_2: iseg.blocks - ConvBnReLU
                - aspp_conv_3: iseg.blocks - ConvBnReLU
                - aspp_pool: iseg.models.deeplabv3plus.blocks - ASPPPool
                - conv_bn_relu: iseg.blocks - ConvBnReLU
                - dropout: torch.nn - Dropout
        """

        super(ASPP, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x_0 = self.aspp_conv_0(x)
        x_1 = self.aspp_conv_1(x)
        x_2 = self.aspp_conv_2(x)
        x_3 = self.aspp_conv_3(x)
        x_4 = self.aspp_pool(x)

        x_5 = torch.cat([x_0, x_1, x_2, x_3, x_4])
        x_5 = self.conv_bn_relu(x_5)
        x_5 = self.dropout(x_5)
        return x_5
