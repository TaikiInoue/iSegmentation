import torch
import torch.nn as nn

import iseg.typehint as T
from iseg.models import Builder


class ASPP(nn.Module, Builder):

    aspp_0: T.Module
    aspp_1: T.Module
    aspp_2: T.Module
    aspp_3: T.Module
    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module
    avgpool: T.Module
    interpolate: T.Module
    dropout: T.Module

    def __init__(self, object_cfg: T.ListConfig, backbone: T.Module):

        """
        Args:
            object_cfg (T.ListConfig):
                - aspp_0: iseg.blocks - ConvBnReLU
                - aspp_1: iseg.blocks - ConvBnReLU
                - aspp_2: iseg.blocks - ConvBnReLU
                - aspp_3: iseg.blocks - ConvBnReLU
                - conv_bn_relu_0 - iseg.blocks - ConvBnReLU
                - conv_bn_relu_1 - iseg.blocks - ConvBnReLU
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - interpolate: iseg.block - Interpolate
                - dropout: torch.nn - Dropout
        """

        super(ASPP, self).__init__()
        self.backbone = backbone

    def forward(self, x: T.Tensor) -> T.Tensor:

        x_0 = self.aspp_0(x)
        x_1 = self.aspp_1(x)
        x_2 = self.aspp_2(x)
        x_3 = self.aspp_3(x)
        x_4 = self.avgpool(x)
        x_4 = self.conv_bn_relu_0(x_4)
        x_4 = self.interpolate(x_4)

        x_5 = torch.cat([x_0, x_1, x_2, x_3, x_4])
        x_5 = self.conv_bn_relu_1(x_5)
        x_5 = self.dropout(x_5)
        return x_5
