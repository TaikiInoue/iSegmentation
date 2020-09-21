import torch
import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class Decoder(nn.Module, Builder):

    conv_bn_relu_0: T.Module
    conv_bn_relu_1: T.Module
    conv_bn_relu_2: T.Module
    dropout_0: T.Module
    dropout_1: T.Module
    interpolate: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu_0: iseg.blocks - ConvBnReLU
                - conv_bn_relu_1: iseg.blocks - ConvBnReLU
                - conv_bn_relu_2: iseg.blocks - ConvBnReLU
                - dropout_0: torch.nn - Dropout
                - dropout_1: torch.nn - Dropout
                - interpolate: iseg.blocks - Interpolate
        """

        super(Decoder, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor, feature_dict: T.Dict[str, T.Tensor]) -> T.Tensor:

        x = self.interpolate(x)
        y = feature_dict["hoge"]
        y = self.conv_bn_relu_0(y)

        z = torch.cat([x, y], dim=1)
        z = self.conv_bn_relu_1(z)
        z = self.dropout_0(z)
        z = self.conv_bn_relu_1(z)
        z = self.dropout_1(z)
        return z
