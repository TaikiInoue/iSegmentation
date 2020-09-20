import torch

import iseg.typehint as T
from iseg.models import BasePart


class Up(BasePart):

    up: T.Module
    conv_norm_relu_0: T.Module
    conv_norm_relu_1: T.Module

    def __init__(self, part_cfg: T.ListConfig) -> None:

        """
        Args:
            part_cfg (T.ListConfig):
                - up: Upsample (or ConvTranspose2d)
                - conv_norm_relu_0: ConvNormReLU
                - conv_norm_relu_1: ConvNormReLU
        """

        super(Up, self).__init__()
        self.build_part(part_cfg)

    def forward(self, x_0: T.Tensor, x_1: T.Tensor) -> T.Tensor:

        x_0 = self.up(x_0)
        x_2 = torch.cat([x_1, x_0], dim=1)
        x_2 = self.conv_norm_relu_0(x_2)
        x_2 = self.conv_norm_relu_1(x_2)
        return x_2
