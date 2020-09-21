import torch.nn as nn

import iseg.typehint as T
from iseg.models import Builder


class LastConv(nn.Module, Builder):

    conv_bn_relu: T.Module

    def __init__(self, object_cfg: T.ListConfig):

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu: iseg.models.unet.blocks - ConvNormReLU
        """

        super(LastConv, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.conv_bn_relu(x)
