import torch
import torch.nn as nn

import iseg.typehint as T
from iseg.models import Builder


class ResNet50(nn.Module, Builder):

    first_conv: T.Module
    res_0: T.Module
    res_1: T.Module
    res_2: T.Module
    res_3: T.Module
    avgpool: T.Module
    fc: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - first_conv: iseg.backbone.resnet.blocks - FirstConv
                - res_0: iseg.backbone.resnet.blocks - Res
                - res_1: iseg.backbone.resnet.blocks - Res
                - res_2: iseg.backbone.resnet.blocks - Res
                - res_3: iseg.backbone.resnet.blocks - Res
                - avgpool: torch.nn - AdaptiveAvgPool2d
                - fc: torch.nn - Linear
        """

        super(ResNet50, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.first_conv(x)
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
