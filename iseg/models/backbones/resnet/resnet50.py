import torch
import torch.nn as nn

import iseg.types as T
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

    def forward(self, x: T.Tensor) -> T.Tuple[T.Dict[str, T.Tensor], T.Tensor]:

        x = self.first_conv(x)
        x_res_0 = self.res_0(x)
        x_res_1 = self.res_1(x_res_0)
        x_res_2 = self.res_2(x_res_1)
        x_res_3 = self.res_3(x_res_2)
        y = self.avgpool(x_res_3)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)

        feature_dict = {"res_0": x_res_0, "res_1": x_res_1, "res_2": x_res_2, "res_3": x_res_3}
        return (feature_dict, y)
