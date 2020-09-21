import torch.nn as nn

import iseg.typehint as T
from iseg.models import Builder


class DeepLabV3Plus(nn.Module, Builder):

    aspp: T.Module
    decoder: T.Module
    interpolate: T.Module

    def __init__(self, object_cfg: T.ListConfig, backbone: T.Module):

        """
        Args:
            object_cfg (T.ListConfig):
                - aspp: iseg.models.deeplabv3plus.blocks - ASPP
                - decoder: iseg.models.deeplabv3plus.blocks - Decoder
                - interpolate: iseg.blocks - Interpolate
            backbone (T.Module):
        """

        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x, feature_dict = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, feature_dict)
        x = self.interpolate(x)

        return x
