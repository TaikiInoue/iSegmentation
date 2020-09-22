import torch.nn as nn
from omegaconf import OmegaConf

import iseg.models.backbones
import iseg.types as T
from iseg.models import Builder


class DeepLabV3Plus(nn.Module, Builder):

    aspp: T.Module
    decoder: T.Module
    interpolate: T.Module

    def __init__(self, yaml_path: str) -> None:

        """
        Args:
            yaml_path (str): Path to yaml file

        deeplabv3plus.yaml:
            backbone:
              name: ResNet50
              yaml:
              low_feature: res_0
              high_feature: res_3
            deeplabv3plus:
            - aspp: iseg.models.deeplabv3plus.blocks - ASPP
            - decoder: iseg.models.deeplabv3plus.blocks - Decoder
            - interpolate: iseg.blocks - Interpolate
        """

        super(DeepLabV3Plus, self).__init__()

        cfg = OmegaConf.load(yaml_path)
        backbone_cfg = cfg.backbone
        backbone_cls = getattr(iseg.models.backbone, backbone_cfg.name)
        self.backbone = backbone_cls(backbone_cfg.yaml)
        self.low_feature = backbone_cfg.low_feature
        self.high_feature = backbone_cfg.high_feature
        self.build_blocks(cfg.deeplabv3plus)

    def forward(self, x: T.Tensor) -> T.Tensor:

        _, feature_dict = self.backbone(x)
        x_0 = self.aspp(feature_dict[self.high_feature])
        x_1 = self.decoder(x_0, feature_dict[self.low_feature])

        return x_1
