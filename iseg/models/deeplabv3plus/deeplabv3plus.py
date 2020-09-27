import iseg.models.backbones
import iseg.types as T
import torch.nn as nn
from iseg.models import Builder
from omegaconf import OmegaConf


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
        self.low_feature = backbone_cfg.low_feature
        self.high_feature = backbone_cfg.high_feature

        backbone_cls = getattr(iseg.models.backbones, backbone_cfg.name)
        backbone_cfg = OmegaConf.load(backbone_cfg.yaml)
        self.backbone = backbone_cls(backbone_cfg)

        self.build_blocks(cfg.deeplabv3plus)

    def forward(self, x: T.Tensor) -> T.Tensor:

        # Update self.interpolate.size from None to (h, w)
        _, _, h, w = x.shape
        self.interpolate.size = (h, w)

        feature_dict, _ = self.backbone(x)
        x = self.aspp(feature_dict[self.high_feature])
        x = self.decoder(x, feature_dict[self.low_feature])
        x = self.interpolate(x)
        return x
