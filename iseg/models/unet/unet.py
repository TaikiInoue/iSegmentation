from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

from iseg.blocks import build_blocks


class UNet(nn.Module):
    def __init__(self):

        super(UNet, self).__init__()

        cfg_path = Path(__file__).parent / "unet.yaml"
        self.cfg = OmegaConf.load(str(cfg_path))
        self.encoder = build_blocks(self.cfg.encoder.layers)
        self.decoder = build_blocks(self.cfg.decoder.layers)
        self.output_dict = {}

    def forward(self, x):

        self.output_dict["input"] = x

        for block in self.encoder:
            if block._get_name() == "SaveOutput":
                self.output_dict[block.output_name] = x
            else:
                x = block(x)

        for block in self.decoder:
            if block._get_name() == "Concat":
                x = block(x, self.output_dict)
            else:
                x = block(x)

        return x
