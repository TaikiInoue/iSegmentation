import torch
import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class Up(nn.Module):
    def __init__(self, block_cfg_list: T.ListConfig) -> None:

        """
        Args:
            block_cfg_list (T.ListConfig):
                - name: Upsample (or ConvTranspose2d)
                - name: ConvNormReLU
                - name: ConvNormReLU
        """

        conv = []
        for block_cfg in block_cfg_list:
            block_attr = getattr(iseg.blocks, block_cfg.name)
            block = block_attr(**block_cfg.args)

            if block_cfg.name in ["Upsample", "ConvTranspose2d"]:
                self.up = block
            else:
                conv.append(block)

        self.conv = nn.Sequential(*conv)

    def forward(self, x0: T.Tensor, x1: T.Tensor) -> T.Tensor:

        x0 = self.up(x0)
        x2 = torch.cat([x1, x0], dim=1)
        return self.conv(x2)
