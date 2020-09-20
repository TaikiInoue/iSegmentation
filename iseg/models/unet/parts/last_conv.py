import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class LastConv(nn.Module):
    def __init__(self, block_cfg_list: T.ListConfig):

        """
        Args:
            block_cfg_list (T.ListConfig):
                - name: ConvNormReLU
        """

        last_conv = []
        for block_cfg in block_cfg_list:
            block_attr = getattr(iseg.blocks, block_cfg.name)
            block = block_attr(**block_cfg.args)
            last_conv.append(block)

        self.last_conv = nn.Sequential(*last_conv)

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.last_conv(x)
