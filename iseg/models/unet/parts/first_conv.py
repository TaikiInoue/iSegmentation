import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class FirstConv(nn.Module):
    def __init__(self, block_cfg_list: T.ListConfig):

        """
        Args:
            block_cfg_list (T.ListConfig):
                - name: ConvNormReLU
                - name: ConvNormReLU
        """

        super(FirstConv, self).__init__()
        first_conv = []
        for block_cfg in block_cfg_list:
            block_attr = getattr(iseg.blocks, block_cfg.name)
            block = block_attr(**block_cfg.args)
            first_conv.append(block)

        self.first_conv = nn.Sequential(*first_conv)

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.first_conv(x)
