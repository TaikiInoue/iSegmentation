import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class Down(nn.Module):
    def __init__(self, block_cfg_list: T.ListConfig) -> None:

        """
        Args:
            block_cfg_list (T.ListConfig):
                - name: MaxPool2d
                - name: ConvNormReLU
                - name: ConvNormReLU
        """

        super(Down, self).__init__()
        down_conv = []
        for block_cfg in block_cfg_list:
            block_attr = getattr(iseg.blocks, block_cfg.name)
            block = block_attr(**block_cfg.args)
            down_conv.append(block)

        self.down_conv = nn.Sequential(*down_conv)

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.down_conv(x)
