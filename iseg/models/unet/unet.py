import torch.nn as nn

import iseg.models.unet.parts
import iseg.typehint as T


class UNet(nn.Module):
    def __init__(self, part_cfg_list: T.ListConfig):

        """
        Args:
            part_cfg_list (T.ListConfig):
                - name: FirstConv
                - name: Down
                - name: Down
                - name: Down
                - name: Down
                - name: Up
                - name: Up
                - name: Up
                - name: Up
                - name: LastConv
        """

        super(UNet, self).__init__()

        self.first_conv = self.build_block(part_cfg_list[0])
        self.down_0 = self.build_block(part_cfg_list[1])
        self.down_1 = self.build_block(part_cfg_list[2])
        self.down_2 = self.build_block(part_cfg_list[3])
        self.down_3 = self.build_block(part_cfg_list[4])
        self.up_0 = self.build_block(part_cfg_list[5])
        self.up_1 = self.build_block(part_cfg_list[6])
        self.up_2 = self.build_block(part_cfg_list[7])
        self.up_3 = self.build_block(part_cfg_list[8])
        self.last_conv = self.build_block(part_cfg_list[9])

    def build_block(self, part_cfg: T.DictConfig) -> nn.Module:

        part_attr = getattr(iseg.models.unet.parts, part_cfg.name)
        return part_attr(part_cfg.blocks)

    def forward(self, x):

        x0 = self.first_conv(x)
        x1 = self.down_0(x0)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        y0 = self.down_3(x3)
        y1 = self.up_0(y0, x3)
        y2 = self.up_1(y1, x2)
        y3 = self.up_2(y2, x1)
        z0 = self.up_3(y3, x0)
        z1 = self.last_conv(z0)
        return z1
