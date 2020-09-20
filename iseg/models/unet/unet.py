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

        self.part_dict = {}
        for i, part_cfg in enumerate(part_cfg_list):
            part_name = part_cfg.name
            block_cfg_list = part_cfg.blocks
            block_attr = getattr(iseg.models.unet.parts, part_cfg.name)
            self.part_dict[f"{part_name}_{i}"] = block_attr(block_cfg_list)

    def forward(self, x):

        x0 = self.part_dict["FirstConv_0"](x)
        x1 = self.part_dict["Down_1"](x0)
        x2 = self.part_dict["Down_2"](x1)
        x3 = self.part_dict["Down_3"](x2)
        y0 = self.part_dict["Down_4"](x3)
        y1 = self.part_dict["Up_5"](y0, x3)
        y2 = self.part_dict["Up_6"](y1, x2)
        y3 = self.part_dict["Up_7"](y2, x1)
        z0 = self.part_dict["Up_8"](y3, x0)
        z1 = self.part_dict["LastConv_9"](z0)
        return z1
