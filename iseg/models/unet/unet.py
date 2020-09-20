import torch.nn as nn

import iseg.models.unet.parts
import iseg.typehint as T


class UNet(nn.Module):
    def __init__(self, model_cfg: T.ListConfig):

        """
        Args:
            model_cfg (T.ListConfig):
                - first_conv: FirstConv
                - down_0: Down
                - down_1: Down
                - down_2: Down
                - down_3: Down
                - up_0: Up
                - up_1: Up
                - up_2: Up
                - up_3: Up
                - last_conv: LastConv
        """

        super(UNet, self).__init__()

        for part_cfg in model_cfg:
            var_name, part_name = part_cfg.popitem()
            _, part_cfg = part_cfg.popitem()
            part = getattr(iseg.models.unet.parts, part_name)
            setattr(self, var_name, part(part_cfg))

    def forward(self, x):

        x_0 = self.first_conv(x)
        x_1 = self.down_0(x_0)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        y_0 = self.down_3(x_3)
        y_1 = self.up_0(y_0, x_3)
        y_2 = self.up_1(y_1, x_2)
        y_3 = self.up_2(y_2, x_1)
        z_0 = self.up_3(y_3, x_0)
        z_1 = self.last_conv(z_0)
        return z_1
