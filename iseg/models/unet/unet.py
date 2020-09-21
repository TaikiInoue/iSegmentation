import torch.nn as nn

import iseg.types as T
from iseg.models import Builder


class UNet(nn.Module, Builder):
    def __init__(self, object_cfg: T.ListConfig):

        """
        Args:
            object_cfg (T.ListConfig):
                - first_conv: iseg.models.unet.blocks - FirstConv
                - down_0: iseg.models.unet.blocks - Down
                - down_1: iseg.models.unet.blocks - Down
                - down_2: iseg.models.unet.blocks - Down
                - down_3: iseg.models.unet.blocks - Down
                - up_0: iseg.models.unet.blocks - Up
                - up_1: iseg.models.unet.blocks - Up
                - up_2: iseg.models.unet.blocks - Up
                - up_3: iseg.models.unet.blocks - Up
                - last_conv: iseg.models.unet.blocks - LastConv
        """

        super(UNet, self).__init__()
        self.build_blocks(object_cfg)

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
