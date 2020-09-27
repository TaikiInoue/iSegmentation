import iseg.types as T
import torch
import torch.nn as nn
from iseg.models import Builder


class Decoder(nn.Module, Builder):

    conv_bn_relu: T.Module
    interpolate: T.Module
    aspp_conv: T.Module
    dropout: T.Module
    conv: T.Module

    def __init__(self, object_cfg: T.ListConfig) -> None:

        """
        Args:
            object_cfg (T.ListConfig):
                - conv_bn_relu: iseg.blocks - ConvBnReLU
                - interpolate: iseg.blocks - Interpolate
                - aspp_conv: iseg.models.deeplabv3plus.blocks - ASPPConv
                - dropout: torch.nn - Dropout2d
                - conv: torch.nn - Conv2d
        """

        super(Decoder, self).__init__()
        self.build_blocks(object_cfg)

    def forward(self, x: T.Tensor, low_feature: T.Tensor) -> T.Tensor:

        low_feature = self.conv_bn_relu(low_feature)

        # Update self.interpolate.size from None to (h, w)
        _, _, h, w = low_feature.shape
        self.interpolate.size = (h, w)

        x = self.interpolate(x)
        out = torch.cat([x, low_feature], dim=1)
        out = self.aspp_conv(out)
        out = self.dropout(out)
        out = self.conv(out)
        return out
