import torch.nn as nn

import iseg.typehint as T


class ConvNormReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        padding_mode: str,
        inplace: bool,
    ):

        super(ConvNormReLU, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=inplace),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.layer(x)
