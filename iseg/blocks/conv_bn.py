import torch.nn as nn

import iseg.typehint as T


class ConvBn(nn.Module):
    def __init__(
        self,
        # nn.Conv2d
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # nn.BatchNorm2d
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):

        super(ConvBn, self).__init__()

        self.conv_bn = nn.Sequential(
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
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=eps,
                momentum=momentum,
                affine=True,
                track_running_stats=True,
            ),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.conv_bn(x)
