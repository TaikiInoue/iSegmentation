import iseg.typehint as T
from iseg.models import BasePart


class Down(BasePart):

    maxpool: T.Module
    conv_norm_relu_0: T.Module
    conv_norm_relu_1: T.Module

    def __init__(self, part_cfg: T.ListConfig) -> None:

        """
        Args:
            part_cfg (T.ListConfig):
                - maxpool: MaxPool2d
                - conv_norm_relu_0: ConvNormReLU
                - conv_norm_relu_1: ConvNormReLU
        """

        super(Down, self).__init__()
        self.build_part(part_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.maxpool(x)
        x = self.conv_norm_relu_0(x)
        x = self.conv_norm_relu_1(x)
        return x
