import iseg.typehint as T
from iseg.models import BasePart


class LastConv(BasePart):

    conv_norm_relu: T.Module

    def __init__(self, part_cfg: T.ListConfig):

        """
        Args:
            part_cfg (T.ListConfig):
                - conv_norm_relu: ConvNormReLU
        """

        super(LastConv, self).__init__()
        self.build_part(part_cfg)

    def forward(self, x: T.Tensor) -> T.Tensor:

        return self.conv_norm_relu(x)
