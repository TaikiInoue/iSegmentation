import iseg.types as T
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: float = None,
        mode: str = "nearest",
        align_corners: bool = None,
        recompute_scale_factor: bool = None,
    ):

        super(Interpolate, self).__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
        return x
