import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class BasePart(nn.Module):
    def __init__(self) -> None:

        super(BasePart, self).__init__()

    def build_part(self, part_cfg: T.ListConfig) -> None:

        for block_cfg in part_cfg:
            var_name, block_name = block_cfg.popitem()
            _, args = block_cfg.popitem()
            block = getattr(iseg.blocks, block_name)
            setattr(self, var_name, block(**args))
