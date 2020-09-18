import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


class BaseModel(nn.Module):
    def __init__(self) -> None:

        super(BaseModel, self).__init__()
        self.tensor_dict: dict = {}

    def build_blocks(self, cfg_list: T.ListConfig) -> T.Module:

        block_list = []
        for cfg in cfg_list:
            block = getattr(iseg.blocks, cfg.name)
            block_list.append(block(**cfg.args))

        return nn.Sequential(*block_list)

    def run_blocks(self, blocks: T.Module, x: T.Tensor) -> T.Tensor:

        for block in blocks:

            if block._get_name() == "SaveTensor":
                self.tensor_dict[block.tensor_name] = x

            elif block._get_name() == "Concat":
                x = block(x, self.tensor_dict)

            else:
                x = block(x)

        return x
