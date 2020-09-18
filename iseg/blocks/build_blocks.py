import torch.nn as nn

import iseg.blocks
import iseg.typehint as T


def build_blocks(cfg_list: T.ListConfig):

    block_list = []
    for cfg in cfg_list:
        block = getattr(iseg.blocks, cfg.name)
        block_list.append(block(**cfg.args))

    return nn.Sequential(*block_list)
