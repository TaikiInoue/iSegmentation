import torch
import torch.nn as nn

import iseg.typehint as T


class Concat(nn.Module):
    def __init__(self, dim: int, output_name_list: T.ListConfig):

        super(Concat, self).__init__()
        self.dim = dim
        self.output_name_list = output_name_list

    def forward(self, x: T.Tensor, output_dict: T.Dict[str, T.Tensor]) -> T.Tensor:

        x_list = [x]
        for output_name in self.output_name_list:
            x_list.append(output_dict[output_name])

        return torch.cat(x_list, dim=self.dim)
