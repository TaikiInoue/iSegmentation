import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self) -> None:

        super(BaseModel, self).__init__()
