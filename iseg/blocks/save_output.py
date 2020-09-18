import torch.nn as nn


class SaveOutput(nn.Module):
    def __init__(self, output_name: str):

        super(SaveOutput, self).__init__()
        self.output_name = output_name

    def forward(self, x):
        return x
