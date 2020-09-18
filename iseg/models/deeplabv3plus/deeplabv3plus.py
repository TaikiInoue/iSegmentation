import iseg.typehint as T
from iseg.models import BaseModel


class DeepLabV3Plus(BaseModel):
    def __init__(self):

        self.backbone = self.build_blocks()
        self.aspp = self.build_blocks()
        self.decoder = self.build_blocks()

    def forward(self, x: T.Tensor) -> T.Tensor:

        x = self.run_blocks(self.backbone, x)
        x = self.run_blocks(self.aspp, x)
        x = self.run_blocks(self.decoder, x)
        return x
