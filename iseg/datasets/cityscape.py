from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset

import iseg.typehint as T


class CityscapeDataset(Dataset):
    def __init__(self, cfg: T.DictConfig, augs_dict: T.Dict[str, T.Compose], data_type: str):

        """
        Args:
            cfg (T.DictConfig): configurations in iseg/yamls/cityscape.yaml
            augs_dict (T.Dict[str, T.Compose]): augmentations dict for the preprocessing
            data_type (str): train, val or test
        """

        self.base = Path(cfg.dataset.base)
        self.augs = augs_dict[data_type]
        self.stem_list = []

        df = pd.read_csv(self.base / "info.csv")
        for query in cfg.dataset[data_type].query:
            stem = df.query(query)["stem"]
            self.stem_list += stem.to_list()

    def __getitem__(self, idx: int) -> dict:

        """
        Args:
            idx (int): Index of data

        Returns:
            data_dict["image"] (T.Tensor): The shape is (b, c, h, w)
            data_dict["mask"] (T.Tensor): The shape is (b, h, w)
            data_dict["stem"] (str): filename without its suffix
        """

        stem = self.stem_list[idx]
        img = cv2.imread(str(self.base / f"images/{stem}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(self.base / f"masks/{stem}.png"), cv2.IMREAD_GRAYSCALE)

        data_dict = self.augs(image=img, mask=msk)
        data_dict["mask"] = data_dict["mask"]
        data_dict["stem"] = stem
        return data_dict

    def __len__(self) -> int:

        return len(self.stem_list)
