from pathlib import Path

import iseg.types as T
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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

        with open(self.base / f"images/{stem}.npy", "rb") as f:
            img = np.load(f)

        with open(self.base / f"masks/{stem}.npy", "rb") as f:
            mask = np.load(f)

        data_dict = self.augs(image=img, mask=mask)
        data_dict["mask"] = data_dict["mask"]
        data_dict["stem"] = stem
        return data_dict

    def __len__(self) -> int:

        return len(self.stem_list)
