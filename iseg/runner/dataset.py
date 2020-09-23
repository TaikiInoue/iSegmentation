import iseg.datasets
import iseg.types as T


class RunnerDataset:

    cfg: T.DictConfig
    augs_dict: T.Dict[str, T.Compose]

    def init_dataset(self, data_type: str) -> T.Dataset:

        dataset_attr = getattr(iseg.datasets, self.cfg.dataset.name)
        dataset = dataset_attr(self.cfg, self.augs_dict, data_type)
        return dataset
