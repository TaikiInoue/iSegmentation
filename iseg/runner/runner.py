import logging
from importlib import import_module
from typing import Any

from omegaconf.dictconfig import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from iseg import transforms
from iseg.transforms import Compose


class Runner:
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.transforms_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "test"]:
            self.transforms_dict[data_type] = self._init_transforms(data_type)
            self.dataset_dict[data_type] = self._init_dataset(data_type)
            self.dataloader_dict[data_type] = self._init_dataloader(data_type)

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()

    def _init_transforms(self, data_type: str) -> Compose:

        cfg = self.cfg.augs[data_type]
        return transforms.load(cfg.yaml, data_format="yaml")

    def _init_dataloader(self, data_type: str) -> DataLoader:

        cfg = self.cfg.dataloader[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, dataset=self.dataset_dict[data_type])

    def _init_dataset(self, data_type: str) -> Dataset:

        cfg = self.cfg.dataset[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, augs_dict=self.transforms_dict)

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def _init_criterion(self) -> Module:

        cfg = self.cfg.criterion
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, params=self.model.parameters())

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
