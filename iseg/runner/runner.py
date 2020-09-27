import logging

import iseg.types as T
from iseg.runner.amp import RunnerAMP
from iseg.runner.augs import RunnerAugs
from iseg.runner.criterion import RunnerCriterion
from iseg.runner.dataloader import RunnerDataLoader
from iseg.runner.dataset import RunnerDataset
from iseg.runner.metrics import RunnerMetrics
from iseg.runner.model import RunnerModel
from iseg.runner.optimizer import RunnerOptimizer
from iseg.runner.train_test import RunnerTrainTest


class Runner(
    RunnerAMP,
    RunnerAugs,
    RunnerCriterion,
    RunnerDataLoader,
    RunnerDataset,
    RunnerMetrics,
    RunnerOptimizer,
    RunnerTrainTest,
    RunnerModel,
):
    def __init__(self, cfg: T.DictConfig):
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "test"]:
            self.augs_dict[data_type] = self.init_augs(data_type)
            self.dataset_dict[data_type] = self.init_dataset(data_type)
            self.dataloader_dict[data_type] = self.init_dataloader(data_type)

        self.model = self.init_model()
        self.model = self.model.to(self.cfg.device)
        self.scaler = self.init_scaler()
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_criterion()
