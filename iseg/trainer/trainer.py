import logging

import iseg.types as T
from iseg.trainer.augs import TrainerAugs
from iseg.trainer.criterion import TrainerCriterion
from iseg.trainer.dataloader import TrainerDataLoader
from iseg.trainer.dataset import TrainerDataset
from iseg.trainer.model import TrainerModel
from iseg.trainer.optimizer import TrainerOptimizer
from iseg.trainer.run_train_val_test import TrainerRunTrainValTest


class Trainer(
    TrainerAugs,
    TrainerCriterion,
    TrainerDataLoader,
    TrainerDataset,
    TrainerOptimizer,
    TrainerRunTrainValTest,
    TrainerModel,
):
    def __init__(self, cfg: T.DictConfig):
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "val", "test"]:
            self.augs_dict[data_type] = self.init_augs(data_type)
            self.dataset_dict[data_type] = self.init_dataset(data_type)
            self.dataloader_dict[data_type] = self.init_dataloader(data_type)

        self.model = self.init_model()
        self.model = self.model.to(self.cfg.device)
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_criterion()
