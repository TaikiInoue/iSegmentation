import torch.nn

import iseg.types as T


class TrainerCriterion:

    cfg: T.DictConfig

    def init_criterion(self) -> T.Loss:

        criterion_attr = getattr(torch.nn, self.cfg.criterion.name)
        args = self.cfg.criterion.args
        if args:
            return criterion_attr(**args)
        else:
            return criterion_attr()
