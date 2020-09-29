import iseg.types as T
import torch.optim.lr_scheduler


class RunnerScheduler:

    cfg: T.DictConfig
    optimizer: T.Optimizer

    def init_scheduler(self):

        cls = getattr(torch.optim.lr_scheduler, self.cfg.scheduler.name)
        args = self.cfg.scheduler.args

        if args:
            return cls(self.optimizer, **args)
        else:
            return cls(self.optimizer)
