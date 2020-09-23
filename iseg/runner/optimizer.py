import torch

import iseg.types as T


class RunnerOptimizer:

    model: T.Module
    cfg: T.DictConfig

    def init_optimizer(self) -> T.Optimizer:

        params = self.model.parameters()
        optimizer_attr = getattr(torch.optim, self.cfg.optimizer.name)
        args = self.cfg.optimizer.args
        if args:
            return optimizer_attr(params, **args)
        else:
            return optimizer_attr(params)
