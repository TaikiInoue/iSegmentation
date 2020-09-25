import iseg.models
import iseg.types as T
from omegaconf import OmegaConf


class RunnerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        model_cfg = OmegaConf.load(self.cfg.model.yaml)
        cls = getattr(iseg.models, self.cfg.model.name)
        return cls(model_cfg)
