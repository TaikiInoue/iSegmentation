from omegaconf import OmegaConf

import iseg.models
import iseg.typehint as T


class TrainerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        model_cfg = OmegaConf.load(self.cfg.model.yaml)
        model = getattr(iseg.models, self.cfg.model.name)
        return model(model_cfg)
