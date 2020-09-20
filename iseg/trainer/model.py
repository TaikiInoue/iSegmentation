from omegaconf import OmegaConf

import iseg.models
import iseg.typehint as T


class TrainerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        part_cfg_list = OmegaConf.load(self.cfg.model.yaml)
        model_attr = getattr(iseg.models, self.cfg.model.name)
        return model_attr(part_cfg_list)
