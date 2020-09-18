import iseg.models
import iseg.typehint as T


class TrainerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        model_attr = getattr(iseg.models, self.cfg.model.name)
        if self.cfg.model.args:
            return model_attr(**self.cfg.model.args)
        else:
            return model_attr()
