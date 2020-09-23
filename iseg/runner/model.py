import iseg.models
import iseg.types as T


class RunnerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        model = getattr(iseg.models, self.cfg.model.name)
        return model(**self.cfg.model.args)
