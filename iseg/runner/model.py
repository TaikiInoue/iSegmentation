import iseg.models
import iseg.types as T


class RunnerModel:

    cfg: T.DictConfig

    def init_model(self) -> T.Module:

        cls = getattr(iseg.models, self.cfg.model.name)
        model = cls(self.cfg.model.yaml)
        return model.to(self.cfg.device)
