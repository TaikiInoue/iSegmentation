from typing import Any, Dict, List, Tuple, TypeVar


Tensor = TypeVar("torch.tensor")
Loss = TypeVar("torch.nn.modules.loss._Loss")
Optimizer = TypeVar("torch.optim.Optimizer")
DataLoader = TypeVar("torch.utils.data.DataLoader")
Module = TypeVar("torch.nn.Module")
DictConfig = TypeVar("omegaconf.dictconfig.DictConfig")
Compose = TypeVar("stad.albu.Compose")
Dataset = TypeVar("torch.utils.data.Dataset")
DataFrame = TypeVar("pandas.DataFrame")
Path = TypeVar("pathlib.Path")
Logger = TypeVar("logging.Logger")
ListConfig = TypeVar("omegaconf.listconfig.ListConfig")
Array = TypeVar("numpy.ndarray")

__all__ = ["Any", "Dict", "List", "Tuple", "TypeVar"]
