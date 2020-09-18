import os
import sys

import hydra

import iseg.typehint as T
from iseg.trainer import Trainer

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")

    trainer = Trainer(cfg)
    trainer.run_train()
    trainer.run_test()


if __name__ == "__main__":
    main()
