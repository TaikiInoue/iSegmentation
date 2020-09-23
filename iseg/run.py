import os
import sys

import hydra

import iseg.types as T
from iseg.runner import Runner

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")

    runner = Runner(cfg)
    runner.run_train()
    runner.run_test()


if __name__ == "__main__":
    main()
