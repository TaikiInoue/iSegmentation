import torch
from tqdm import tqdm

import iseg.typehint as T


class TrainerRunTrainValTest:

    model: T.Module
    cfg: T.DictConfig
    log: T.Logger
    dataloader_dict: T.Dict[str, T.DataLoader]
    criterion: T.Loss
    optimizer: T.Optimizer
    augs_dict: T.Dict[str, T.Compose]

    def run_train(self):

        self.model.train()

        pbar = tqdm(range(1, self.cfg.run.train.epochs + 1), desc="train")
        for epoch in pbar:

            self.log.info(f"epoch - {epoch}")
            loss_sum = 0
            for data_dict in self.dataloader_dict["train"]:
                img = data_dict["image"].to(self.cfg.device)
                msk = data_dict["mask"].long().to(self.cfg.device)
                semseg_map = self.model(img)
                loss = self.criterion(semseg_map, msk)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss = loss_sum / len(self.dataloader_dict["train"])
            self.log.info(f"loss - {epoch_loss}")

            if epoch % (self.cfg.run.train.epochs // 10) == 0:
                self.run_val()
                self.model.train()

        torch.save(self.school.state_dict(), "pretrained.pth")

    def run_val(self):

        self.model.eval()
        pbar = tqdm(self.dataloader_dict["val"], desc="val")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            msk = data_dict["mask"].long().to(self.cfg.device)
            semseg_map = self.model(img)
            loss = self.criterion(semseg_map, msk)

    def run_test(self):

        self.model.eval()
        pbar = tqdm(self.dataloader_dict["test"], desc="test")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            msk = data_dict["mask"].long().to(self.cfg.device)
            semseg_map = self.model(img)
