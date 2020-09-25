import iseg.types as T
import torch
from tqdm import tqdm


class RunnerTrainValTest:

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
            cumulative_loss = 0
            for data_dict in self.dataloader_dict["train"]:
                img = data_dict["image"].to(self.cfg.device)
                mask = data_dict["mask"].long().to(self.cfg.device)
                segmap = self.model(img)
                loss = self.criterion(segmap, mask)
                loss.backward()
                cumulative_loss += loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss = cumulative_loss / len(self.dataloader_dict["train"])
            self.log.info(f"loss - {epoch_loss}")

            if epoch % (self.cfg.run.train.epochs // 10) == 0:
                self.run_val()
                self.model.train()

        torch.save(self.model.state_dict(), f"{self.cfg.model.name}.pth")

    def run_val(self):

        self.model.eval()
        with torch.no_grad():
            cumulative_loss = 0
            mask_list = []
            segmap_list = []
            pbar = tqdm(self.dataloader_dict["val"], desc="val")
            for i, data_dict in enumerate(pbar):

                img = data_dict["image"].to(self.cfg.device)
                mask = data_dict["mask"].long().to(self.cfg.device)
                segmap = self.model(img)
                cumulative_loss += self.criterion(segmap, mask)

                mask_list.extend(mask.cpu().detach().numpy())
                segmap_list.extend(segmap.cpu().detach().numpy())

        self.log.info(f"val loss - {cumulative_loss}")
        self.compute_and_log_metrics("val", segmap_list, mask_list)

    def run_test(self):

        self.model.eval()
        mask_list = []
        segmap_list = []
        pbar = tqdm(self.dataloader_dict["test"], desc="test")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            mask = data_dict["mask"].long().to(self.cfg.device)
            segmap = self.model(img)

            mask_list.extend(mask.cpu().detach().numpy())
            segmap_list.extend(segmap.cpu().detach().numpy())

        self.compute_and_log_metrics("test", segmap_list, mask_list)
