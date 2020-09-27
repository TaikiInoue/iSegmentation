import iseg.types as T
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


class RunnerTrainTest:

    model: T.Module
    cfg: T.DictConfig
    log: T.Logger
    dataloader_dict: T.Dict[str, T.DataLoader]
    criterion: T.Loss
    optimizer: T.Optimizer
    scaler: T.Scaler
    augs_dict: T.Dict[str, T.Compose]

    def run_train(self):

        self.model.train()
        pbar = tqdm(range(1, self.cfg.run.train.epochs + 1), desc="train")
        for epoch in pbar:

            self.log.info(f"epoch - {epoch}")
            cumulative_loss = 0
            for data_dict in self.dataloader_dict["train"]:

                self.optimizer.zero_grad()
                img = data_dict["image"].to(self.cfg.device)
                mask = data_dict["mask"].long().to(self.cfg.device)

                with autocast():
                    segmap = self.model(img)
                    loss = self.criterion(segmap, mask)
                    cumulative_loss += loss.item()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            epoch_loss = cumulative_loss / len(self.dataloader_dict["train"])
            self.log.info(f"loss - {epoch_loss}")

        torch.save(self.model.state_dict(), f"{self.cfg.model.name}.pth")

    def run_test(self):

        self.model.eval()
        mask_list = []
        segmap_list = []
        pbar = tqdm(self.dataloader_dict["test"], desc="test")
        for i, data_dict in enumerate(pbar):

            img = data_dict["image"].to(self.cfg.device)
            mask = data_dict["mask"].long().to(self.cfg.device)
            segmap = self.model(img)

            # convert (b, num_classes, h, w) to (b, h, w)
            segmap = segmap.argmax(dim=1)

            mask_list.extend(mask.cpu().detach().numpy())
            segmap_list.extend(segmap.cpu().detach().numpy())

        self.compute_and_log_metrics("test", segmap_list, mask_list)
