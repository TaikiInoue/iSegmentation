from torch.cuda.amp import GradScaler


class RunnerAMP:
    def init_scaler(self):

        return GradScaler()
