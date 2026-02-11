from pytorch_lightning.callbacks import Callback

class LossHistory(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"].detach().cpu().item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "loss" in outputs:
            loss = outputs["loss"].detach().cpu().item()
            self.val_losses.append(loss)
