import os
from abc import abstractmethod
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class LitModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            save_pth: int,
            save_pth_path: str,
            save_pth_name: str,
            cluster_interval: int = 1,
            log_tmp_output_every_step: int = None,
            log_tmp_output_every_epoch: int = None,
            example_input: torch.Tensor = None
    ):
        super().__init__()
        #  save params
        self.model = model.to(self.device)
        self.cluster_interval = cluster_interval
        self.log_tmp_output_every_step = log_tmp_output_every_step
        self.log_tmp_output_every_epoch = log_tmp_output_every_epoch
        self.example_input = example_input
        self.save_pth = save_pth
        self.save_pth_path = save_pth_path
        self.save_pth_name = save_pth_name

        #  store outputs intercepted by hooks
        self.intercept_output: dict[str, torch.Tensor] = {}
        self.grid_images: dict[str, np.ndarray] = {}

        #  lock, to avoid hooking recursively
        self.log_lock = False

        #  register hooks for Conv2d layers
        for name, layer in model.named_modules():
            flag, formatted_name = self.conv_2d_filter(name, layer)
            if flag:
                layer.register_forward_hook(self.hook_feature_map(formatted_name, layer))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.log_lock = True
        outputs = self.model(inputs)
        self.log_lock = False
        log_dict = self.training_step_loss_fn(inputs, outputs, labels)

        train_loss = log_dict["train_loss"]

        self.logger.experiment.log(log_dict)

        #  intermittently log feature maps
        if self.log_tmp_output_every_step and self.global_step % self.log_tmp_output_every_step == 0:
            self.log_tmp_output()

        return train_loss

    def on_train_epoch_end(self) -> None:
        #  calculate accuracy every single epoch
        correct_sum = 0
        samples_sum = 0
        with torch.no_grad():
            for inputs, labels in self.train_dataloader():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct = (pred == labels).sum()
                samples = labels.size(0)
                correct_sum += correct.item()
                samples_sum += samples
        self.logger.experiment.log({"accuracy": correct_sum / samples_sum})
        #  intermittently log feature maps
        if self.log_tmp_output_every_epoch and self.current_epoch % self.log_tmp_output_every_epoch == 0:
            self.log_tmp_output()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]

    @abstractmethod
    def conv_2d_filter(self, name: str, layer: nn.Module) -> tuple[bool, str]:
        """feature map hook interface"""
        pass

    @abstractmethod
    def hook_feature_map(self, name: str, layer: nn.Module) -> tuple[bool, str]:
        pass

    @abstractmethod
    def training_step_loss_fn(
            self, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        loss function interface, key `train_loss` required
        :param inputs: (B, C, H, W)
        :param outputs: (B, classes)
        :param labels: (B)
        """
        pass

    @abstractmethod
    def log_tmp_output(self):
        pass

    def add_intercept_output(self, from_key: str, to_key: str):
        """
        get outputs by the `from_key` from NN then add it to the `to_key` in `intercept_output`
        """
        layer = dict([*self.model.named_modules()])[from_key]

        def hook(module, input, output):
            self.intercept_output[to_key] = output

        layer.register_forward_hook(hook)

    def forward(self, x):
        return self.model(x)

    def on_train_end(self) -> None:
        if self.save_pth:
            name = self.save_pth_name if self.save_pth_name else f"model_{self.current_epoch}.pth"
            torch.save(self.model.state_dict(), os.path.join(self.save_pth_path, name))
