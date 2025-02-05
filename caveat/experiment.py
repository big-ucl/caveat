from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import Tensor, nn, optim

from caveat.models.utils import ScheduledOptim


class Experiment(pl.LightningModule):

    def __init__(
        self,
        in_shape: tuple,
        encodings: int,
        encoding_weights: Optional[Tensor] = None,
        conditionals_size: Optional[tuple] = None,
        sos: int = 0,
        gen: bool = False,
        test: bool = False,
        LR: float = 0.001,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> None:
        super(Experiment, self).__init__()
        self.gen = gen
        self.test = test
        self.LR = LR
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.curr_device = None
        self.save_hyperparameters()
        """Base VAE.

        Args:
            in_shape (tuple[int, int]): [time_step, activity one-hot encoding].
            encodings (int): Number of activity encodings.
            encoding_weights (tensor): Weights for activity encodings.
            conditionals_size (int, optional): Size of conditionals encoding. Defaults to None.
            sos (int, optional): Start of sequence token. Defaults to 0.
            config: Additional arguments from config.
        """
        # encoding params
        self.in_shape = in_shape
        print(f"Found input shape: {self.in_shape}")
        self.encodings = encodings
        print(f"Found encodings: {self.encodings}")
        self.encoding_weights = encoding_weights
        print(f"Found encoding weights: {self.encoding_weights}")
        self.conditionals_size = conditionals_size
        if self.conditionals_size is not None:
            print(f"Found conditionals size: {self.conditionals_size}")
        self.sos = sos
        print(f"Found start of sequence token: {self.sos}")
        self.teacher_forcing_ratio = kwargs.get("teacher_forcing_ratio", 0.5)
        print(f"Found teacher forcing ratio: {self.teacher_forcing_ratio}")

        # loss function params
        self.kld_loss_weight = kwargs.get("kld_weight", 0.001)
        print(f"Found KLD weight: {self.kld_loss_weight}")

        self.activity_loss_weight = kwargs.get("activity_loss_weight", 1.0)
        print(f"Found activity loss weight: {self.activity_loss_weight}")

        self.duration_loss_weight = kwargs.get("duration_loss_weight", 1.0)
        print(f"Found duration loss weight: {self.duration_loss_weight}")

        self.label_loss_weight = kwargs.get("label_loss_weight", 0.0001)
        print(f"Found labels loss weight: {self.label_loss_weight}")

        self.use_mask = kwargs.get("use_mask", True)
        print(f"Using mask: {self.use_mask}")

        self.use_weighted_loss = kwargs.get("weighted_loss", True)
        print(f"Using weighted loss: {self.use_weighted_loss}")

        # set up weighted loss
        if self.use_weighted_loss and self.encoding_weights is not None:
            print("Using weighted loss function")
            self.NLLL = nn.NLLLoss(weight=self.encoding_weights)
        else:
            self.NLLL = nn.NLLLoss()

        self.base_NLLL = nn.NLLLoss(reduction="none")

        self.loss = nn.NLLLoss(reduction="none")
        self.MSE = nn.MSELoss(reduction="none")

        # set up scheduled loss function weights
        self.scheduled_kld_weight = 1.0
        self.scheduled_act_weight = 1.0
        self.scheduled_dur_weight = 1.0
        self.scheduled_label_weight = 1.0

        self.build(**kwargs)

    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()

    def training_step(self, batch, batch_idx):
        (x, x_weights), (y, y_weights), (labels, _) = batch

        self.curr_device = x.device
        log_probs, mu, log_var, z = self.forward(
            x, conditionals=labels, target=y, input_mask=x_weights
        )
        train_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=y,
            mask=y_weights,
            kld_weight=self.kld_loss_weight,
            duration_weight=self.duration_loss_weight,
            batch_idx=batch_idx,
        )
        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        (x, x_weights), (y, y_weights), (labels, _) = batch
        self.curr_device = x.device

        log_probs, mu, log_var, z = self.forward(
            x, conditionals=labels, input_mask=x_weights
        )
        val_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=y,
            mask=y_weights,
            kld_weight=self.kld_loss_weight,
            duration_weight=self.duration_loss_weight,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("hp_metric", val_loss["loss"])

    def on_validation_end(self) -> None:
        if self.gen:
            self.regenerate_val_batch()
            self.sample_sequences()

    def test_step(self, batch, batch_idx):
        if self.test:
            (x, x_weights), (y, y_weights), conditionals = batch
            self.curr_device = x.device

            log_probs_x, mu, log_var, z = self.forward(
                x, conditionals=conditionals, input_mask=x_weights
            )
            test_loss = self.loss_function(
                log_probs_x=log_probs_x,
                mu=mu,
                log_var=log_var,
                target=y,
                mask=y_weights,
                kld_weight=self.kld_loss_weight,
                duration_weight=self.duration_loss_weight,
                batch_idx=batch_idx,
            )

            self.log_dict(
                {f"test_{key}": val.item() for key, val in test_loss.items()},
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def regenerate_val_batch(self):
        (x, _), (y, _), (labels, _) = next(
            iter(self.trainer.datamodule.val_dataloader())
        )
        x = x.to(self.curr_device)
        y = y.to(self.curr_device)
        labels = labels.to(self.curr_device)
        self.regenerate_batch(
            x, target=y, name="reconstructions", conditionals=labels
        )

    def regenerate_batch(
        self,
        x: Tensor,
        target: Tensor,
        name: str,
        conditionals: Optional[Tensor] = None,
    ):
        probs, _ = self.infer(
            x, conditionals=conditionals, device=self.curr_device
        )
        probs = probs.squeeze()
        image = unpack(target, probs, self.curr_device)
        div = torch.ones_like(probs)
        images = torch.cat((image.squeeze(), div, probs), dim=-1)
        vutils.save_image(
            pre_process(images.data),
            Path(
                self.logger.log_dir,
                name,
                f"recons_{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=False,
            nrow=1,
            pad_value=1,
        )

    def sample_sequences(self, name: str = "samples") -> None:
        _, _, (labels, _) = next(
            iter(self.trainer.datamodule.test_dataloader())
        )
        labels = labels.to(self.curr_device)
        z = torch.randn(len(labels), self.latent_dim)
        y_probs = self.predict(z, conditionals=labels, device=self.curr_device)
        vutils.save_image(
            pre_process(y_probs.cpu().data),
            Path(
                self.logger.log_dir,
                name,
                f"{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=False,
            nrow=1,
            pad_value=1,
        )

    def get_scheduler(self, optimizer):
        if self.kwargs.get("scheduler_gamma") is not None:
            print("Using ExponentialLR scheduler")
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.kwargs["scheduler_gamma"]
            )

        elif self.kwargs.get("warmup") is not None:
            lr_mul = self.kwargs.get("lr_mul", 2)
            d_model = self.kwargs.get("d_model", 512)
            n_warmup_steps = self.kwargs.get("warmup")
            print("Using ScheduledOptim scheduler")
            scheduler = {
                "scheduler": ScheduledOptim(
                    optimizer,
                    lr_mul=lr_mul,
                    d_model=d_model,
                    n_warmup_steps=n_warmup_steps,
                ),
                "monitor": "val_loss",
                "interval": "step",
            }

        else:
            raise UserWarning("No scheduler specified")

        return scheduler

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.LR, weight_decay=self.weight_decay
        )

        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def predict_step(self, batch):
        # YUCK
        if len(batch) == 2:  # generative process
            zs, labels = batch
            return (
                labels,
                self.predict(zs, conditionals=labels, device=self.curr_device),
                zs,
            )
        # inference process
        (x, _), (_, _), (labels, _) = batch
        preds, zs = self.infer(x, conditionals=labels, device=self.curr_device)
        return x, preds, zs, labels


def unpack(x, y, current_device):
    if x.dim() == 2:
        # assume cat encoding and unpack into image
        channels = y.shape[-1]
        eye = torch.eye(channels)
        eye = eye.to(current_device)
        ximage = eye[x.long()].squeeze()
        return ximage

    elif x.shape[-1] == 2:
        # assume cat encoding and unpack into image
        channels = y.shape[-1] - 1
        acts, durations = x.split([1, 1], dim=-1)
        eye = torch.eye(channels)
        eye = eye.to(current_device)
        ximage = eye[acts.long()].squeeze()
        ximage = torch.cat((ximage, durations), dim=-1)
        return ximage
    elif 1 in x.shape:
        print(f"Unknown encoding; {x.shape}, squeezing")
        return unpack(x.squeeze(), y.squeeze(), current_device)
    else:
        print(f"Unknown encoding; {x.shape}, returning input x")
    return x


def pre_process(images):
    # hack for dealing with outputs encoded as [N, h, w]
    # need to add channel dim and rearrange to [N, C, h, w]
    # todo remove C/3d encoder
    s = images.shape
    if len(s) == 3:
        # need to add dim and move channel to front
        return images.reshape(s[0], s[1], s[2], 1).permute(0, 3, 1, 2)
    return images
