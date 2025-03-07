from pathlib import Path
from typing import Optional

import torch
import torchvision.utils as vutils
from torch import Tensor

from caveat.experiment import Experiment, pre_process, unpack


class JointExperiment(Experiment):

    def __init__(self, *args, **kwargs):
        self.label_embed_sizes = kwargs.get("label_embed_sizes", None)
        if self.label_embed_sizes is None:
            raise UserWarning("ConditionalLSTM requires label_embed_sizes")
        if not isinstance(self.label_embed_sizes, list):
            raise UserWarning(
                "ConditionalLSTM requires label_embed_sizes to be a list of label embedding sizes"
            )
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (
            (x, _),
            (y, (y_weights, y_joint)),
            (labels, (label_weights, label_joint)),
        ) = batch

        self.curr_device = x.device

        log_probs, mu, log_var, z = self.forward(
            x, conditionals=labels, target=y
        )
        train_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            targets=(y, labels),
            masks=(y_weights, label_weights),
            kld_weight=self.kld_loss_weight,
            duration_weight=self.duration_loss_weight,
            batch_idx=batch_idx,
        )
        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        (
            (x, _),
            (y, (y_weights, y_joint)),
            (labels, (label_weights, label_joint)),
        ) = batch
        self.curr_device = x.device

        log_probs, mu, log_var, z = self.forward(x, conditionals=labels)
        val_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            targets=(y, labels),
            masks=(y_weights, label_weights),
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

    def on_validation_end(self) -> None:
        if self.gen:
            self.regenerate_val_batch()
            self.sample_sequences()

    def test_step(self, batch, batch_idx):
        if self.test:
            (
                (x, _),
                (y, (y_weights, y_joint)),
                (labels, (label_weights, label_joint)),
            ) = batch
            self.curr_device = x.device

            log_probs, mu, log_var, z = self.forward(x, conditionals=labels)
            test_loss = self.loss_function(
                log_probs_x=log_probs,
                mu=mu,
                log_var=log_var,
                targets=(y, labels),
                masks=(y_weights, label_weights),
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
        probs_x, probs_y, _ = self.infer(
            x, conditionals=conditionals, device=self.curr_device
        )
        probs = probs_x.squeeze()
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
        probs, _ = self.predict(z, conditionals=labels, device=self.curr_device)
        vutils.save_image(
            pre_process(probs.cpu().data),
            Path(
                self.logger.log_dir,
                name,
                f"{self.logger.name}_epoch_{self.current_epoch}.png",
            ),
            normalize=False,
            nrow=1,
            pad_value=1,
        )

    def predict_step(self, batch):
        # YUCK
        if len(batch) == 2:  # generative process
            zs, labels = batch
            pred_x, pred_labels = self.predict(
                zs, conditionals=labels, device=self.curr_device
            )
            return (pred_x, pred_labels, zs)
        # inference process
        (x, _), (_, _), (labels, _) = batch
        pred_x, pred_labels, zs = self.infer(
            x, conditionals=labels, device=self.curr_device
        )
        return x, pred_x, labels, pred_labels, zs
