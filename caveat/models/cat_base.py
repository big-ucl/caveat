from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, exp, nn
from torch.distributions import OneHotCategorical
from torch.nn import functional as F

from caveat.cat_experiment import CatExperiment
from caveat.models import utils


class BaseEncoder(LightningModule):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class BaseDecoder(LightningModule):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class CatBase(CatExperiment):
    def build(self, **config):
        self.latent_dim = config["latent_dim", 2]
        self.latent_cats = config.get("latent_categories", 2)
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config.get("dropout", 0)
        length, _ = self.in_shape

        self.encoder = BaseEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.decoder = BaseDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_n,
            max_length=length,
            sos=self.sos,
        )
        self.unflattened_shape = (self.hidden_n, self.hidden_size)
        hidden_out_size = self.hidden_n * self.hidden_size
        self.latent_shape = (self.latent_dim, self.latent_cats)
        self.latent_size = self.latent_dim * self.latent_cats

        self.encoder_resize = nn.Linear(hidden_out_size, self.latent_size)
        self.latent_activation = nn.Softmax(dim=-1)
        self.decoder_resize = nn.Linear(self.latent_size, hidden_out_size)

    def forward(
        self, x: Tensor, labels: Optional[Tensor] = None, target=None, **kwargs
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        p = self.encode(x, labels)
        z = self.sample_latent_gumbel(p)
        log_probs_x = self.decode(z, labels=labels, target=target)
        return [log_probs_x, p.mean(-1).mean(-1), p.mean(-1).var(-1), (p, z)]

    def loss_function(
        self,
        log_probs: Tensor,
        mu: Tensor,
        log_var: Tensor,
        target: Tensor,
        weights: Tuple[Tensor, Tensor],
        cat_p: Tensor,
        cat_z: Tensor,
        **kwargs,
    ) -> dict:
        """Computes the loss function. Different models are expected to need different loss functions
        depending on the data structure. Typically it will either be a sequence encoding [N, L, 2],
        or discretized encoding [N, L, C] or [N, L].

        The default is to use the sequence loss function. But child classes can override this method.

        Returns losses as a dictionary. Which must include the keys "loss" and "recon_loss".

        Args:
            log_probs (Tensor): Log probabilities of the output.
            mu (Tensor): Latent layer means.
            log_var (Tensor): Latent layer log variances.
            target (Tensor): Target sequences.
            weights (Tensor, Tensor): activity and joint weights.

        Returns:
            dict: Losses.
        """

        return self.continuous_loss(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=target,
            weights=weights,
            cat_p=cat_p,
            cat_z=cat_z,
            **kwargs,
        )

    def sample_latent(self, probs) -> Tensor:
        m = OneHotCategorical(probs=probs)
        return m.sample()

    def sample_latent_gumbel(
        self, logits: Tensor, temperature: float = 0.5
    ) -> Tensor:
        """
        Sample from categorical distribution using Gumbel-Softmax.
        Returns a soft one-hot vector of shape [B, K].
        """
        return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)

    def entropy(self, probs: Tensor) -> Tensor:
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        return entropy

    def encode(self, input: Tensor, labels: Optional[Tensor]) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # [N, L, C]
        hidden = self.encoder(input)
        logits = self.encoder_resize(hidden).view(
            -1, self.latent_dim, self.latent_cats
        )

        return self.latent_activation(logits)

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output batch as tuple of log probs and probs ([N, L, C]).
        """
        z = z.view(-1, self.latent_size)
        hidden = self.decoder_resize(z)
        hidden = hidden.unflatten(1, self.unflattened_shape).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])
        batch_size = z.shape[0]

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # attempt to use teacher forcing by passing target
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        return log_probs

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        # sample latent_dim tokens from range latent_cats with uniform distribution
        z = torch.randint(
            0, self.latent_cats, (z.shape[0], self.latent_dim), device=device
        )
        z = F.one_hot(z, num_classes=self.latent_cats).float()
        prob_samples = exp(self.decode(z, **kwargs))
        return prob_samples

    def infer(self, x: Tensor, device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output and z samples.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            (tensor: [N, steps, acts], tensor: [N, latent_dims]).
        """
        log_probs_x, _, _, (_, z) = self.forward(x, **kwargs)
        prob_samples = exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        cats = z.argmax(dim=-1)  # [N, latent_dims]
        cats = cats.to(device)
        return prob_samples, cats

    def act_seq_loss(
        self, preds, targets, weights, seq_weights, joint_weights
    ) -> Tensor:
        """Loss function for activity encoding [B, L]."""
        B, L, _ = targets.shape
        losses = self.base_NLLL(
            preds.view(-1, self.encodings), targets.view(-1).long()
        )
        losses = losses * weights.view(-1)
        losses = losses.view(B, L) * seq_weights
        if joint_weights is not None:
            losses = losses * joint_weights
        return losses.mean()

    def dur_mse_loss(
        self, preds, targets, weights, seq_weights, joint_weights
    ) -> Tensor:
        """MSE loss function for durations [B, L]."""
        losses = self.MSE(preds, targets)
        losses = losses * weights * seq_weights
        if joint_weights is not None:
            losses = losses * joint_weights
        return losses.mean()

    def continuous_loss(
        self,
        log_probs,
        mu,
        log_var,
        target,
        weights: Tuple[Tensor, Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]] = (None, None),
        cat_p: Tensor = None,
        cat_z: Tensor = None,
        **kwargs,
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durs = self.unpack_encoding(target)
        pred_acts, pred_durs = self.unpack_encoding(log_probs)
        pred_durs = torch.exp(pred_durs)

        act_weights, seq_weights = weights
        _, joint_weights = label_weights
        dur_weights = utils.duration_mask(act_weights)
        # dur_weights = seq_weights  # use seq_weights as dur_weights

        # activity loss
        act_weight = self.activity_loss_weight * self.scheduled_act_weight
        act_recon = self.act_seq_loss(
            preds=pred_acts,
            targets=target_acts,
            weights=act_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_act_recon = act_weight * act_recon

        # duration loss
        dur_weight = self.duration_loss_weight * self.scheduled_dur_weight
        dur_recon = self.dur_mse_loss(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_dur_recon = dur_weight * dur_recon

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # kld loss
        entropy = self.entropy(cat_p)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_entropy_loss = scheduled_kld_weight * entropy

        # final loss
        loss = w_recons_loss + w_entropy_loss

        return {
            "loss": loss,
            "entropy": w_entropy_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "entropy_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_weight]).float(),
            "dur_weight": torch.tensor([dur_weight]).float(),
        }

    def unpack_encoding(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Split the input into activity and duration.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            tuple[tensor, tensor]: [activity [N, steps, acts], duration [N, steps, 1]].
        """
        acts = input[:, :, :-1].contiguous()
        durations = input[:, :, -1:].squeeze(-1).contiguous()
        return acts, durations

    def pack_encoding(self, acts: Tensor, durations: Tensor) -> Tensor:
        """Pack the activity and duration into input.

        Args:
            acts (tensor): Activity [N, steps, acts].
            durations (tensor): Duration [N, steps, 1].

        Returns:
            tensor: Input sequences [N, steps, acts].
        """
        if len(durations.shape) == 2:
            durations = durations.unsqueeze(-1)
        return torch.cat((acts, durations), dim=-1)
