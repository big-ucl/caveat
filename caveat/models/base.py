from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, exp, nn

from caveat.experiment import Experiment
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


class Base(Experiment):

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
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
        flat_size_encode = self.hidden_n * self.hidden_size
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

    def forward(
        self, x: Tensor, labels: Optional[Tensor] = None, target=None, **kwargs
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(z, labels=labels, target=target)
        return [log_probs_x, mu, log_var, z]

    def loss_function(
        self,
        log_probs: Tensor,
        mu: Tensor,
        log_var: Tensor,
        target: Tensor,
        weights: Tuple[Tensor, Tensor],
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

        return self.seq_loss(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=target,
            weights=weights,
            **kwargs,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Re-parameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (tensor): Mean of the latent Gaussian [N x latent_dims].
            logvar (tensor): Standard deviation of the latent Gaussian [N x latent_dims].

        Returns:
            tensor: [N x latent_dims].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def kld(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # from https://kvfrans.com/deriving-the-kl/
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

    def encode(self, input: Tensor, labels: Optional[Tensor]) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # [N, L, C]
        hidden = self.encoder(input)
        # [N, flatsize]

        # Split the result into mu and var components
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output batch as tuple of log probs and probs ([N, L, C]).
        """
        hidden = self.fc_hidden(z)
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
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        prob_samples = exp(self.decode(z, **kwargs))
        return prob_samples

    def infer(self, x: Tensor, device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output and z samples.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            (tensor: [N, steps, acts], tensor: [N, latent_dims]).
        """
        log_probs_x, _, _, z = self.forward(x, **kwargs)
        prob_samples = exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        z = z.to(device)
        return prob_samples, z

    # def unweighted_seq_loss(
    #     self, log_probs, mu, log_var, target, mask, **kwargs
    # ) -> dict:
    #     """Loss function for sequence encoding [N, L, 2]."""

    #     # unpack act probs and durations
    #     target_acts, target_durations = self.unpack_encoding(target)
    #     pred_acts, pred_durations = self.unpack_encoding(log_probs)
    #     pred_durations = torch.exp(pred_durations)

    #     if self.use_mask:  # default is to use masking
    #         flat_mask = mask.view(-1).bool()
    #     else:
    #         flat_mask = torch.ones_like(target_acts).view(-1).bool()

    #     # activity loss
    #     act_recon = self.NLLL(
    #         pred_acts.view(-1, self.encodings)[flat_mask],
    #         target_acts.view(-1).long()[flat_mask],
    #     )
    #     act_scheduled_weight = (
    #         self.activity_loss_weight * self.scheduled_act_weight
    #     )
    #     w_act_recon = act_scheduled_weight * act_recon

    #     # duration loss
    #     dur_recon = self.duration_loss_weight * self.MSE(
    #         pred_durations.view(-1)[flat_mask],
    #         target_durations.view(-1)[flat_mask],
    #     )
    #     dur_scheduled_weight = (
    #         self.duration_loss_weight * self.scheduled_dur_weight
    #     )
    #     w_dur_recon = dur_scheduled_weight * dur_recon

    #     # reconstruction loss
    #     w_recons_loss = act_recon + dur_recon

    #     # # hamming distance
    #     # recon_argmax = torch.argmax(pred_acts, dim=-1)
    #     # recon_act_ham = self.hamming(recon_argmax, target_acts.squeeze().long())

    #     # kld loss
    #     kld_loss = self.kld(mu, log_var)
    #     scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
    #     w_kld_loss = scheduled_kld_weight * kld_loss

    #     # final loss
    #     loss = w_recons_loss + w_kld_loss

    #     return {
    #         "loss": loss,
    #         "KLD": w_kld_loss.detach(),
    #         "recon_loss": w_recons_loss.detach(),
    #         "act_recon": w_act_recon.detach(),
    #         "dur_recon": w_dur_recon.detach(),
    #         "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
    #         "act_weight": torch.tensor([act_scheduled_weight]).float(),
    #         "dur_weight": torch.tensor([dur_scheduled_weight]).float(),
    #     }

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

    def dur_seq_loss(
        self, preds, targets, weights, seq_weights, joint_weights
    ) -> Tensor:
        """Loss function for durations [B, L]."""
        losses = self.MSE(preds, targets)
        losses = losses * weights * seq_weights
        if joint_weights is not None:
            losses = losses * joint_weights
        return losses.mean()

    def seq_loss(
        self,
        log_probs,
        mu,
        log_var,
        target,
        weights: Tuple[Tensor, Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]] = (None, None),
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

        # normalise weights to sum to batch size
        B = act_weights.shape[0]
        act_weights = B * act_weights / act_weights.sum()
        seq_weights = B * seq_weights / seq_weights.sum()
        if joint_weights is not None:
            joint_weights = B * joint_weights / joint_weights.sum()
        dur_weights = B * dur_weights / dur_weights.sum()

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
        dur_recon = self.dur_seq_loss(
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
        kld_loss = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * kld_loss

        # final loss
        loss = w_recons_loss + w_kld_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_weight]).float(),
            "dur_weight": torch.tensor([dur_weight]).float(),
        }

    def seq_loss_no_kld(
        self, log_probs, target, weights, label_weights, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durs = self.unpack_encoding(target)
        pred_acts, pred_durs = self.unpack_encoding(log_probs)
        pred_durs = torch.exp(pred_durs)

        act_weights, seq_weights = weights
        label_weights, joint_weights = label_weights
        dur_weights = utils.duration_mask(act_weights)

        assert seq_weights.shape[1] == joint_weights.shape[1] == 1

        # normalise weights to sum to batch size
        B = act_weights.shape[0]
        act_weights = B * act_weights / act_weights.sum()
        seq_weights = B * seq_weights / seq_weights.sum()
        label_weights = B * label_weights / label_weights.sum()
        joint_weights = B * joint_weights / joint_weights.sum()
        dur_weights = B * dur_weights / dur_weights.sum()

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
        dur_recon = self.dur_seq_loss(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_dur_recon = dur_weight * dur_recon

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # final loss
        loss = w_recons_loss

        return {
            "loss": loss,
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "act_weight": torch.tensor([act_weight]).float(),
            "dur_weight": torch.tensor([dur_weight]).float(),
        }

    def end_time_seq_loss(
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)
        pred_durations = torch.exp(pred_durations)

        # normalise mask weights
        mask = mask / mask.mean(-1).unsqueeze(-1)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        act_recon = (recon_act_nlll * mask.view(-1)).mean()
        act_scheduled_weight = (
            self.activity_loss_weight * self.scheduled_act_weight
        )
        w_act_recon = act_scheduled_weight * act_recon

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * mask).mean()
        dur_scheduled_weight = (
            self.duration_loss_weight * self.scheduled_dur_weight
        )
        w_dur_recon = dur_scheduled_weight * recon_end_mse

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # kld loss
        kld_loss = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * kld_loss

        # final loss
        loss = w_recons_loss + w_kld_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_scheduled_weight]).float(),
            "dur_weight": torch.tensor([dur_scheduled_weight]).float(),
        }

    def combined_seq_loss(
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)
        pred_durations = torch.exp(pred_durations)

        # normalise mask weights
        mask = mask / mask.mean(-1).unsqueeze(-1)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        act_recon = (recon_act_nlll * mask.view(-1)).mean()
        act_scheduled_weight = (
            self.activity_loss_weight * self.scheduled_act_weight
        )
        w_act_recon = act_scheduled_weight * act_recon

        # duration loss
        recon_dur_mse = self.MSE(pred_durations, target_durations)
        recon_dur_mse = (recon_dur_mse * mask).mean()

        # ends loss
        target_ends = torch.cumsum(target_durations, dim=-1)
        pred_ends = torch.cumsum(pred_durations, dim=-1)

        recon_end_mse = self.MSE(pred_ends, target_ends)
        recon_end_mse = (recon_end_mse * mask).mean()

        dur_scheduled_weight = (
            self.duration_loss_weight * self.scheduled_dur_weight
        )
        w_dur_recon = dur_scheduled_weight * (recon_end_mse + recon_dur_mse)

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # kld loss
        kld_loss = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * kld_loss

        # final loss
        loss = w_recons_loss + w_kld_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_scheduled_weight]).float(),
            "dur_weight": torch.tensor([dur_scheduled_weight]).float(),
        }

    def discretized_loss(
        self, log_probs, mu, log_var, target, weights, **kwargs
    ) -> dict:
        """Loss function for discretized encoding [N, L]."""
        # activity loss
        recon_act_nlll = self.NLLL(
            log_probs.squeeze().permute(0, 2, 1), target.long()
        )
        scheduled_act_weight = (
            self.activity_loss_weight * self.scheduled_act_weight
        )
        w_recons_loss = scheduled_act_weight * recon_act_nlll

        # kld loss
        unweighted_kld = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * unweighted_kld

        # loss
        loss = recon_act_nlll + w_kld_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([scheduled_act_weight]).float(),
        }

    def discretized_loss_encoded(
        self, log_probs, mu, log_var, target, mask, **kwargs
    ) -> dict:
        """Computes the loss function for discretized encoding [N, L, C]."""

        target_argmax = target.squeeze().argmax(dim=-1)
        return self.discretized_loss(
            log_probs, mu, log_var, target_argmax, mask, **kwargs
        )

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
