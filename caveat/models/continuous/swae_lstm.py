from typing import Optional, Tuple

import torch
from torch import Tensor

from caveat.models import utils
from caveat.models.continuous.vae_lstm import VAEContLSTM


class SWAEContLSTM(VAEContLSTM):
    def loss_function(
        self,
        log_probs,
        mu,
        log_var,
        target,
        weights: Tuple[Tensor, Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]] = (None, None),
        z: Optional[Tensor] = None,
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

        # start time loss
        start_weight = self.start_loss_weight * self.scheduled_start_weight
        start_recon = self.start_seq_loss_detached(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_start_recon = start_weight * start_recon

        # end time loss
        end_weight = self.end_loss_weight * self.scheduled_end_weight
        end_recon = self.end_seq_loss_detached(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_end_recon = end_weight * end_recon

        # total_duration loss
        total_dur_weight = (
            self.total_duration_loss_weight * self.scheduled_total_dur_weight
        )
        total_dur_recon = self.total_duration_loss(
            preds=pred_durs,
            targets=target_durs,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_total_dur_recon = total_dur_weight * total_dur_recon

        # reconstruction loss
        w_recons_loss = (
            w_act_recon
            + w_dur_recon
            + w_start_recon
            + w_end_recon
            + w_total_dur_recon
        )

        # kld loss
        prior_z = torch.randn_like(mu)
        kld_loss = self.sliced_wasserstein_distance(
            z, prior_z, num_projections=100
        )
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
            "end_recon": w_end_recon.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_weight]).float(),
            "dur_weight": torch.tensor([dur_weight]).float(),
            "end_weight": torch.tensor([end_weight]).float(),
        }

    def sliced_wasserstein_distance(
        self, z: Tensor, prior_z: Tensor, num_projections: int = 20
    ) -> Tensor:
        """Computes the Sliced Wasserstein Distance between two distributions.

        Args:
            z (Tensor): Samples from the first distribution [N, latent_dim].
            prior_z (Tensor): Samples from the second distribution [N, latent_dim].
            num_projections (int): Number of random projections to use.

        Returns:
            Tensor: The Sliced Wasserstein Distance.
        """
        latent_dim = z.size(1)

        # Generate random projections and normalize them
        projections = torch.randn(num_projections, latent_dim).to(z.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)

        # Project both distributions onto the random directions
        z_projections = z @ projections.t()
        prior_z_projections = prior_z @ projections.t()

        # Sort the projections
        z_projections, _ = torch.sort(z_projections, dim=0)
        prior_z_projections, _ = torch.sort(prior_z_projections, dim=0)

        # Compute the average squared differences between the sorted projections
        swd = torch.mean((z_projections - prior_z_projections) ** 2)

        return swd
