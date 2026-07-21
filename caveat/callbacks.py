from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class EpsilonAnchorScheduler(Callback):
    """
    Linearly ramps `pl_module.anchor_alpha` from `start_alpha` to `end_alpha`
    over `warmup_epochs`. The module reads `self.anchor_alpha` inside its
    reparameterization step to interpolate between the conditional prior
    mean (alpha=0) and the true posterior sample (alpha=1).
    """

    def __init__(self, config: dict):
        super().__init__()
        warmup_epochs: int = config.get("warmup_epochs", 20)
        start_alpha: float = config.get("start_alpha", 0.0)
        end_alpha: float = config.get("end_alpha", 1.0)
        print(
            f"[EpsilonAnchorScheduler] warmup_epochs={warmup_epochs}, "
            f"start_alpha={start_alpha}, end_alpha={end_alpha}"
        )
        self.warmup_epochs = warmup_epochs
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha

    def _compute_alpha(self, current_epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.end_alpha
        progress = min(max(current_epoch / self.warmup_epochs, 0.0), 1.0)
        return self.start_alpha + (self.end_alpha - self.start_alpha) * progress

    def on_train_epoch_start(self, trainer, pl_module):
        alpha = self._compute_alpha(trainer.current_epoch)
        pl_module.anchor_alpha = alpha
        pl_module.log("train/anchor_alpha", alpha, on_step=False, on_epoch=True)


class PriorSeparationMonitor(Callback):
    def __init__(self, config: dict):
        self.source_index = config.get(
            "source_index", 0
        )  # column in `c` holding source
        self.check_every_n_epochs = config.get("check_every_n_epochs", 5)
        self.warn_separation_below = config.get("warn_separation_below", 1.0)
        self.min_samples_per_source = config.get("min_samples_per_source", 8)

        # Storage across batches within an epoch, keyed by source value
        self._mu_p_by_source = defaultdict(list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Collect conditional-prior means per source for separation diagnostics."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return

        (x, _), _, (c, l_weights) = batch
        with torch.no_grad():
            mu_p, log_var_p = pl_module.prior_net(pl_module.label_encoder(c))

        source_vals = c[:, self.source_index] if c.dim() > 1 else c
        for source_id in source_vals.unique().tolist():
            mask = source_vals == source_id
            self._mu_p_by_source[source_id].append(mu_p[mask].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute prior separation diagnostics and log."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            self._reset()
            return

        sources = {
            source_id: torch.cat(chunks, dim=0)
            for source_id, chunks in self._mu_p_by_source.items()
            if sum(c.shape[0] for c in chunks) >= self.min_samples_per_source
        }

        if len(sources) < 2:
            print(
                f"\n  [PriorSeparationMonitor] epoch {trainer.current_epoch}: "
                f"fewer than 2 sources with sufficient samples — skipping."
            )
            self._reset()
            return

        metrics = {}

        # 1. Pairwise centroid distances between sources
        centroids = {sid: mu.mean(dim=0) for sid, mu in sources.items()}
        ids = list(centroids.keys())
        pairwise_dists = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = torch.norm(
                    centroids[ids[i]] - centroids[ids[j]], p=2
                ).item()
                pairwise_dists.append(dist)
                metrics[f"separation/prior_dist_{ids[i]}_{ids[j]}"] = dist

        # 2. Separation ratio: between-source spread vs within-source spread
        # (analogous to an F-ratio; >1 means sources are more separated than
        # each source's own internal scatter, i.e. genuinely distinguishable)
        all_mu = torch.cat(list(sources.values()), dim=0)
        grand_mean = all_mu.mean(dim=0)

        between_var = (
            sum(
                mu.shape[0] * (centroids[sid] - grand_mean).pow(2).sum().item()
                for sid, mu in sources.items()
            )
            / all_mu.shape[0]
        )

        within_var = (
            sum(
                (mu - centroids[sid]).pow(2).sum().item()
                for sid, mu in sources.items()
            )
            / all_mu.shape[0]
        )

        separation_ratio = between_var / (within_var + 1e-8)
        metrics["separation/prior_separation_ratio"] = separation_ratio
        metrics["separation/mean_pairwise_dist"] = float(
            np.mean(pairwise_dists)
        )

        pl_module.log_dict(metrics, on_epoch=True)

        # 3. Human-readable warnings
        self._emit_warnings(trainer, separation_ratio, pairwise_dists)

        self._reset()

    def _emit_warnings(self, trainer, separation_ratio, pairwise_dists):
        epoch = trainer.current_epoch
        issues = []

        if separation_ratio < self.warn_separation_below:
            issues.append(
                f"  [PRIOR NOT SEPARATED] separation_ratio = {separation_ratio:.4f} "
                f"(< {self.warn_separation_below}). Conditional prior means are not "
                f"well-separated across sources relative to within-source scatter."
            )

        if issues:
            print(f"\n  Epoch {epoch} prior separation warnings:")
            for msg in issues:
                print(msg)
        # else:
        #     print(
        #         f"\n Epoch {epoch}: prior separation healthy "
        #         f"(ratio={separation_ratio:.3f}, mean_dist={np.mean(pairwise_dists):.3f})"
        #     )

    def _reset(self):
        self._mu_p_by_source.clear()


class CollapseMonitor(Callback):
    def __init__(self, config: dict):
        self.au_threshold = config.get("au_threshold", 0.01)
        self.kl_collapse_threshold = config.get("kl_collapse_threshold", 0.1)
        self.conditional_threshold = config.get("conditional_threshold", 0.05)
        self.check_every_n_epochs = config.get("check_every_n_epochs", 5)
        self.warn_au_below = config.get("warn_au_below", 0.5)

        # Early stopping: opt-in via collapse_patience (falls back to patience).
        # Counts check epochs below conditional_threshold; None disables stopping.
        self.stopping_patience = config.get(
            "collapse_patience", config.get("patience", None)
        )
        self._bad_epochs = 0

        # Storage across batches within an epoch
        self._mus = []
        self._log_vars = []
        self._kl_per_dim = []
        # Decoder sensitivity accumulators (scalar per batch)
        self._decoder_swap_mse = []
        self._decoder_out_var = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Collect latent stats for collapse diagnostics."""
        (x, _), _, (c, l_weights) = batch
        with torch.no_grad():
            batch = pl_module.encode(x, c)
            if len(batch) == 2:
                mu, log_var = batch
                kl = pl_module.kld(mu, log_var)
            elif len(batch) == 4:
                mu, c_mu, log_var, c_log_var = batch
                kl = pl_module.kld([mu, c_mu], [log_var, c_log_var])
                mu = mu - c_mu  # AU diagnostic needs residual from prior mean
            else:
                raise ValueError(
                    f"Expected encode() to return 2 or 4 items, got {len(batch)}"
                )

        self._mus.append(mu.cpu())
        self._log_vars.append(log_var.cpu())

        self._kl_per_dim.append(kl.mean(dim=0).cpu())  # mean over batch

        # Decoder sensitivity: decode same z with real vs. shuffled conditions.
        if hasattr(pl_module, "label_encoder"):
            c_unique = c.unique(dim=0) if c.dim() > 1 else c.unique()
            if len(c_unique) < 2:
                print(
                    f"  [CollapseMonitor] batch {batch_idx}: only {len(c_unique)} distinct "
                    f"condition(s) — skipping decoder sensitivity for this batch."
                )
            else:
                if mu.dim() != 2:
                    raise ValueError(
                        f"[CollapseMonitor] Expected mu to be 2D [N, latent_dim], "
                        f"got shape {tuple(mu.shape)}"
                    )
                if mu.shape[0] != c.shape[0]:
                    raise ValueError(
                        f"[CollapseMonitor] Batch size mismatch: "
                        f"mu has {mu.shape[0]} rows, c has {c.shape[0]} rows"
                    )
                perm = self._derangement(len(c))
                with torch.no_grad():
                    out_real = pl_module.decode(mu, labels=c).float()
                    out_perm = pl_module.decode(mu, labels=c[perm]).float()
                self._decoder_swap_mse.append(
                    (out_real - out_perm).pow(2).mean().item()
                )
                self._decoder_out_var.append(out_real.var().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute collapse diagnostics and log."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            self._reset()
            return

        mu_all = torch.cat(self._mus, dim=0)  # [N, latent_dim]
        log_var_all = torch.cat(self._log_vars, dim=0)
        kl_mean = torch.stack(self._kl_per_dim).mean(dim=0)  # [latent_dim]

        metrics = {}

        # 1. Posterior collapse: active units
        au_mask = mu_all.var(dim=0) > self.au_threshold
        au_pct = au_mask.float().mean().item()
        metrics["collapse/active_units_pct"] = au_pct
        # metrics["collapse/n_active_dims"] = au_mask.sum().item()

        # 2. Posterior collapse: per-dim KL
        collapsed_dims = (kl_mean < self.kl_collapse_threshold).sum().item()
        metrics["collapse/kl_collapsed_dims"] = collapsed_dims
        metrics["collapse/kl_mean"] = kl_mean.mean().item()
        metrics["collapse/kl_min"] = kl_mean.min().item()

        # 3. Decoder sensitivity: ratio of condition-swap MSE to output variance.
        # Accumulated per-batch during on_validation_batch_end.
        if self._decoder_swap_mse:
            swap_mse = np.mean(self._decoder_swap_mse)
            out_var = np.mean(self._decoder_out_var)
            dec_sens = swap_mse / (out_var + 1e-8)
        else:
            dec_sens = float("nan")
        metrics["collapse/decoder_sensitivity"] = dec_sens

        # 4. Posterior variance health
        mean_posterior_var = log_var_all.exp().mean().item()
        metrics["collapse/mean_posterior_var"] = mean_posterior_var

        pl_module.log_dict(metrics, on_epoch=True)

        # 5. Human-readable warnings
        self._emit_warnings(trainer, au_pct, collapsed_dims, dec_sens, kl_mean)

        # 6. Optional early stopping on persistent decoder insensitivity
        if self.stopping_patience is not None and not np.isnan(dec_sens):
            if dec_sens < self.conditional_threshold:
                self._bad_epochs += 1
                if self._bad_epochs >= self.stopping_patience:
                    print(
                        f"\n Stopping: decoder sensitivity {dec_sens:.4f} "
                        f"below {self.conditional_threshold} for "
                        f"{self.stopping_patience} check epochs."
                    )
                    trainer.should_stop = True
            else:
                self._bad_epochs = 0

        self._reset()

    def _derangement(self, n):
        """Random permutation with no fixed points (ensures conditions are always swapped)."""
        for _ in range(10):
            perm = torch.randperm(n)
            if not (perm == torch.arange(n)).any():
                return perm
        # Fallback: cyclic shift always produces a derangement
        return torch.roll(torch.arange(n), 1)

    def _emit_warnings(
        self, trainer, au_pct, collapsed_dims, dec_sens, kl_mean
    ):
        epoch = trainer.current_epoch
        issues = []

        if au_pct < self.warn_au_below:
            issues.append(
                f"  [POSTERIOR COLLAPSE] Only {au_pct:.1%} of latent dims active. "
                f"Consider raising free_bits or reducing beta."
            )

        if collapsed_dims > 0:
            issues.append(
                f"  [KL COLLAPSE] {collapsed_dims} dims have KL < {self.kl_collapse_threshold}. "
                f"Min KL: {kl_mean.min():.4f}"
            )

        if isinstance(dec_sens, float) and not np.isnan(dec_sens):
            if dec_sens < self.conditional_threshold:
                issues.append(
                    f"  [DECODER INSENSITIVE] Decoder sensitivity = {dec_sens:.4f}. "
                    f"Conditions are not influencing output sequences."
                )

        if issues:
            print(f"\n  Epoch {epoch} collapse warnings:")
            for msg in issues:
                print(msg)
        # else:
        #     print(
        #         f"\n Epoch {epoch}: No collapse detected "
        #         f"(AU={au_pct:.1%}, cond_sep={cond_sep:.3f})"
        #     )

    def _reset(self):
        self._mus.clear()
        self._log_vars.clear()
        self._kl_per_dim.clear()
        self._decoder_swap_mse.clear()
        self._decoder_out_var.clear()


class CyclicalBetaAnnealer(Callback):
    def __init__(self, config: dict) -> None:
        """
        n_cycles: how many times to repeat the ramp
        max_beta: the maximum value of beta to reach at the end of each ramp
        ratio: fraction of each cycle spent ramping (rest stays at max_beta)
        """
        self.cycle_len = config.get("cycle_len", 50)
        self.max_beta_multiplier = config.get("max_beta", 10)
        self.target_beta = config.get("target_beta", 0.01)
        self.ratio = config.get("ratio", 0.5)

    def on_train_epoch_start(self, trainer, pl_module):
        cycle_len = self.cycle_len
        cycle_pos = trainer.current_epoch % cycle_len
        ramp_end = cycle_len * self.ratio

        if cycle_pos < ramp_end:
            beta_multiplier = self.max_beta_multiplier * (
                1 - (cycle_pos / ramp_end)
            )
        else:
            beta_multiplier = 1
        beta = self.target_beta * beta_multiplier

        pl_module.beta = beta
        pl_module.log("beta", beta)


class LinearLossScheduler(Callback):
    def __init__(self, config: dict) -> None:
        self.min_epochs = config.get("min_epochs", 0)
        self.kld_schedule = config.get("kld_loss_schedule", None)
        self.act_schedule = config.get("activity_loss_schedule", None)
        self.dur_schedule = config.get("duration_loss_schedule", None)
        self.start_schedule = config.get("start_loss_schedule", None)
        self.end_schedule = config.get("end_loss_schedule", None)
        self.total_dur_schedule = config.get(
            "total_duration_loss_schedule", None
        )
        self.label_schedule = config.get("label_loss_schedule", None)
        self.validate_weights_schedule("KLD", self.kld_schedule)
        self.validate_weights_schedule("ACT", self.act_schedule)
        self.validate_weights_schedule("DUR", self.dur_schedule)
        self.validate_weights_schedule("END", self.end_schedule)
        self.validate_weights_schedule("ATT", self.label_schedule)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        current_epoch = trainer.current_epoch
        if self.kld_schedule is not None:
            s, e = self.kld_schedule
            if current_epoch < s:
                pl_module.scheduled_kld_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_kld_weight = 1.0
            else:
                pl_module.scheduled_kld_weight = (current_epoch - s) / (e - s)

        if self.act_schedule is not None:
            s, e = self.act_schedule
            if current_epoch < s:
                pl_module.scheduled_act_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_act_weight = 1.0
            else:
                pl_module.scheduled_act_weight = (current_epoch - s) / (e - s)

        if self.dur_schedule is not None:
            s, e = self.dur_schedule
            if current_epoch < s:
                pl_module.scheduled_dur_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_dur_weight = 1.0
            else:
                pl_module.scheduled_dur_weight = (current_epoch - s) / (e - s)

        if self.start_schedule is not None:
            s, e = self.start_schedule
            if current_epoch < s:
                pl_module.scheduled_start_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_start_weight = 1.0
            else:
                pl_module.scheduled_start_weight = (current_epoch - s) / (e - s)

        if self.end_schedule is not None:
            s, e = self.end_schedule
            if current_epoch < s:
                pl_module.scheduled_end_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_end_weight = 1.0
            else:
                pl_module.scheduled_end_weight = (current_epoch - s) / (e - s)

        if self.total_dur_schedule is not None:
            s, e = self.total_dur_schedule
            if current_epoch < s:
                pl_module.scheduled_total_dur_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_total_dur_weight = 1.0
            else:
                pl_module.scheduled_total_dur_weight = (current_epoch - s) / (
                    e - s
                )

        if self.label_schedule is not None:
            s, e = self.label_schedule
            if current_epoch < s:
                pl_module.scheduled_label_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_label_weight = 1.0
            else:
                pl_module.scheduled_label_weight = (current_epoch - s) / (e - s)

    def validate_weights_schedule(self, name, schedule):
        if schedule is None:
            return None
        s, e = schedule
        if s > e:
            raise ValueError(f"Invalid schedule for {name}: {schedule}")
        if s < 0 or e < 0:
            raise ValueError(f"Invalid schedule for {name}: {schedule}")
        if e < self.min_epochs:
            print(
                f"WARNING: {name} schedule {schedule} ends after min_epochs {self.min_epochs}"
            )
        print(
            f"Found {name} schedule: {s} -> {e}. Check that this is ok with your epochs."
        )
