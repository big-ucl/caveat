from typing import List, Optional

import torch
import torch.nn.functional as F

# the label cross-attention encoder from the earlier file
from torch import Tensor, nn

from caveat.models import Base


class AutoEventLSTM(Base):
    def __init__(self, *args, **kwargs):
        """Event-level transformer decoder (type, start, end per token),
        conditioned on labels via cross-attention. Use with EventEncoder.

        Same API as AutoDiscLSTM / AutoDiscTransformer. Output log-probs are one
        tensor [N, L, E + K + K] (type | start | end, each log-softmaxed).
        """
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
            raise UserWarning(
                "AutoDiscEventLSTM requires labels_size, please check you have configured a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = 1  # dummy value for the predict dataloader
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        self.nhead = config.get("nhead", 4)
        self.ff_mult = config.get("ff_mult", 4)
        top_sampler = config.get("top_sampler", False)
        length = self.in_shape[0]

        # must match EventEncoder(n_bins=...)
        self.n_bins = config.get("n_bins", 96)
        self.time_classes = self.n_bins + 1

        # soft-label settings and loss weights (paper: w1..w5, values not given)
        self.soft_window = config.get("soft_window", 2)
        self.soft_main = config.get("soft_main", 1.0)
        self.soft_side = config.get("soft_side", 0.1)
        self.loss_weights = config.get("loss_weights", [1.0] * 5)
        # predict(): dense one-hot [N,L,E+2K] (works with decode(argmax=True)) or
        # index tensor [N,L,3] (use with decode(argmax=False), much lighter)
        self.predict_indices = config.get("predict_indices", False)
        # label dropout: during training each label field is independently replaced
        # by a learned per-field "missing" token with this probability
        self.label_dropout = config.get("label_dropout", 0.0)

        if self.hidden_size % self.nhead != 0:
            raise UserWarning(
                f"hidden_size ({self.hidden_size}) must be divisible by nhead ({self.nhead})"
            )

        self.label_encoder = LabelMemoryEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )
        self.missing_label = nn.Parameter(
            torch.zeros(len(self.label_embed_sizes), self.hidden_size)
        )
        nn.init.normal_(self.missing_label, std=0.02)
        self.decoder = EventDecoder(
            n_types=self.encodings,
            time_classes=self.time_classes,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            top_sampler=top_sampler,
        )

    # ------------------------------------------------------------------ api
    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> List[Tensor]:
        log_probs = self.decode(z=x, labels=labels, target=target)
        return [log_probs, Tensor([]), Tensor([]), Tensor([])]

    def encode(self, input: Tensor):
        return None

    def decode(
        self,
        z: Tensor,
        labels: Tensor,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        memory = self.label_encoder(labels)  # [N, F, H]
        if self.training and self.label_dropout > 0 and target is not None:
            memory = self._drop_labels(memory)
        if target is None:
            return self.decoder.generate_log_probs(memory)
        return self.decoder(memory, target.long())

    def _drop_labels(self, memory: Tensor) -> Tensor:
        """Replace each label-field token by its learned 'missing' token with
        probability label_dropout (training only, independently per sample/field).
        """
        N = memory.shape[0]
        drop = (
            torch.rand(memory.shape[:2], device=memory.device)
            < self.label_dropout
        )
        return torch.where(
            drop.unsqueeze(-1), self.missing_label.expand(N, -1, -1), memory
        )

    def predict(
        self, z: Tensor, labels: Tensor, device: int, **kwargs
    ) -> Tensor:
        labels = labels.to(device)
        memory = self.label_encoder(labels)
        tokens = self.decoder.generate(memory)  # [N, L, 3] long (sampled)
        if self.predict_indices:
            return tokens.float()
        E, K = self.encodings, self.time_classes
        return torch.cat(
            [
                F.one_hot(tokens[..., 0], E),
                F.one_hot(tokens[..., 1], K),
                F.one_hot(tokens[..., 2], K),
            ],
            dim=-1,
        ).float()

    # ------------------------------------------------------------------ loss
    def loss_function(
        self, log_probs: Tensor, target: Tensor, weights: Tensor, **kwargs
    ) -> dict:
        """Type CE + soft-label CE on start/end + ordering penalties.

        If `weights` is a tensor with one value per sequence ([N] or [N, 1]), every
        term is a weighted mean with that weight per sequence (e.g. inverse
        sequence frequency from EventEncoder). Anything else is ignored."""
        w = self._seq_weights(weights, target.shape[0], log_probs.device)

        def mm(x, mask):
            return masked_mean(x, mask, w)

        E, K = self.encodings, self.time_classes
        lp_act, lp_start, lp_end = torch.split(log_probs, [E, K, K], dim=-1)
        t_act, t_start, t_end = target.long().unbind(dim=-1)

        is_eos = (t_act == 1).long()  # 0=<SOS>, 1=<EOS>, 2+ = activities
        live = (
            torch.cumsum(is_eos, 1) - is_eos
        ) == 0  # up to and incl. first EOS
        real = t_act > 1  # real activity slots

        # activity type: all live positions (includes learning to emit EOS)
        nll_act = -lp_act.gather(-1, t_act.unsqueeze(-1)).squeeze(-1)
        l_type = mm(nll_act, live)

        # start / end: soft-label CE on real events only
        l_start = mm(self._soft_nll(lp_start, t_start), real)
        l_end = mm(self._soft_nll(lp_end, t_end), real)

        # ordering penalties on expected bin positions (differentiable)
        bins = torch.arange(K, device=log_probs.device, dtype=log_probs.dtype)
        e_start = (lp_start.exp() * bins).sum(-1) / self.n_bins
        e_end = (lp_end.exp() * bins).sum(-1) / self.n_bins
        l_seq = mm(F.relu(e_start - e_end), real)  # start <= end
        l_ord = mm(
            F.relu(e_end[:, :-1] - e_start[:, 1:]), real[:, :-1] & real[:, 1:]
        )  # prev end <= next start

        w1, w2, w3, w4, w5 = self.loss_weights
        loss = w1 * l_type + w2 * l_start + w3 * l_end + w4 * l_ord + w5 * l_seq
        return {
            "loss": loss,
            "recon_loss": (l_type + l_start + l_end).detach(),
            "type_loss": l_type.detach(),
            "start_loss": l_start.detach(),
            "end_loss": l_end.detach(),
            "order_loss": l_ord.detach(),
            "seq_loss": l_seq.detach(),
        }

    def _seq_weights(self, weights, N, device):
        if isinstance(weights, Tensor) and weights.numel() == N:
            return weights.to(device=device, dtype=torch.float32).reshape(N, 1)
        if weights is not None and not getattr(self, "_warned_w", False):
            print(
                "AutoDiscEventTransformer: ignoring weights (expected one per sequence)"
            )
            self._warned_w = True
        return None

    def _soft_nll(self, logp: Tensor, y: Tensor) -> Tensor:
        """Soft-label cross-entropy: weight w_main on the true bin, w_side on
        +-1..soft_window neighbours (normalised to sum to 1). Returns [N, L]."""
        K = logp.shape[-1]
        n = self.soft_window
        offs = torch.arange(-n, n + 1, device=y.device)
        w = torch.full(
            offs.shape, self.soft_side, device=y.device, dtype=logp.dtype
        )
        w[n] = self.soft_main
        idx = y.unsqueeze(-1) + offs  # [N, L, 2n+1]
        valid = (idx >= 0) & (idx < K)
        soft = torch.zeros_like(logp).scatter_add_(
            -1, idx.clamp(0, K - 1), w * valid.to(logp.dtype)
        )
        soft = soft / soft.sum(-1, keepdim=True)
        return -(soft * logp).sum(-1)


def masked_mean(x: Tensor, mask: Tensor, w: Optional[Tensor] = None) -> Tensor:
    """Mean of x over mask; w [N, 1] optionally weights each sequence."""
    m = mask.to(x.dtype)
    if w is not None:
        m = m * w
    return (x * m).sum() / m.sum().clamp(min=1e-8)


class EventDecoder(nn.Module):
    def __init__(
        self,
        n_types,
        time_classes,
        hidden_size,
        num_layers,
        max_length,
        dropout=0.0,
        sos=0,
        top_sampler=False,
    ):
        """LSTM counterpart of EventDecoder (same API and heads).

        Conditioning follows AutoDiscLSTM: the label memory [N, F, H] is pooled
        and projected to the initial (h, c) of every LSTM layer, instead of
        being cross-attended to.

        Input token = type_emb + start_emb + end_emb (no position embedding,
        the recurrence tracks position and the previous end bin carries the
        time of day). Heads, as in EventDecoder:
            type  <- h
            start <- h + emb(type)
            end   <- h + emb(type) + emb(start)
        """
        super().__init__()
        self.n_types = n_types
        self.K = time_classes
        self.max_length = max_length
        self.sos = sos
        self.num_layers = num_layers
        d = hidden_size
        self.hidden_size = d

        self.type_emb = nn.Embedding(n_types, d)
        self.start_emb = nn.Embedding(time_classes, d)
        self.end_emb = nn.Embedding(time_classes, d)
        for e in (self.type_emb, self.start_emb, self.end_emb):
            nn.init.normal_(e.weight, std=0.02)
        self.emb_dropout = nn.Dropout(dropout)

        # pooled labels -> initial (h, c) for all layers
        self.init_norm = nn.LayerNorm(d)
        self.init_ff = nn.Sequential(
            nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, 2 * num_layers * d)
        )

        self.lstm = nn.LSTM(
            d,
            d,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)

        self.type_head = nn.Linear(d, n_types)
        self.type_cond = nn.Embedding(n_types, d)
        self.start_head = nn.Linear(d, time_classes)
        self.start_cond = nn.Embedding(time_classes, d)
        self.end_head = nn.Linear(d, time_classes)
        nn.init.normal_(self.type_cond.weight, std=0.02)
        nn.init.normal_(self.start_cond.weight, std=0.02)

        self.sample = self.topk if top_sampler else self.multinomial

    def init_state(self, memory: Tensor):
        """memory [N, F, H] -> (h0, c0), each [layers, N, H]."""
        pooled = self.init_norm(memory.mean(dim=1))
        flat = self.init_ff(pooled)  # [N, 2*layers*H]
        h, c = flat.view(-1, 2, self.num_layers, self.hidden_size).unbind(1)
        return (
            h.permute(1, 0, 2).contiguous(),
            c.permute(1, 0, 2).contiguous(),
        )

    def embed(self, tokens: Tensor) -> Tensor:
        """tokens [N, T, 3] -> [N, T, d]."""
        x = (
            self.type_emb(tokens[..., 0])
            + self.start_emb(tokens[..., 1])
            + self.end_emb(tokens[..., 2])
        )
        return self.emb_dropout(x)

    def run(self, tokens: Tensor, state):
        """tokens [N, T, 3] -> (hidden [N, T, d], new state)."""
        out, state = self.lstm(self.embed(tokens), state)
        return self.out_dropout(out), state

    def shift_right(self, target: Tensor) -> Tensor:
        N = target.shape[0]
        start = torch.zeros((N, 1, 3), dtype=torch.long, device=target.device)
        start[..., 0] = self.sos
        return torch.cat([start, target[:, :-1]], dim=1)

    def forward(self, memory: Tensor, target: Tensor) -> Tensor:
        """Teacher forcing in one pass (cuDNN LSTM over the whole sequence).
        Returns [N, L, E + K + K]."""
        h, _ = self.run(self.shift_right(target), self.init_state(memory))
        h_s = h + self.type_cond(target[..., 0])
        h_e = h_s + self.start_cond(target[..., 1])
        return torch.cat(
            [
                F.log_softmax(self.type_head(h), dim=-1),
                F.log_softmax(self.start_head(h_s), dim=-1),
                F.log_softmax(self.end_head(h_e), dim=-1),
            ],
            dim=-1,
        )

    @torch.no_grad()
    def generate(self, memory: Tensor) -> Tensor:
        return self._generate(memory)[0]

    def generate_log_probs(self, memory: Tensor) -> Tensor:
        return self._generate(memory)[1]

    def _generate(self, memory: Tensor):
        """Free-running generation, sampling type -> start -> end per step.
        The LSTM state is carried forward, so this is O(L) (no re-running the
        prefix). Returns sampled tokens [N, L, 3] and log-probs [N, L, E+K+K].
        """
        N = memory.shape[0]
        state = self.init_state(memory)
        tok = torch.zeros((N, 1, 3), dtype=torch.long, device=memory.device)
        tok[..., 0] = self.sos
        toks, lps = [], []
        for _ in range(self.max_length):
            h, state = self.run(tok, state)
            h = h[:, -1]
            lt = F.log_softmax(self.type_head(h), dim=-1)
            a = self.sample(lt)
            h_s = h + self.type_cond(a)
            ls = F.log_softmax(self.start_head(h_s), dim=-1)
            s = self.sample(ls)
            h_e = h_s + self.start_cond(s)
            le = F.log_softmax(self.end_head(h_e), dim=-1)
            e = self.sample(le)
            lps.append(torch.cat([lt, ls, le], dim=-1))
            tok = torch.stack([a, s, e], -1).unsqueeze(1)
            toks.append(tok)
        return torch.cat(toks, dim=1), torch.stack(lps, dim=1)

    # sampling: log-probs [N, V] -> [N]
    def multinomial(self, logp: Tensor) -> Tensor:
        return torch.multinomial(logp.exp(), 1).squeeze(-1)

    def topk(self, logp: Tensor) -> Tensor:
        return logp.argmax(dim=-1)


class LabelMemoryEncoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size, dropout=0.0):
        """Embed each label field to its own token, giving a memory [N, F, H]
        for cross-attention.

        Uses the same single combined embedding trick as the LSTM LabelEncoder
        (offsets shift each field into its own slice of the vocab, so each
        field has distinct embeddings), but keeps the fields as separate
        tokens instead of summing them. The decoder can then attend to
        individual attributes.
        """
        super().__init__()
        offsets = [0]
        for s in label_embed_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_vocab_size = sum(label_embed_sizes)
        self.embed = nn.Embedding(total_vocab_size, hidden_size)
        nn.init.orthogonal_(self.embed.weight)
        # learned per-field embedding (extra identity signal, cheap)
        self.field_embed = nn.Parameter(
            torch.zeros(len(label_embed_sizes), hidden_size)
        )
        nn.init.normal_(self.field_embed, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        # x: [N, F]
        shifted = x.long() + self.offsets
        h = self.norm(self.embed(shifted) + self.field_embed)  # [N, F, H]
        return h + self.ff(h)
