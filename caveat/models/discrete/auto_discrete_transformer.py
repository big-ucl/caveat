from typing import List, Optional

import torch
from torch import Tensor, exp, nn

from caveat.models import Base


class AutoDiscTransformer(Base):
    def __init__(self, *args, **kwargs):
        """Transformer decoder, conditioned on labels via cross-attention.

        Drop-in equivalent of AutoDiscLSTM: no encoder, a label-conditioned
        autoregressive decoder over discretised activity sequences [N, L].
        """
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
            raise UserWarning(
                "AutoDiscTransformer requires labels_size, please check you have configured a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = 1  # dummy value for the predict dataloader
        self.hidden_size = config["hidden_size"]  # d_model
        self.hidden_n = config["hidden_n"]  # number of decoder layers
        self.dropout = config["dropout"]
        self.nhead = config.get("nhead", 4)
        self.ff_mult = config.get("ff_mult", 4)
        top_sampler = config.get("top_sampler", False)
        length = self.in_shape[0]

        if self.hidden_size % self.nhead != 0:
            raise UserWarning(
                f"hidden_size ({self.hidden_size}) must be divisible by nhead ({self.nhead})"
            )

        self.label_encoder = LabelMemoryEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        self.decoder = TransformerDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_layers=self.hidden_n,
            nhead=self.nhead,
            ff_size=self.ff_mult * self.hidden_size,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            top_sampler=top_sampler,
        )

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> List[Tensor]:
        log_probs = self.decode(z=x, labels=labels, target=target)
        return [log_probs, Tensor([]), Tensor([]), Tensor([])]

    def loss_function(
        self, log_probs: Tensor, target: Tensor, weights: Tensor, **kwargs
    ) -> dict:
        """Loss function for discretized encoding [N, L]."""
        recon_act_nlll = self.NLLL(log_probs.permute(0, 2, 1), target)
        return {"loss": recon_act_nlll, "recon_loss": recon_act_nlll}

    def encode(self, input: Tensor):
        return None

    def decode(
        self,
        z: Tensor,
        labels: Tensor,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Decode labels to a batch of output sequences.

        Returns:
            tensor: log probs [N, steps, acts].
        """
        memory = self.label_encoder(labels)  # [N, num_labels, hidden]
        return self.decoder(
            memory=memory,
            target=target,
            teacher_forcing_ratio=(
                self.teacher_forcing_ratio if target is not None else 0.0
            ),
        )

    def predict(
        self, z: Tensor, labels: Tensor, device: int, **kwargs
    ) -> Tensor:
        z = z.to(device)
        labels = labels.to(device)
        return exp(self.decode(z=z, labels=labels))


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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        nhead,
        ff_size,
        max_length,
        dropout: float = 0.0,
        sos: int = 0,
        top_sampler: bool = False,
    ):
        """Causal transformer decoder with cross-attention to label memory.

        Args:
            input_size (int): vocab size (encodings).
            hidden_size (int): model width.
            output_size (int): output vocab size.
            num_layers (int): number of decoder layers.
            nhead (int): attention heads.
            ff_size (int): feed-forward width.
            max_length (int): sequence length.
            dropout (float): dropout probability.
            sos (int): start of sequence token.
            top_sampler (bool): greedy sampling instead of multinomial.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.sos = sos

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)

        if top_sampler:
            print("Using topk sampling")
            self.sample = self.topk
        else:
            print("Using multinomial sampling")
            self.sample = self.multinomial

    # ---- core pass -------------------------------------------------------
    def run(self, tokens: Tensor, memory: Tensor) -> Tensor:
        """tokens [N, T] -> logits [N, T, V], causal over T."""
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        x = self.embedding(tokens) + self.pos_embedding(pos)
        x = self.embedding_dropout(x)
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device
        )
        h = self.transformer(
            tgt=x, memory=memory, tgt_mask=mask, tgt_is_causal=True
        )
        return self.fc(h)

    def shift_right(self, target: Tensor) -> Tensor:
        sos = torch.full(
            (target.shape[0], 1),
            self.sos,
            dtype=torch.long,
            device=target.device,
        )
        return torch.cat([sos, target[:, :-1]], dim=1)

    # ---- forward ---------------------------------------------------------
    def forward(self, memory, target=None, teacher_forcing_ratio=0.0, **kwargs):
        if target is None or teacher_forcing_ratio <= 0:
            logits = self.generate(memory)
        elif teacher_forcing_ratio >= 1:
            # standard parallel teacher forcing: one pass over the whole sequence
            logits = self.run(self.shift_right(target.long()), memory)
        else:
            logits = self.scheduled_sampling(
                memory, target.long(), teacher_forcing_ratio
            )
        return self.activity_logprob_activation(logits)

    def generate(self, memory: Tensor) -> Tensor:
        """Free-running autoregressive decoding (no KV cache, so O(L^2) passes)."""
        N = memory.shape[0]
        tokens = torch.full(
            (N, 1), self.sos, dtype=torch.long, device=memory.device
        )
        outputs = []
        for _ in range(self.max_length):
            logits = self.run(tokens, memory)[:, -1:]  # [N, 1, V]
            outputs.append(logits)
            tokens = torch.cat([tokens, self.sample(logits)], dim=1)
        return torch.cat(outputs, dim=1)

    def scheduled_sampling(self, memory, target, ratio) -> Tensor:
        """Two-pass approximation of per-token scheduled sampling.

        Pass 1 (no grad): parallel teacher-forced predictions are sampled.
        Inputs are then mixed per position: target with prob `ratio`, own
        sample otherwise. Pass 2 (grad) runs on the mixed inputs.
        """
        with torch.no_grad():
            logits0 = self.run(self.shift_right(target), memory)
            sampled = self.sample_all(logits0)  # [N, L]
            use_teacher = torch.rand(target.shape, device=target.device) < ratio
            mixed = torch.where(use_teacher, target, sampled)
        return self.run(self.shift_right(mixed), memory)

    # ---- sampling --------------------------------------------------------
    def multinomial(self, logits):
        # logits [N, 1, V] -> [N, 1]
        probs = torch.softmax(logits[:, -1], dim=-1)
        return torch.multinomial(probs, 1)

    def topk(self, logits):
        return logits[:, -1].argmax(dim=-1, keepdim=True).detach()

    def sample_all(self, logits):
        # logits [N, L, V] -> [N, L]
        N, L, V = logits.shape
        flat = logits.reshape(N * L, 1, V)
        return self.sample(flat).reshape(N, L)
