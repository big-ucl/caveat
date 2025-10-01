from typing import Tuple

import torch
from torch import Tensor, nn

from caveat.models import CatBase, utils
from caveat.models.continuous.vae_lstm import Decoder, Encoder


class CatVAEContLSTM(CatBase):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.latent_cats = config.get("latent_categories", 2)
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config.get("dropout", 0)
        length, _ = self.in_shape
        self.head_hidden_size = config.get("head_hidden_size", self.hidden_size)
        self.head_depth = config.get("head_depth", 2)

        self.encoder = Encoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            eos=self.eos,
            head_depth=self.head_depth,
            head_hidden_size=self.head_hidden_size,
        )
        self.unflattened_shape = (self.hidden_n, self.hidden_size * 2)
        hidden_out_size = 2 * self.hidden_n * self.hidden_size
        self.latent_shape = (self.latent_dim, self.latent_cats)
        self.latent_size = self.latent_dim * self.latent_cats

        self.encoder_resize = nn.Linear(hidden_out_size, self.latent_size)
        self.latent_activation = nn.Softmax(dim=-1)
        self.decoder_resize = nn.Linear(self.latent_size, hidden_out_size)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        z = z.view(-1, self.latent_size)
        h = self.decoder_resize(z)

        # initialize hidden state
        hidden = h.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])
        batch_size = z.shape[0]

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # use teacher forcing
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        if self.norm_durations:
            log_probs = utils.normalise_log_durations(
                log_probs, sos=self.sos, eos=self.eos
            )

        return log_probs
