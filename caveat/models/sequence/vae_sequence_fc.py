from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from caveat.models import Base, CustomDurationEmbedding, utils


class VAESeqFC(Base):
    def __init__(self, *args, **kwargs):
        """Fully connected encoder and decoder with embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        hidden_layers = list
        latent_dim = int
        dropout = Optional[float]

        encoded_size = self.encodings + 1
        hidden_layers = utils.build_hidden_layers(config)
        latent_dim = config["latent_dim"]
        dropout = config.get("dropout", 0)
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            length=self.in_shape[0],
            encoded_size=encoded_size,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            length=self.in_shape[0],
            in_size=self.encoder.flat_size,
            encoded_size=encoded_size,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        self.fc_mu = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, self.encoder.flat_size)

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
        hidden = self.fc_hidden(z)
        log_probs = self.decoder(hidden)
        return log_probs


class Encoder(nn.Module):
    def __init__(
        self,
        length: int,
        encoded_size: int,
        hidden_layers: list,
        dropout: float = 0.1,
    ):
        """2d flatten to 1d then fully connected.

        Args:
            length (int): number of time steps.
            encoded_size (int): number of encoding classes and hidden size.
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.embedding = CustomDurationEmbedding(
            encoded_size, encoded_size, dropout=dropout
        )
        modules = []

        input_size = length * encoded_size
        self.flat_embed_size = input_size
        for hidden_channels in hidden_layers:
            size = length * hidden_channels
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, size),
                    nn.BatchNorm1d(size),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
            )
            input_size = size
        self.flat_size = size
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        y = self.dropout(self.embedding(x.int()))
        y = y.flatten(1)
        y = self.encoder(y)
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        length: int,
        in_size: int,
        encoded_size: int,
        hidden_layers: list,
        dropout: float = 0.1,
    ):
        """2d flatten to 1d then fully connected.

        Args:
            length (int): number of time steps.
            in_size (int): input size.
            encoded_size (list): list of target shapes from encoder.
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Decoder, self).__init__()
        self.length = length
        self.hidden_size = encoded_size
        hidden_layers.reverse()
        modules = []

        input_size = in_size
        for i in range(len(hidden_layers)):
            hidden_channels = hidden_layers[i]
            size = length * hidden_channels
            block = [nn.Linear(input_size, size), nn.BatchNorm1d(size)]
            if i < len(hidden_layers) - 1:
                block.append(nn.LeakyReLU())
                block.append(nn.Dropout(dropout))
            modules.append(nn.Sequential(*block))
            input_size = size

        # Final layer
        modules.append(nn.Linear(input_size, length * encoded_size))

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(*modules)
        self.logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, hidden, **kwargs):
        y = self.decoder(hidden)
        y = y.view(-1, self.length, self.hidden_size)
        acts_logits, durations = torch.split(
            y, [self.hidden_size - 1, 1], dim=-1
        )
        acts_log_probs = self.logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        durations = torch.log(durations)
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)

        return log_prob_outputs
