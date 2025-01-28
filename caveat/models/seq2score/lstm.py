from typing import List, Optional

import torch
from torch import Tensor, nn

from caveat.models import Base, CustomDurationModeDistanceEmbedding


class Seq2ScoreLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with conditionality."""
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "ConditionalLSTM requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        # self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.act_encodings, self.mode_encodings = self.encodings

        # encodings
        if self.hidden_size < 4:
            raise ValueError("Hidden size must be at least 4.")
        self.hidden_mode_size = config.get("hidden_mode_size")
        if self.hidden_mode_size is None:
            self.hidden_mode_size = (self.hidden_size - 2) // 2
        self.hidden_act_size = config.get("hidden_act_size")
        if self.hidden_act_size is None:
            self.hidden_act_size = self.hidden_size - 2 - self.hidden_mode_size

        self.encoder = Encoder(
            act_embeddings=self.act_encodings,
            mode_embeddings=self.mode_encodings,
            hidden_size=self.hidden_size,
            hidden_act_size=self.hidden_act_size,
            hidden_mode_size=self.hidden_mode_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2
        self.fc_hidden = nn.Linear(
            flat_size_encode + self.conditionals_size, flat_size_encode
        )
        self.score_layer = nn.Linear(flat_size_encode, 1)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(
        self,
        x: Tensor,
        conditionals: Optional[Tensor] = None,
        target=None,
        **kwargs,
    ) -> List[Tensor]:
        z = self.encode(x)  # [N, flat]
        score = self.decode(z, conditionals=conditionals, target=target)
        return score

    def encode(self, input: Tensor) -> Tensor:
        # [N, L, C]
        return self.encoder(input)

    def decode(
        self, z: Tensor, conditionals: Tensor, target=None, **kwargs
    ) -> List[Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # add conditionlity to z
        z = torch.cat((z, conditionals), dim=-1)
        # initialize hidden state as inputs
        h = self.fc_hidden(z)
        h = self.score_layer(h)
        return h

    def loss_function(
        self, scores: Tensor, target: Tensor, mask: Tensor, **kwargs
    ) -> dict:

        # duration loss
        loss = self.MSE(scores.squeeze(), target)

        return {"loss": loss}

    def predict_step(self, batch, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            batch
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        (x, _), (y, _), (labels, _) = batch
        x = x.to(device)
        return (x, y, labels, self.forward(x=x, conditionals=labels, **kwargs))


class Encoder(nn.Module):
    def __init__(
        self,
        act_embeddings: int,
        mode_embeddings: int,
        hidden_size: int,
        hidden_act_size: int,
        hidden_mode_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder.

        Args:
            act_embeddings (int): number of activity embeddings.
            mode_embeddings (int): number of mode embeddings.
            hidden_size (int): lstm hidden size.
            hidden_act_size (int): hidden size for activity embeddings.
            hidden_mode_size (int): hidden size for mode embeddings.
            num_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = CustomDurationModeDistanceEmbedding(
            act_embeddings=act_embeddings,
            mode_embeddings=mode_embeddings,
            hidden_act_size=hidden_act_size,
            hidden_mode_size=hidden_mode_size,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden
