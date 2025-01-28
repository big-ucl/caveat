from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from caveat.models import Base, CustomDurationEmbedding, utils
from caveat.models.utils import calc_output_padding_1d, conv1d_size


class VAESeqCNN1D(Base):
    def __init__(self, *args, **kwargs):
        """CNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        hidden_layers = list
        latent_dim = int
        dropout = Optional[float]
        kernel_size = Optional[int]
        stride = Optional[int]
        padding = Optional[int]

        encoded_size = config.get("embed_size", self.encodings + 1)
        hidden_layers = utils.build_hidden_layers(config)
        latent_dim = config["latent_dim"]
        dropout = config.get("dropout", 0)
        kernel_size = config.get("kernel_size", 2)
        stride = config.get("stride", 2)
        padding = config.get("padding", 1)

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_encoding=self.encodings,
            encoded_size=encoded_size,
            in_shape=self.in_shape,
            hidden_layers=hidden_layers,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.decoder = Decoder(
            encoded_size=encoded_size,
            target_shapes=self.encoder.shapes,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
        hidden = hidden.view(self.encoder.shape_before_flattening)
        log_probs = self.decoder(hidden)
        return log_probs


class Encoder(nn.Module):
    def __init__(
        self,
        input_encoding: int,
        encoded_size: int,
        in_shape: tuple,
        hidden_layers: list,
        dropout: float = 0.1,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 1,
    ):
        """2d Convolutions Encoder.

        Args:
            encoded_size (int): number of encoding classes and hidden size.
            in_shape (tuple[int, int, int]): [C, time_step, activity_encoding].
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (int): kernel size. Defaults to 2.
            stride (int): stride. Defaults to 2.
            padding (int): padding. Defaults to 1.
        """
        super(Encoder, self).__init__()
        print(in_shape)
        length = in_shape[0]
        self.embedding = CustomDurationEmbedding(
            input_encoding, encoded_size, dropout=dropout
        )

        channels = encoded_size
        self.shapes = []
        modules = []

        for hidden_channels in hidden_layers:
            self.shapes.append((channels, length))
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
            )
            length = conv1d_size(
                length=length,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            channels = hidden_channels
        self.shapes.append((channels, length))

        self.encoder = nn.Sequential(*modules)
        self.shape_before_flattening = (-1, channels, length)
        self.flat_size = int(channels * length)

    def forward(self, x):
        y = self.embedding(x.int())
        y = y.permute(0, 2, 1)
        y = self.encoder(y)
        y = y.flatten(start_dim=1)
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        encoded_size: int,
        target_shapes: list,
        dropout: float = 0.1,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
    ):
        """1d Conv Decoder.

        Args:
            encoded_size (int): number of encoding classes and hidden size.
            target_shapes (list): list of target shapes.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (int): kernel size. Defaults to 3.
            stride (int): stride. Defaults to 2.
            padding (int): padding. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.hidden_size = encoded_size
        modules = []
        target_shapes.reverse()

        for i in range(len(target_shapes) - 1):
            c_in, l_in = target_shapes[i]
            c_out, l_out = target_shapes[i + 1]
            out_padding = calc_output_padding_1d(
                length=l_in,
                target=l_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            block = [
                nn.ConvTranspose1d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=out_padding,
                    bias=False,
                ),
                nn.BatchNorm1d(c_out),
            ]
            if i < len(target_shapes) - 2:
                block.append(nn.LeakyReLU())
                block.append(nn.Dropout(dropout))
            modules.append(nn.Sequential(*block))

        self.decoder = nn.Sequential(*modules)
        self.logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, hidden, **kwargs):
        y = self.decoder(hidden)
        y = y.permute(0, 2, 1)
        acts_logits, durations = torch.split(
            y, [self.hidden_size - 1, 1], dim=-1
        )
        acts_log_probs = self.logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        durations = torch.log(durations)
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)

        return log_prob_outputs
