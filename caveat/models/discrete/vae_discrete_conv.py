from typing import Optional, Tuple, Union

from torch import Tensor, nn

from caveat.models import utils
from caveat.models.base import Base
from caveat.models.utils import calc_output_padding_2d, conv2d_size


class VAEDiscConv(Base):
    def __init__(self, *args, **kwargs):
        """Convolution based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        hidden_layers = list
        latent_dim = int
        dropout = Optional[float]
        kernel_size = Optional[Union[tuple[int, int], int]]
        stride = Optional[Union[tuple[int, int], int]]
        padding = Optional[Union[tuple[int, int], int]]

        embed_size = config.get("embed_size", self.encodings)
        hidden_layers = utils.build_hidden_layers(config)
        latent_dim = config["latent_dim"]
        dropout = config.get("dropout", 0)
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 2)
        padding = config.get("padding", 1)

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_size=self.encodings,
            embed_size=embed_size,
            in_shape=self.in_shape,
            hidden_layers=hidden_layers,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.decoder = Decoder(
            target_shapes=self.encoder.target_shapes,
            hidden_layers=hidden_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.fc_mu = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder.flat_size, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, self.encoder.flat_size)

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

    def loss_function(
        self, log_probs, mu, log_var, target, mask, *args, **kwargs
    ) -> dict:
        return self.discretized_loss(
            log_probs, mu, log_var, target, mask, *args, **kwargs
        )


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embed_size: int,
        in_shape: tuple,
        hidden_layers: list,
        dropout: float = 0.1,
        kernel_size: Union[tuple[int, int], int] = 3,
        stride: Union[tuple[int, int], int] = 2,
        padding: Union[tuple[int, int], int] = 1,
    ):
        """2d Convolutions Encoder.

        Args:
            in_shape (tuple[int, int, int]): [C, time_step, activity_encoding].
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(Encoder, self).__init__()
        h = in_shape[0]
        self.embedding = nn.Embedding(input_size, embed_size)
        w = embed_size
        channels = 1

        modules = []
        self.target_shapes = [(channels, h, w)]

        for hidden_channels in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        # bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(),
                )
            )
            h, w = conv2d_size(
                (h, w), kernel_size=kernel_size, padding=padding, stride=stride
            )
            self.target_shapes.append((hidden_channels, h, w))
            channels = hidden_channels

        self.dropout = nn.Dropout(dropout)

        self.shape_before_flattening = (-1, channels, h, w)
        self.encoder = nn.Sequential(*modules)
        self.flat_size = int(channels * h * w)

    def forward(self, x):
        y = self.dropout(self.embedding(x.int()))
        y = y.unsqueeze(1)  # add channel dim for Conv
        y = self.encoder(y)
        y = y.flatten(start_dim=1)
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        target_shapes,
        hidden_layers: list,
        kernel_size: Union[tuple[int, int], int] = 3,
        stride: Union[tuple[int, int], int] = 2,
        padding: Union[tuple[int, int], int] = 1,
    ):
        """2d Conv Decoder.

        Args:
            target_shapes (list): list of target shapes from encoder.
            hidden_layers (list, optional): _description_. Defaults to None.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(Decoder, self).__init__()
        modules = []
        target_shapes.reverse()

        for i in range(len(hidden_layers) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=target_shapes[i][0],
                        out_channels=target_shapes[i + 1][0],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=calc_output_padding_2d(
                            target_shapes[i + 1]
                        ),
                        # bias=False,
                    ),
                    nn.BatchNorm2d(target_shapes[i + 1][0]),
                    nn.LeakyReLU(),
                )
            )

        # Final layer with Tanh activation
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=target_shapes[-2][0],
                    out_channels=target_shapes[-1][0],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=calc_output_padding_2d(target_shapes[-1]),
                ),
                nn.BatchNorm2d(target_shapes[-1][0]),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)
        self.logprob_activation = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, **kwargs):
        y = self.decoder(hidden)
        y = y.squeeze(1)  # remove conv channel dim
        return self.logprob_activation(y)
