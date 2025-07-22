from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from caveat import current_device
from caveat.models import Base, utils
from caveat.models.embed import CustomDurationEmbeddingConcat


class VAEContLSTMCountdown(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
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
        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

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
        h = self.fc_hidden(z)

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
            log_probs = utils.normalise_durations(
                log_probs, sos=self.sos, eos=self.eos
            )

        return log_probs


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
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


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
        dropout: float = 0.0,
        sos: int = 0,
        eos: int = 1,
        head_depth: int = 2,
        head_hidden_size: Optional[int] = None,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            output_size (int): lstm output size
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
            sos (int): start of sequence token. Defaults to 0.
            eos (int): end of sequence token. Defaults to 1.
            head_depth (int): number of hidden layers in the linear head. Defaults to 2.
            head_hidden_size (Optional[int]): hidden size of the linear head. Defaults to None.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos
        self.eos = eos

        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.budget_fc = nn.Linear(1, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.act_fc = LinearHead(
            input_size=hidden_size,
            hidden_size=head_hidden_size,
            output_size=output_size,  # exclude duration
            depth=head_depth,
            dropout=0,
        )
        self.duration_fc = LinearHead(
            input_size=hidden_size,
            hidden_size=head_hidden_size,
            output_size=1,  # single duration output
            depth=head_depth,
            dropout=0,
        )
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        budget = torch.ones(batch_size, 1, 1, device=hidden.device)
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, budget
            )
            outputs.append(decoder_output.squeeze(-2))

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

            # remove durations from budget
            budget = (budget - decoder_input[:, :, 1:]).detach()
            budget = budget.clamp(
                min=1e-6
            )  # ensure budget is not negative and > 0

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        log_prob_outputs = torch.log(outputs)
        # TODO: remove logs, includes utils, losses, etc

        return log_prob_outputs

    def forward_step(self, x, hidden, budget):
        # [N, 1, 2]
        embedded = self.embedding(x)
        # embedded_budget = self.budget_fc(budget)
        # embedded = embedded + budget

        # add budget to hidden and cell state
        # hidden = (
        #     hidden[0] + budget.permute(1, 0, 2),
        #     hidden[1] + budget.permute(1, 0, 2),
        # )  # add budget to cell state

        output, hidden = self.lstm(embedded, hidden)

        # output = output + embedded_budget  # add budget to output

        act_prediction = self.act_fc(output)
        act_probs = self.activity_prob_activation(act_prediction)

        durations = self.duration_fc(output)
        durations = self.duration_activation(durations)
        durations = durations * budget

        prediction = torch.cat((act_probs, durations), dim=-1)
        # [N, 1, encodings+1]
        return prediction, hidden

    def pack(self, x):
        # [N, 1, encodings+1]
        acts, duration = torch.split(x, [self.output_size, 1], dim=-1)
        _, topi = acts.topk(1)
        act = (
            topi.squeeze(-1).detach().unsqueeze(-1)
        )  # detach from history as input
        # duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs


class LinearHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int = 1,
        dropout: float = 0.0,
        norm: bool = False,
    ):
        """Linear head for VAE decoder.

        Args:
            input_size (int): input size.
            hidden_size (int): hidden layer size.
            output_size (int): output size.
            depth (int): number of hidden layers.
            dropout (float): dropout probability.
        """
        super(LinearHead, self).__init__()
        if depth == 0:
            raise ValueError("Depth must be greater than 0")
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError(
                "Input, hidden, and output sizes must be positive integers"
            )
        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")

        if depth == 1:
            layers = [nn.Linear(input_size, output_size)]
        else:
            layers = []
            in_features = input_size
            for _ in range(depth):
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.LeakyReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if norm:
                    layers.append(nn.LayerNorm(hidden_size))
                in_features = hidden_size
            layers.append(nn.Linear(hidden_size, output_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear head.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor with activity probabilities.
        """
        return self.block(x)
