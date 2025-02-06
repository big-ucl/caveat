from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base, CustomDurationEmbeddingConcat


class LabelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LabelNetwork, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_var = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class CVAESeqLSTMNudgeFeed(Base):
    def __init__(self, *args, **kwargs):
        """
        RNN based encoder and decoder with encoder embedding layer and conditionality.
        Normalises attributes size at decoder to match latent size.
        Adds latent layer to decoder instead of concatenating.
        """
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "ConditionalLSTM requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape
        self.label_network = LabelNetwork(
            input_size=self.conditionals_size,
            hidden_size=self.hidden_size,
            output_size=self.latent_dim,
        )
        self.encoder = Encoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
        )
        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2
        self.fc_conditionals = nn.Linear(
            self.conditionals_size, flat_size_encode
        )
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)
        self.fc_x = nn.Linear(self.conditionals_size, self.hidden_size)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(
        self,
        x: Tensor,
        conditionals: Optional[Tensor] = None,
        target=None,
        **kwargs,
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x, conditionals)

        z = self.reparameterize(mu, log_var)

        log_prob_y = self.decode(z, conditionals=conditionals, target=target)
        return [log_prob_y, mu, log_var, z]

    def nudge(
        self, conditionals: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        # encode labels
        return self.label_network(conditionals)

    def encode(self, input: Tensor, conditionals: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # encode labels for hidden state
        h1, h2 = (
            self.fc_conditionals(conditionals)
            .unflatten(1, (2 * self.hidden_n, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_n)  # ([hidden, N, layers, [hidden, N, layers]])
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        # [N, L, C]
        hidden = self.encoder(input, (h1, h2))
        # [N, flatsize]

        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        # nudge z using label encoding
        mu_nudge, var_nudge = self.nudge(conditionals)
        mu = mu + mu_nudge
        log_var = log_var + var_nudge

        return [mu, log_var]

    def decode(
        self, z: Tensor, conditionals: Tensor, target=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # un-nudge z using label encoding
        mu_nudge, var_nudge = self.nudge(conditionals)
        std_nudge = torch.exp(0.5 * var_nudge)
        z = (z - mu_nudge) / std_nudge

        # initialize hidden state as inputs
        h = self.fc_hidden(z)
        # initialize step inputs
        x = self.fc_x(conditionals).unsqueeze(-2)

        # initialize hidden state
        hidden = h.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])

        log_probs = self.decoder(hidden=hidden, x=x, target=None)

        # TODO: add forcing back?
        # if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
        #     # use teacher forcing
        #     log_probs, probs = self.decoder(
        #         batch_size=batch_size, hidden=hidden, target=target
        #     )
        # else:
        #     log_probs, probs = self.decoder(
        #         batch_size=batch_size, hidden=hidden, target=None
        #     )

        return log_probs


class CVAESeqLSTMNudgeFeedPre(CVAESeqLSTMNudgeFeed):

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape
        self.encoder = ConditionalEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            conditionals_size=self.conditionals_size,
            max_length=length,
            dropout=self.dropout,
        )
        self.label_network = LabelNetwork(
            input_size=self.conditionals_size,
            hidden_size=self.hidden_size,
            output_size=self.latent_dim,
        )
        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_layers,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
        )
        self.unflattened_shape = (2 * self.hidden_layers, self.hidden_size)
        flat_size_encode = self.hidden_layers * self.hidden_size * 2
        self.fc_conditionals = nn.Linear(
            self.conditionals_size, flat_size_encode
        )
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)
        self.fc_x = nn.Linear(self.conditionals_size, self.hidden_size)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def encode(
        self, input: Tensor, conditionals: Optional[Tensor]
    ) -> list[Tensor]:
        h1, h2 = (
            self.fc_conditionals(conditionals)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        hidden = self.encoder(input, (h1, h2), conditionals)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        # nudge z using label encoding
        mu_nudge, var_nudge = self.nudge(conditionals)
        mu = mu + mu_nudge
        log_var = log_var + var_nudge
        return [mu, log_var]

    def predict(
        self, z: Tensor, conditionals: Tensor, device: int, **kwargs
    ) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        conditionals = conditionals.to(device)
        prob_samples = exp(
            self.decode(z=z, conditionals=conditionals, **kwargs)
        )
        return prob_samples


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
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded, hidden)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        conditionals_size: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(ConditionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conditionals_size, hidden_size)
        self.nl = nn.LeakyReLU()

    def forward(self, x, hidden, conditionals):
        conditionals = (
            self.nl(self.fc(conditionals))
            .unsqueeze(1)
            .repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = embedded + conditionals
        _, (h1, h2) = self.lstm(embedded, hidden)
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
        top_sampler: bool = True,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos

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
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, hidden, x, **kwargs):
        hidden, cell = hidden
        # decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        # decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for _ in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                x, decoder_hidden
            )
            outputs.append(decoder_output.squeeze())

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]

        acts_logits, durations = torch.split(
            outputs, [self.output_size - 1, 1], dim=-1
        )
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = torch.log(self.duration_activation(durations))
        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)

        return log_prob_outputs

    def forward_step(self, x, hidden):
        # [N, 1, 2]
        # embedded = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        prediction = self.fc(output)
        # [N, 1, encodings+1]
        return prediction, hidden
