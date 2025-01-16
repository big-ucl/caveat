from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base, CustomDurationEmbedding


class CVAESeqLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer and conditionality."""
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "ConditionalLSTM requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.unflattened_shape = (2 * self.hidden_layers, self.hidden_size)
        flat_size_encode = self.hidden_layers * self.hidden_size * 2

        # encoder
        if config.get("encoder_conditionality", False):
            print("Encoder conditionality is True")

            self.encoder = ConditionalEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                num_layers=self.hidden_layers,
                conditionals_size=self.conditionals_size,
                flat_size_encode=flat_size_encode,
                dropout=self.dropout,
            )
        else:
            self.encoder = Encoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                num_layers=self.hidden_layers,
                dropout=self.dropout,
            )

        # encoder to latent
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)

        # latent conditionality
        latent_conditionality = config.get("label_conditionality", "concat")
        if latent_conditionality == "concat":
            print("Label conditionality is concat")
            latent_size = self.latent_dim + self.conditionals_size
        elif latent_conditionality == "add":
            print("Label conditionality is add")
            latent_size = self.latent_dim
            self.labels_fc = nn.Linear(self.conditionals_size, self.latent_dim)
        else:
            raise ValueError(
                "label_conditionality must be either 'concat' or 'add'"
            )

        # latent to decoder hidden
        self.latent_fc = nn.Linear(latent_size, flat_size_encode)

        # decoder conditionality
        decoder_conditionality = config.get("decoder_conditionality", False)
        if decoder_conditionality:
            print("Decoder conditionality is True")
            self.decoder_x_fc = nn.Linear(
                self.conditionals_size, self.hidden_size
            )

        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_layers,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            conditionality=decoder_conditionality,
        )

        # share embedding
        if config.get("share_embed", False):
            print("Embedding is shared")
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

    def encode(self, input: Tensor, conditionals: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        h1, h2 = (
            self.fc_conditionals(conditionals)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(
                self.hidden_layers
            )  # ([hidden, N, layers, [hidden, N, layers]])
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        # [N, L, C]
        hidden = self.encoder(input, (h1, h2))
        # [N, flatsize]

        # Split the result into mu and var components
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

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
        # add conditionlity to z
        z = torch.cat((z, conditionals), dim=-1)
        # initialize hidden state as inputs
        h = self.latent_fc(z)

        # initialize hidden state
        hidden = h.unflatten(
            1, (2 * self.hidden_layers, self.hidden_size)
        ).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_layers
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

        return log_probs

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


class CVAESeqLSTMPre(CVAESeqLSTM):

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
        self.fc_hidden = nn.Linear(
            self.latent_dim + self.conditionals_size, flat_size_encode
        )
        self.fc_x = nn.Linear(self.conditionals_size, self.hidden_size)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def encode(self, input: Tensor, conditionals: Tensor) -> list[Tensor]:
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
        return [mu, log_var]


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
        self.embedding = CustomDurationEmbedding(
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


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        conditionals_size: int,
        flat_size_encode: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder with label conditionality added at input hidden state.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        # self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.flat_size_encode = flat_size_encode

        self.conditionals_fc = nn.Linear(conditionals_size, hidden_size)
        self.embedding = CustomDurationEmbedding(
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

    def forward(self, x, conditionals):
        # label conditionality
        h1, h2 = (
            self.fc_conditionals(conditionals)
            .unflatten(1, self.flat_size_encode)
            .permute(1, 0, 2)
            .split(self.num_layers)
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()

        # input encoding
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded, (h1, h2))
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
        self.embedding = CustomDurationEmbedding(
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

        self.embedding = CustomDurationEmbedding(
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
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            outputs.append(decoder_output.squeeze())

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]  # (slice maintains dim)
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

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
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        # [N, 1, encodings+1]
        return prediction, hidden

    def pack(self, x):
        # [N, 1, encodings+1]
        acts, duration = torch.split(x, [self.output_size - 1, 1], dim=-1)
        _, topi = acts.topk(1)
        act = (
            topi.squeeze(-1).detach().unsqueeze(-1)
        )  # detach from history as input
        duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs
