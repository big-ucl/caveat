from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat.models import Base, CustomDurationEmbeddingConcat


class CVAESeqLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoders with optional conditionalities at encoder, latent and decoder."""
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "ConditionalLSTM requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )
        if self.label_embed_sizes is None:
            raise UserWarning("ConditionalLSTM requires label_embed_sizes")
        if not isinstance(self.label_embed_sizes, list):
            raise UserWarning(
                "ConditionalLSTM requires label_embed_sizes to be a list of label embedding sizes"
            )

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.labels_hidden_size = config.get(
            "labels_hidden_size", self.hidden_size
        )
        print(f"Found label encoder hidden size = {self.labels_hidden_size}")

        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2

        # label encoder
        self.label_encoder = LabelEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.labels_hidden_size,
        )

        # encoder
        encoder_conditionality = config.get("encoder_conditionality", "none")
        if encoder_conditionality == "none":
            print("No encoder conditionality")
            self.encoder = Encoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                dropout=self.dropout,
            )
        elif encoder_conditionality == "hidden":
            print("Using hidden state encoder conditionality")
            self.encoder = HiddenConditionalEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                conditionals_size=self.labels_hidden_size,
                dropout=self.dropout,
            )
        elif encoder_conditionality == "inputs_add":
            print("Using inputs addition encoder conditionality")
            self.encoder = InputsAddConditionalEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                conditionals_size=self.labels_hidden_size,
                max_length=length,
                dropout=self.dropout,
            )
        elif encoder_conditionality == "inputs_concat":
            print("Using inputs concat encoder conditionality")
            self.encoder = InputsConcatConditionalEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                conditionals_size=self.labels_hidden_size,
                max_length=length,
                dropout=self.dropout,
            )
        elif (
            encoder_conditionality == "hidden_and_inputs"
            or encoder_conditionality == "both"
        ):
            print("Using hidden and inputs encoder conditionality")
            self.encoder = HiddenInputsConditionalEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                conditionals_size=self.labels_hidden_size,
                max_length=length,
                dropout=self.dropout,
            )
        else:
            raise ValueError(
                "encoder_conditionality must be either 'none', 'hidden', 'inputs_add/concat', or 'hidden_and_inputs'"
            )

        # encoder to latent
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)

        # latent block (add or concat)
        latent_conditionality = config.get("latent_conditionality", "concat")
        if latent_conditionality == "concat":
            print("Label conditionality is concat")
            self.latent_block = ConcatLatent(
                latent_dim=self.latent_dim,
                conditionals_size=self.labels_hidden_size,
                flat_size_encode=flat_size_encode,
                hidden_layers=self.hidden_n,
                hidden_size=self.hidden_size,
            )
        elif latent_conditionality == "add":
            print("Label conditionality is add")
            self.latent_block = AddLatent(
                conditionals_size=self.labels_hidden_size,
                latent_dim=self.latent_dim,
                flat_size_encode=flat_size_encode,
                hidden_layers=self.hidden_n,
                hidden_size=self.hidden_size,
            )
        else:
            raise ValueError(
                "label_conditionality must be either 'concat' or 'add'"
            )

        # decoder conditionality
        decoder_conditionality = config.get("decoder_conditionality", "none")
        if decoder_conditionality == "none":
            print("Decoder conditionality is 'none'")
            self.decoder = Decoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=length,
                dropout=self.dropout,
                sos=self.sos,
            )
        elif decoder_conditionality == "inputs_add":
            print("Decoder conditionality is 'inputs'")
            self.decoder = InputsAddConditionalDecoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=length,
                conditionals_size=self.labels_hidden_size,
                dropout=self.dropout,
                sos=self.sos,
            )
        elif decoder_conditionality == "inputs_concat":
            print("Decoder conditionality is 'inputs_concat'")
            self.decoder = InputsConcatConditionalDecoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=length,
                conditionals_size=self.labels_hidden_size,
                dropout=self.dropout,
                sos=self.sos,
            )
        else:
            raise ValueError(
                "Decoder conditionality must be 'none'. 'inputs_add' or 'inputs_concat'"
            )

        # share embedding
        if config.get("share_embed", False):
            print("Decoder and Encoder Embedding is shared")
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
        conditionals_hidden = self.label_encoder(conditionals)
        hidden = self.encoder(input, conditionals_hidden)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(
        self, z: Tensor, conditionals: Tensor, target=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            hidden (tensor): Latent space batch [N, latent_dims].
            conditionals (tensor): Conditional labels [N, conditionals_size].
            target (tensor): Target sequence batch [N, steps, acts].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        batch_size = conditionals.shape[0]

        conditionals_hidden = self.label_encoder(conditionals)
        conditioned_z = self.latent_block(z, conditionals_hidden)

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # use teacher forcing
            log_probs = self.decoder(
                batch_size=batch_size,
                hidden=conditioned_z,
                target=target,
                conditionals=conditionals_hidden,
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size,
                hidden=conditioned_z,
                target=None,
                conditionals=conditionals_hidden,
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


class LabelEncoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size):
        """Label Encoder using token embedding.
        Embedding outputs are the same size but use different weights so that they can be different sizes.
        Each embedding is then stacked and summed to give single encoding."""
        super(LabelEncoder, self).__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(s, hidden_size) for s in label_embed_sizes]
        )
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        # self.fc_out = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        x = torch.stack(
            [embed(x[:, i]) for i, embed in enumerate(self.embeds)], dim=-1
        ).sum(dim=-1)
        x = self.fc(x)
        x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder without conditionality.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, conditionals):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class HiddenConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        conditionals_size: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder with label conditionality added at RNN hidden state.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            conditionals_size (int): size of conditionals.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(HiddenConditionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        flat_size = 2 * hidden_layers * hidden_size

        self.conditionals_ff = nn.Sequential(
            nn.Linear(conditionals_size, flat_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, conditionals):
        # label conditionality
        h1, h2 = (
            self.conditionals_ff(conditionals)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
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


class InputsAddConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        conditionals_size: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """LSTM Conditional Encoder. Labels are introduced at the input by addition.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            conditionals_size (int): size of conditionals.
            max_length (int): max length of sequences.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(InputsAddConditionalEncoder, self).__init__()
        self.max_length = max_length

        self.inputs_ff = nn.Sequential(
            nn.Linear(conditionals_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, conditionals):
        conditionals = (
            self.inputs_ff(conditionals)
            .unsqueeze(1)
            .repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = embedded + conditionals
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class InputsConcatConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        conditionals_size: int,
        max_length: int,
        dropout: float = 0.1,
        conditional_hidden_size: Optional[int] = None,
    ):
        """LSTM Conditional Encoder. Labels are introduced at the input by concatenation.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            conditionals_size (int): size of conditionals.
            max_length (int): max length of sequences.
            dropout (float): dropout. Defaults to 0.1.
        """
        super().__init__()
        self.max_length = max_length

        if conditional_hidden_size is None:
            conditional_hidden_size = int(hidden_size / 2)
        else:
            conditional_hidden_size = conditional_hidden_size
        encoding_size = hidden_size - conditional_hidden_size
        if encoding_size < 0:
            raise ValueError(
                "conditional_hidden_size must be less than or equal to hidden_size"
            )

        self.inputs_ff = nn.Sequential(
            nn.Linear(conditionals_size, conditional_hidden_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, encoding_size, dropout=dropout
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, conditionals):
        conditionals = (
            self.inputs_ff(conditionals)
            .unsqueeze(1)
            .repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = torch.cat((embedded, conditionals), dim=-1)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class HiddenInputsConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        conditionals_size: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """LSTM Conditional Encoder. Labels are introduced at the hidden state and input.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            conditionals_size (int): size of conditionals.
            flat_size_encode (int): flattened hidden size.
            max_length (int): max length of sequences.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(HiddenInputsConditionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.max_length = max_length
        flat_size = 2 * hidden_layers * hidden_size

        self.inputs_ff = nn.Sequential(
            nn.Linear(conditionals_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.conditionals_ff = nn.Sequential(
            nn.Linear(conditionals_size, flat_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, conditionals):
        h1, h2 = (
            self.conditionals_ff(conditionals)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()

        inputs_conditionals = (
            self.inputs_ff(conditionals)
            .unsqueeze(1)
            .repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = embedded + inputs_conditionals
        _, (h1, h2) = self.lstm(embedded, (h1, h2))
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class ConcatLatent(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        conditionals_size: int,
        flat_size_encode: int,
        hidden_layers: int,
        hidden_size: int,
    ):
        super(ConcatLatent, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.latent_fc = nn.Linear(
            latent_dim + conditionals_size, flat_size_encode
        )

    def forward(self, z: Tensor, conditionals: Tensor) -> Tuple[Tensor, Tensor]:

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
        return hidden


class AddLatent(nn.Module):
    def __init__(
        self,
        conditionals_size: int,
        latent_dim: int,
        flat_size_encode: int,
        hidden_layers: int,
        hidden_size: int,
    ):
        super(AddLatent, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.conditionals_fc = nn.Linear(conditionals_size, latent_dim)
        self.latent_fc = nn.Linear(latent_dim, flat_size_encode)

    def forward(self, z: Tensor, conditionals: Tensor) -> Tuple[Tensor, Tensor]:

        # add conditionlity to z
        conditionals_z = self.conditionals_fc(conditionals)
        z = z + conditionals_z
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
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, conditionals, target=None, **kwargs):
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


class InputsAddConditionalDecoder(Decoder):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
        conditionals_size,
        dropout=0,
        sos=0,
    ):
        super().__init__(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            max_length,
            dropout,
            sos,
        )
        self.inputs_ff = nn.Sequential(
            nn.Linear(conditionals_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch_size, hidden, conditionals, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        inputs_conditionals = self.inputs_ff(conditionals).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, inputs_conditionals
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

    def forward_step(self, x, hidden, conditionals):
        # [N, 1, 2]
        embedded = self.embedding(x) + conditionals
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        # [N, 1, encodings+1]
        return prediction, hidden


class InputsConcatConditionalDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        max_length,
        conditionals_size,
        dropout=0,
        sos=0,
        conditional_hidden_size: Optional[int] = None,
    ):
        """LSTM Decoder with teacher forcing and label injection at step input via concatenation.
        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
        """
        super().__init__()
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos

        if conditional_hidden_size is None:
            conditional_hidden_size = int(hidden_size / 2)
        else:
            conditional_hidden_size = conditional_hidden_size
        encoding_size = hidden_size - conditional_hidden_size
        if encoding_size < 0:
            raise ValueError(
                "conditional_hidden_size must be less than or equal to hidden_size"
            )

        self.embedding = CustomDurationEmbeddingConcat(
            input_size, encoding_size, dropout=dropout
        )
        self.inputs_ff = nn.Sequential(
            nn.Linear(conditionals_size, conditional_hidden_size),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
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

    def forward(self, batch_size, hidden, conditionals, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        inputs_conditionals = self.inputs_ff(conditionals).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, inputs_conditionals
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

    def forward_step(self, x, hidden, conditionals):
        # [N, 1, 2]
        embedded = self.embedding(x)
        embedded = torch.cat((embedded, conditionals), dim=-1)
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
