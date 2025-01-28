from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base, CustomDurationEmbedding


class CVAESeqLSTMNudger(Base):
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
        self.encoder = Encoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
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
        self.fc_attributes = nn.Linear(self.conditionals_size, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

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

    def encode(self, input: Tensor, conditionals: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        hidden = self.encoder(input)
        conditionals = self.fc_conditionals(conditionals)
        hidden = hidden + conditionals

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
        # encode labels
        label_mu, label_var = self.label_network(conditionals)
        # manipulate z using label encoding
        z = (z * label_var) + label_mu

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

        return log_probs  # modified z removed fro refactor

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

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


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
        if top_sampler:
            print("Decoder using topk sampling")
            self.sample = self.topk
        else:
            print("Decoder using multinomial sampling")
            self.sample = self.multinomial

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
        act = self.sample(acts)
        duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs

    def multinomial(self, x):
        # [N, 1, encodings]
        acts = torch.multinomial(self.activity_prob_activation(x.squeeze()), 1)
        # DETACH?
        return acts

    def topk(self, x):
        _, topi = x.topk(1)
        act = topi.detach()  # detach from history as input
        # DETACH?
        return act
