from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base
from caveat.models.embed import CustomDurationModeDistanceEmbedding


class Seq2SeqLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with conditionality."""
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
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
        self.decoder = Decoder(
            act_embeddings=self.act_encodings,
            mode_embeddings=self.mode_encodings,
            hidden_size=self.hidden_size,
            hidden_act_size=self.hidden_act_size,
            hidden_mode_size=self.hidden_mode_size,
            output_size=self.act_encodings + self.mode_encodings + 2,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
        )
        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2
        self.fc_hidden = nn.Linear(
            flat_size_encode + self.labels_size, flat_size_encode
        )

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
        log_probs = self.decode(z, conditionals=conditionals, target=target)
        return log_probs

    def encode(self, input: Tensor) -> Tensor:
        # [N, L, C]
        return self.encoder(input)

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

        return log_probs

    def loss_function(
        self, log_probs: Tensor, target: Tensor, mask: Tensor, **kwargs
    ) -> dict:
        # unpack log_probs
        log_act_probs, durations, log_mode_probs, distances = torch.split(
            log_probs, [self.act_encodings, 1, self.mode_encodings, 1], dim=-1
        )
        # unpack target
        (
            target_acts,
            target_durations,
            target_mode,
            target_distances,
        ) = target.split([1, 1, 1, 1], dim=-1)

        # acts = input[:, :, :-1].contiguous()
        # durations = input[:, :, -1:].squeeze(-1).contiguous()

        # activity loss
        recon_act_nlll = self.base_NLLL(
            log_act_probs.view(-1, self.act_encodings),
            target_acts.contiguous().view(-1).long(),
        )
        recon_act_nlll = (recon_act_nlll * mask.view(-1)).sum() / mask.sum()

        # duration loss
        recon_dur_mse = self.duration_loss_weight * self.MSE(
            durations, target_durations
        )
        recon_dur_mse = (recon_dur_mse.squeeze(-1) * mask).sum() / mask.sum()

        # mode loss
        recon_mode_nlll = self.base_NLLL(
            log_mode_probs.view(-1, self.mode_encodings),
            target_mode.contiguous().view(-1).long(),
        )
        recon_mode_nlll = (recon_mode_nlll * mask.view(-1)).sum() / mask.sum()

        # distance loss
        recon_dist_mse = self.MSE(distances, target_distances)
        recon_dist_mse = (recon_dist_mse.squeeze(-1) * mask).sum() / mask.sum()

        # reconstruction loss
        recons_loss = (
            recon_act_nlll + recon_dur_mse + recon_mode_nlll + recon_dist_mse
        )

        return {
            "loss": recons_loss,
            "recon_loss": recons_loss.detach(),
            "recon_act_loss": recon_act_nlll.detach(),
            "recon_duration_loss": recon_dur_mse.detach(),
            "recon_mode_loss": recon_mode_nlll.detach(),
            "recon_distance_loss": recon_dist_mse.detach(),
        }

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
        prob_samples = exp(self.forward(x=x, conditionals=labels, **kwargs))
        return x, y, labels, prob_samples


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


class Decoder(nn.Module):
    def __init__(
        self,
        act_embeddings: int,
        mode_embeddings: int,
        hidden_size,
        hidden_act_size,
        hidden_mode_size,
        output_size,
        num_layers,
        max_length,
        dropout: float = 0.0,
        sos: int = 0,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            act_embeddings (int): number of activity embeddings.
            mode_embeddings (int): number of mode embeddings.
            hidden_size (int): lstm hidden size.
            hidden_act_size (int): hidden size for activity embeddings.
            hidden_mode_size (int): hidden size for mode embeddings.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.act_embeddings = act_embeddings
        self.mode_embeddings = mode_embeddings
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos
        self.hidden_act_size = hidden_act_size
        self.hidden_mode_size = hidden_mode_size

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
        self.fc = nn.Linear(hidden_size, output_size)
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Softmax(dim=-2)
        self.mode_prob_activation = nn.Softmax(dim=-1)
        self.mode_logprob_activation = nn.LogSoftmax(dim=-1)
        self.distance_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 4, device=hidden.device)
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

        acts_logits, durations, mode_logits, distances = torch.split(
            outputs, [self.act_embeddings, 1, self.mode_embeddings, 1], dim=-1
        )
        act_log_probs = self.activity_logprob_activation(acts_logits)

        durations = self.duration_activation(durations)

        mode_log_probs = self.mode_logprob_activation(mode_logits)

        distances = self.distance_activation(distances)

        log_prob_outputs = torch.cat(
            (act_log_probs, durations, mode_log_probs, distances), dim=-1
        )

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
        act, duration, mode, distance = torch.split(
            x, [self.act_embeddings, 1, self.mode_embeddings, 1], dim=-1
        )
        _, topi = act.topk(1)
        act = (
            topi.squeeze(-1).detach().unsqueeze(-1)
        )  # detach from history as input
        duration = self.duration_activation(duration)
        _, topi = mode.topk(1)
        mode = topi.squeeze(-1).detach().unsqueeze(-1)
        distance = self.distance_activation(distance)
        outputs = torch.cat((act, duration, mode, distance), dim=-1)
        # [N, 1, 4]
        return outputs
