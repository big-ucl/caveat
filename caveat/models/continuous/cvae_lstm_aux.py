from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, exp, nn

from caveat.models import Base, utils
from caveat.models.embed import CustomDurationEmbeddingConcat


class CVAEContLSTMAux(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoders with optional conditionalities at encoder, latent and decoder."""
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
            raise UserWarning(
                "ConditionalLSTM requires labels_size, please check you have configures a compatible encoder and condition attributes"
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
        self.length, _ = self.in_shape

        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        self.flat_size_encode = self.hidden_n * self.hidden_size * 2

        # label encoder
        self.label_encoder = self.build_label_encoder()

        # initial encoder hidden state
        self.encoder_hidden = self.build_encoder_hidden(config)

        # step encoder hidden state
        self.encoder = self.build_encoder(config)

        # encoder to latent
        self.fc_mu = nn.Linear(self.flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(self.flat_size_encode, self.latent_dim)

        # latent block (add or concat)
        self.latent_block = self.build_latent_block(config)

        # decoder conditionality
        self.decoder = self.build_decoder(config)

        # auxilary head
        self.aux_head = Aux(
            seq_length=self.length,
            seq_size=self.encodings + 1,
            hidden_size=self.hidden_size,
            attribute_embed_sizes=self.label_embed_sizes,
        )

        # share embedding
        if config.get("share_embed", False):
            print("Decoder and Encoder Embedding is shared")
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def build_label_encoder(self):
        return LabelEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.labels_hidden_size,
        )

    def build_encoder_hidden(self, config):
        encoder_hidden = config.get("hidden_conditionality", "none")
        if encoder_hidden == "none":
            print("No encoder hidden state")
            return HiddenNone()

        if encoder_hidden in ["add", "concat", "hidden"]:
            print("Using label encoder hidden state")
            return HiddenLabel(
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                labels_size=self.labels_hidden_size,
                dropout=self.dropout,
                activation=config.get("hidden_activation", False),
            )
        raise ValueError(
            f"encoder_hidden ({encoder_hidden}) must be either 'none' or 'add/concat/hidden'"
        )

    def build_encoder(self, config):
        # encoder
        encoder_conditionality = config.get("encoder_conditionality", "none")
        if encoder_conditionality == "none":
            print("No encoder conditionality")
            return Encoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                dropout=self.dropout,
            )
        if encoder_conditionality == "add":
            print("Using inputs addition encoder conditionality")
            return AddEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                labels_size=self.labels_hidden_size,
                max_length=self.length,
                dropout=self.dropout,
            )
        elif encoder_conditionality == "concat":
            print("Using inputs concat encoder conditionality")
            return ConcatEncoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_n,
                labels_size=self.labels_hidden_size,
                max_length=self.length,
                dropout=self.dropout,
            )
        raise ValueError(
            f"encoder_conditionality ({encoder_conditionality}) must be either 'none', 'hidden', 'add', or 'concat'"
        )

    def build_latent_block(self, config):
        latent_conditionality = config.get("latent_conditionality", "concat")
        if latent_conditionality == "none":
            print("No latent conditionality")
            return Latent(
                latent_dim=self.latent_dim,
                flat_size_encode=self.flat_size_encode,
                hidden_n=self.hidden_n,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        if latent_conditionality == "concat":
            print("Label conditionality is concat")
            return ConcatLatent(
                latent_dim=self.latent_dim,
                labels_size=self.labels_hidden_size,
                flat_size_encode=self.flat_size_encode,
                hidden_n=self.hidden_n,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        if latent_conditionality == "add":
            print("Label conditionality is add")
            return AddLatent(
                labels_size=self.labels_hidden_size,
                latent_dim=self.latent_dim,
                flat_size_encode=self.flat_size_encode,
                hidden_n=self.hidden_n,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        if latent_conditionality == "film":
            print("Label conditionality is FILM")
            return FilmLatent(
                labels_size=self.labels_hidden_size,
                latent_dim=self.latent_dim,
                flat_size_encode=self.flat_size_encode,
                hidden_n=self.hidden_n,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        raise ValueError(
            "label_conditionality must be either 'concat' or 'add'"
        )

    def build_decoder(self, config):
        decoder_conditionality = config.get("decoder_conditionality", "none")
        if decoder_conditionality == "none":
            print("Decoder conditionality is 'none'")
            return Decoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=self.length,
                dropout=self.dropout,
                sos=self.sos,
            )
        elif decoder_conditionality in {"add", "inputs_add"}:
            print("Decoder conditionality is 'inputs'")
            return InputsAddConditionalDecoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=self.length,
                labels_size=self.labels_hidden_size,
                dropout=self.dropout,
                sos=self.sos,
            )
        elif decoder_conditionality in {"concat", "inputs_concat"}:
            print("Decoder conditionality is 'inputs_concat'")
            return InputsConcatConditionalDecoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings + 1,
                num_layers=self.hidden_n,
                max_length=self.length,
                labels_size=self.labels_hidden_size,
                dropout=self.dropout,
                sos=self.sos,
            )
        raise ValueError(
            "Decoder conditionality must be 'none', 'add/inputs_add' or 'concat/inputs_concat'"
        )

    def forward(
        self, x: Tensor, labels: Optional[Tensor] = None, target=None, **kwargs
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)

        log_prob_y = self.decode(z, labels=labels, target=target)
        p_labels = self.aux_head(log_prob_y)
        return [log_prob_y, (mu, p_labels), log_var, z]

    def loss_function(
        self,
        log_probs,
        mu,
        log_var,
        target,
        weights: Tuple[Tensor, Tensor],
        labels: Tensor,
        label_weights: Optional[Tuple[Tensor, Tensor]] = (None, None),
        **kwargs,
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        mu, p_labels = mu

        # unpack act probs and durations
        target_acts, target_durs = self.unpack_encoding(target)
        pred_acts, pred_durs = self.unpack_encoding(log_probs)
        pred_durs = torch.exp(pred_durs)

        act_weights, seq_weights = weights
        _, joint_weights = label_weights
        dur_weights = utils.duration_mask(act_weights)
        # dur_weights = seq_weights  # use seq_weights as dur_weights

        # activity loss
        act_weight = self.activity_loss_weight * self.scheduled_act_weight
        act_recon = self.act_seq_loss(
            preds=pred_acts,
            targets=target_acts,
            weights=act_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_act_recon = act_weight * act_recon

        # duration loss
        dur_weight = self.duration_loss_weight * self.scheduled_dur_weight
        dur_recon = self.dur_mse_loss(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_dur_recon = dur_weight * dur_recon

        # start time loss
        start_weight = self.start_loss_weight * self.scheduled_start_weight
        start_recon = self.start_seq_loss_detached(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_start_recon = start_weight * start_recon

        # end time loss
        end_weight = self.end_loss_weight * self.scheduled_end_weight
        end_recon = self.end_seq_loss_detached(
            preds=pred_durs,
            targets=target_durs,
            weights=dur_weights,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_end_recon = end_weight * end_recon

        # total_duration loss
        total_dur_weight = (
            self.total_duration_loss_weight * self.scheduled_total_dur_weight
        )
        total_dur_recon = self.total_duration_loss(
            preds=pred_durs,
            targets=target_durs,
            seq_weights=seq_weights,
            joint_weights=joint_weights,
        )
        w_total_dur_recon = total_dur_weight * total_dur_recon

        # reconstruction loss
        w_recons_loss = (
            w_act_recon
            + w_dur_recon
            + w_start_recon
            + w_end_recon
            + w_total_dur_recon
        )

        # kld loss
        kld_loss = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * kld_loss

        # labels loss
        logs = {}
        attribute_loss = 0
        for i, p in enumerate(p_labels):
            target = labels[:, i].long()
            aux_loss = F.cross_entropy(p, target)
            logs[f"label_loss_{i}"] = aux_loss
            attribute_loss += aux_loss
        # attribute_loss = attribute_loss / len(log_probs_ys)
        scheduled_label_weight = (
            self.scheduled_label_weight * self.label_loss_weight
        )
        w_label_loss = scheduled_label_weight * attribute_loss * 100

        # final loss
        loss = w_recons_loss + w_kld_loss + w_label_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "label_loss": w_label_loss.detach(),
        }

    def encode(self, input: Tensor, labels: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        encoded_labels = self.label_encoder(labels)
        hidden = self.encoder_hidden(encoded_labels)
        hidden = self.encoder(input, encoded_labels, hidden)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(
        self, z: Tensor, labels: Tensor, target=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (Tensor): Latent space batch [N, latent_dims].
            labels (Tensor): Conditional labels [N, labels_size].
            target (Tensor): Target sequence batch [N, steps, acts].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        batch_size = labels.shape[0]

        labels_hidden = self.label_encoder(labels)
        conditioned_z = self.latent_block(z, labels_hidden)

        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            # use teacher forcing
            log_probs = self.decoder(
                batch_size=batch_size,
                hidden=conditioned_z,
                target=target,
                labels=labels_hidden,
            )
        else:
            log_probs = self.decoder(
                batch_size=batch_size,
                hidden=conditioned_z,
                target=None,
                labels=labels_hidden,
            )

        return log_probs

    def predict(
        self, z: Tensor, labels: Tensor, device: int, **kwargs
    ) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (Tensor): Latent space batch [N, latent_dims].
            labels (Tensor): Conditional labels [N, labels_size].
            device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        labels = labels.to(device)
        prob_samples = exp(self.decode(z=z, labels=labels, **kwargs))
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

    def forward(self, x):
        x = torch.stack(
            [embed(x[:, i]) for i, embed in enumerate(self.embeds)], dim=-1
        ).sum(dim=-1)
        return x


class HiddenNone(nn.Module):
    def forward(self, labels):
        return None


class HiddenLabel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        labels_size: int,
        dropout: float = 0.1,
        activation: bool = True,
    ):
        super(HiddenLabel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        flat_size = 2 * hidden_layers * hidden_size
        self.ff = nn.Sequential(nn.Linear(labels_size, flat_size))

    def forward(self, labels):
        h1, h2 = (
            self.ff(labels)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        return (h1, h2)


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

    def forward(self, x, labels, hidden):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded, hidden)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class AddEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        labels_size: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """LSTM Conditional Encoder. Labels are introduced at the input by addition.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            labels_size (int): size of labels.
            max_length (int): max length of sequences.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(AddEncoder, self).__init__()
        self.max_length = max_length

        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
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

    def forward(self, x, labels, hidden=None):
        labels = (
            self.labels_ff(labels).unsqueeze(1).repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = embedded + labels
        _, (h1, h2) = self.lstm(embedded, hidden)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class ConcatEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_layers: int,
        labels_size: int,
        max_length: int,
        dropout: float = 0.1,
        conditional_hidden_size: Optional[int] = None,
    ):
        """LSTM Conditional Encoder. Labels are introduced at the input by concatenation.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            hidden_layers (int): number of lstm layers.
            labels_size (int): size of labels.
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

        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, conditional_hidden_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
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

    def forward(self, x, labels, hidden=None):
        labels = (
            self.labels_ff(labels).unsqueeze(1).repeat(1, self.max_length, 1)
        )
        embedded = self.embedding(x)
        embedded = torch.cat((embedded, labels), dim=-1)
        _, (h1, h2) = self.lstm(embedded, hidden)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class Latent(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        flat_size_encode: int,
        hidden_n: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        """Latent block for CVAE.

        Args:
            latent_dim (int): Latent dimension.
            flat_size_encode (int): Flattened size of the encoder output.
        """
        super(Latent, self).__init__()
        self.hidden_n = hidden_n
        self.hidden_size = hidden_size
        self.latent_ff = nn.Linear(latent_dim, flat_size_encode)

    def forward(self, z: Tensor, args) -> Tuple[Tensor, Tensor]:
        z = self.latent_ff(z)
        z = z.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        z = z.split(self.hidden_n)  # ([hidden, N, layers, [hidden, N, layers]])
        return z


class ConcatLatent(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        labels_size: int,
        flat_size_encode: int,
        hidden_n: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super(ConcatLatent, self).__init__()
        self.hidden_n = hidden_n
        self.hidden_size = hidden_size
        flat_size_encode_a = int(flat_size_encode / 2)
        flat_size_encode_b = flat_size_encode - flat_size_encode_a
        self.latent_ff = nn.Sequential(
            nn.Linear(latent_dim, flat_size_encode_a),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )
        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, flat_size_encode_b),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Concatenate labels to latent vector and initialize hidden state."""
        # resize
        z = self.latent_ff(z)
        labels = self.labels_ff(labels)
        # add conditionlity to z
        h = torch.cat((z, labels), dim=-1)
        # initialize hidden state
        hidden = h.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])
        return hidden


class AddLatent(nn.Module):
    def __init__(
        self,
        labels_size: int,
        latent_dim: int,
        flat_size_encode: int,
        hidden_n: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super(AddLatent, self).__init__()
        self.hidden_n = hidden_n
        self.hidden_size = hidden_size
        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, flat_size_encode),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )
        self.latent_ff = nn.Linear(latent_dim, flat_size_encode)

    def forward(self, z: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        # resize
        z_hidden = self.latent_ff(z)
        labels_hidden = self.labels_ff(labels)
        # add conditionlity to z
        h = z_hidden + labels_hidden
        # initialize hidden state
        hidden = h.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])
        return hidden


class FilmLatent(nn.Module):
    def __init__(
        self,
        labels_size: int,
        latent_dim: int,
        flat_size_encode: int,
        hidden_n: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super(FilmLatent, self).__init__()
        self.hidden_n = hidden_n
        self.hidden_size = hidden_size
        self.gamma_ff = nn.Linear(labels_size, flat_size_encode)
        self.beta_ff = nn.Linear(labels_size, flat_size_encode)
        self.latent_ff = nn.Linear(latent_dim, flat_size_encode)

    def forward(self, z: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        z_hidden = self.latent_ff(z)
        labels_beta = self.beta_ff(labels)
        labels_gamma = self.gamma_ff(labels)
        # add conditionlity to z
        h = labels_gamma * z_hidden + labels_beta
        # initialize hidden state
        hidden = h.unflatten(1, (2 * self.hidden_n, self.hidden_size)).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.hidden_n
        )  # ([hidden, N, layers, [hidden, N, layers]])
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers,
        max_length,
        dropout: float = 0.0,
        sos: int = 0,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            output_size (int): lstm output size.
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
        self.outputs_ff = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
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
        prediction = self.outputs_ff(output)
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
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        max_length: int,
        labels_size: int,
        dropout: float = 0,
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
        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        labels_hidden = self.labels_ff(labels).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, labels_hidden
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

    def forward_step(self, x, hidden, labels):
        # [N, 1, 2]
        embedded = self.embedding(x)
        embedded = embedded
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.outputs_ff(output + labels)
        # [N, 1, encodings+1]
        return prediction, hidden


class InputsConcatConditionalDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        max_length: int,
        labels_size: int,
        dropout: float = 0,
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
            conditional_hidden_size = hidden_size
        else:
            conditional_hidden_size = conditional_hidden_size

        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
        self.labels_ff = nn.Sequential(
            nn.Linear(labels_size, conditional_hidden_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output_ff = nn.Sequential(
            nn.Linear(hidden_size + conditional_hidden_size, output_size),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )
        # activations
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        hidden_labels = self.labels_ff(labels).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, hidden_labels
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

    def forward_step(self, x, hidden, labels):
        # [N, 1, 2]
        embedded = self.embedding(x)
        # embedded = torch.cat((embedded, labels), dim=-1)
        output, hidden = self.lstm(embedded, hidden)
        output = torch.cat((output, labels), dim=-1)
        prediction = self.output_ff(output)
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


class Aux(nn.Module):
    def __init__(
        self, seq_length, seq_size, hidden_size, attribute_embed_sizes
    ):
        super(Aux, self).__init__()
        flat_size = seq_length * seq_size
        self.fc = nn.Linear(flat_size, hidden_size)
        self.activation = nn.ReLU()
        self.attribute_nets = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, s), nn.LogSoftmax(dim=-1))
                for s in attribute_embed_sizes
            ]
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.activation(x)
        log_probs = [net(x) for net in self.attribute_nets]
        return log_probs
