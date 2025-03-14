from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models.embed import CustomDurationEmbeddingConcat
from caveat.models.joint_vaes.experiment import JointExperiment


class JVAEContLSTMRerouted(JointExperiment):
    def __init__(self, *args, **kwargs):
        """
        Joint Sequence and Label generating VAE with LSTM sequence encoder and decoder.
        """

        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.label_hidden_size = config.get(
            "label_hidden_size", self.hidden_size
        )
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape
        self.encoder = ScheduleEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )

        self.decoder = ScheduleDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings + 1,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
        )

        self.label_encoder = LabelEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.label_hidden_size,
            latent_size=self.latent_dim,
        )

        self.label_decoder = AttributeDecoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.label_hidden_size,
            latent_size=self.latent_dim,
        )

        self.unflattened_shape = (2 * self.hidden_n, self.hidden_size)
        flat_size_encode = self.hidden_n * self.hidden_size * 2
        self.fc_conditionals = nn.Linear(self.labels_size, flat_size_encode)
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_attributes = nn.Linear(flat_size_encode, self.latent_dim)
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
            conditionals (tensor): Input attributes [N, Ain].

        Returns:
            list: [Log probs x, [Log probs y], mu, var].
        """
        # schedule encode
        mu, log_var = self.encode(x, conditionals=conditionals)
        z = self.reparameterize(mu, log_var)
        log_prob_x, log_prob_y = self.decode(
            z, conditionals=conditionals, target=target
        )
        return [(log_prob_x, log_prob_y), mu, log_var, z]

    def loss_function(
        self,
        log_probs: Tuple[Tensor, Tensor],
        mu: Tensor,
        log_var: Tensor,
        targets: Tuple[Tensor, Tensor],
        masks: Tuple[Tensor, Tensor],
        **kwargs,
    ) -> dict:
        """Calculate the loss function for the model.

        Args:
            log_probs ((tensor, tensor)): Log probabilities for the output sequence.
            mu (tensor): Mean of the latent space.
            log_var (tensor): Log variance of the latent space.

        Returns:
            dict: Loss dictionary.
        """
        # unpack inputs
        log_probs_x, log_probs_ys = log_probs
        target_x, target_y = targets
        mask_x, mask_y = masks

        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target_x)
        pred_acts, pred_durations = self.unpack_encoding(log_probs_x)
        pred_durations = exp(pred_durations)

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        recon_act_nlll = (recon_act_nlll * mask_x.view(-1)).sum() / mask_x.sum()
        scheduled_act_weight = (
            self.scheduled_act_weight * self.activity_loss_weight
        )
        w_act_recon = scheduled_act_weight * recon_act_nlll

        # duration loss
        recon_dur_mse = self.MSE(pred_durations, target_durations)
        recon_dur_mse = (recon_dur_mse * mask_x).sum() / mask_x.sum()
        scheduled_dur_weight = (
            self.duration_loss_weight * self.scheduled_dur_weight
        )
        w_dur_recon = scheduled_dur_weight * recon_dur_mse

        # TODO: could combine above to only apply mask once

        # schedule reconstruction loss
        w_schedule_recons_loss = w_act_recon + w_dur_recon

        # attributes loss
        attribute_loss = 0
        for i, y in enumerate(log_probs_ys):
            target = target_y[:, i].long()
            weight = mask_y[:, i].long()
            nll = self.base_NLLL(y, target)
            weighted_nll = nll * weight
            attribute_loss += weighted_nll.sum()
        attribute_loss = attribute_loss / len(log_probs_ys)
        scheduled_label_weight = (
            self.scheduled_label_weight * self.label_loss_weight
        )
        w_label_loss = scheduled_label_weight * attribute_loss

        # recon loss
        w_recons_loss = w_schedule_recons_loss + w_label_loss

        # kld loss
        kld_loss = self.kld(mu, log_var)
        scheduled_kld_weight = self.kld_loss_weight * self.scheduled_kld_weight
        w_kld_loss = scheduled_kld_weight * kld_loss

        # final loss
        loss = w_recons_loss + w_kld_loss

        return {
            "loss": loss,
            "KLD": w_kld_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "label_recon": w_label_loss.detach(),
            "kld_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([scheduled_act_weight]).float(),
            "dur_weight": torch.tensor([scheduled_dur_weight]).float(),
            "label_weight": torch.tensor([scheduled_label_weight]).float(),
        }

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Re-parameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (tensor): Mean of the latent Gaussian [N x latent_dims].
            logvar (tensor): Standard deviation of the latent Gaussian [N x latent_dims].

        Returns:
            tensor: [N x latent_dims].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def kld(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # from https://kvfrans.com/deriving-the-kl/
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        log_probs_x, log_probs_y = self.decode(z=z, **kwargs)
        prob_x = exp(log_probs_x)
        probs_y = [exp(lpy) for lpy in log_probs_y]
        return prob_x, probs_y

    def infer(self, x: Tensor, device: int, **kwargs) -> Tensor:
        """Given an encoder input, return reconstructed output and z samples.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            (tensor: [N, steps, acts], tensor: [N, latent_dims]).
        """
        (log_prob_x, log_probs_y), _, _, z = self.forward(x, **kwargs)
        prob_x = exp(log_prob_x).to(device)
        probs_y = [exp(lpy) for lpy in log_probs_y]
        z = z.to(device)
        return prob_x, probs_y, z

    def encode(self, input: Tensor, conditionals: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # schedule encode
        hidden = self.encoder(input)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        # attributes encode
        mu_label, log_var_label = self.label_encoder(conditionals)
        # combine encodings
        mu += mu_label
        log_var += log_var_label

        return [mu, log_var]

    def decode(
        self, z: Tensor, target=None, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            log_probs_x, hidden_x = self.decoder(
                batch_size=batch_size, hidden=hidden, target=target
            )
        else:
            log_probs_x, hidden_x = self.decoder(
                batch_size=batch_size, hidden=hidden, target=None
            )

        hidden_x = torch.cat(hidden_x).permute(1, 0, 2).flatten(start_dim=1)
        z_x = self.fc_attributes(hidden_x)

        # decode labels
        log_probs_ys = self.label_decoder(z_x.detach())

        return log_probs_x, log_probs_ys

    def unpack_encoding(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Split the input into activity and duration.

        Args:
            input (tensor): Input sequences [N, steps, acts].

        Returns:
            tuple[tensor, tensor]: [activity [N, steps, acts], duration [N, steps, 1]].
        """
        acts = input[:, :, :-1].contiguous()
        durations = input[:, :, -1:].squeeze(-1).contiguous()
        return acts, durations

    def pack_encoding(self, acts: Tensor, durations: Tensor) -> Tensor:
        """Pack the activity and duration into input.

        Args:
            acts (tensor): Activity [N, steps, acts].
            durations (tensor): Duration [N, steps, 1].

        Returns:
            tensor: Input sequences [N, steps, acts].
        """
        if len(durations.shape) == 2:
            durations = durations.unsqueeze(-1)
        return torch.cat((acts, durations), dim=-1)


class LabelEncoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size, latent_size):
        """Attribute Encoder using token embedding.
        Embedding outputs are the same size but use different weights so that they can be different sizes.
        Each embedding is then stacked and summed to give single encoding."""
        super(LabelEncoder, self).__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(s, hidden_size) for s in label_embed_sizes]
        )
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = torch.stack(
            [embed(x[:, i]) for i, embed in enumerate(self.embeds)], dim=-1
        ).sum(dim=-1)
        x = self.fc(x)
        x = self.activation(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class AttributeDecoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size, latent_size):
        super(AttributeDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, hidden_size)
        self.activation = nn.ReLU()
        self.attribute_nets = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.Linear(hidden_size, hidden_size),
                    # nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(hidden_size, s),
                    nn.LogSoftmax(dim=-1),
                )
                for s in label_embed_sizes
            ]
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        log_probs = [net(x) for net in self.attribute_nets]
        return log_probs


class ScheduleEncoder(nn.Module):
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
        super(ScheduleEncoder, self).__init__()
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

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h1, h2) = self.lstm(embedded)
        # ([layers, N, C (output_size)], [layers, N, C (output_size)])
        h1 = self.norm(h1)
        h2 = self.norm(h2)
        hidden = torch.cat((h1, h2)).permute(1, 0, 2).flatten(start_dim=1)
        # [N, flatsize]
        return hidden


class ScheduleDecoder(nn.Module):
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
        super(ScheduleDecoder, self).__init__()
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

        return log_prob_outputs, decoder_hidden

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
