from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base
from caveat.models.embed import CustomDurationEmbeddingConcat


class CVAEContLSTMNudgerAdversarial(LightningModule):
    def __init__(
        self,
        in_shape: tuple,
        encodings: int,
        encoding_weights: Optional[Tensor] = None,
        labels_size: Optional[tuple] = None,
        sos: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        latent_dim = kwargs.get("latent_dim", 6)
        hidden_size = kwargs.get("hidden_size", 256)
        self.LR = kwargs.get("LR", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.kld_weight = kwargs.get("kld_weight", 0.0001)
        self.duration_weight = kwargs.get("duration_weight", 1.0)
        self.adv_weight = kwargs.get("adv_weight", 1.0)
        print(f"KLD weight: {self.kld_weight}")
        print(f"duration weight: {self.duration_weight}")
        print(f"adversarial weight: {self.adv_weight}")

        self.generator = CVAESeqLSTMNudger(
            in_shape, encodings, encoding_weights, labels_size, sos, **kwargs
        )
        self.discriminator = Discriminator(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            output_size=labels_size,
        )

    def adversarial_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Adversarial loss for nudging the latent space."""
        return nn.functional.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        (x, _), (y, y_mask), (labels, _) = batch
        optimizer_g, optimizer_d = self.optimizers()

        self.curr_device = x.device

        # train generator
        self.toggle_optimizer(optimizer_g)
        log_probs, mu, log_var, z = self.generator.forward(
            x, labels=labels, target=y
        )
        losses = self.generator.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=y,
            weights=y_mask,
            kld_weight=self.kld_weight,
            duration_weight=self.duration_weight,
            batch_idx=batch_idx,
        )

        # generator loss
        labels_hat = self.discriminator(z)

        adversarial_loss = self.adversarial_loss(labels_hat, labels).detach()
        losses["adversarial_loss"] = adversarial_loss
        losses["adversarial_weight"] = torch.Tensor([self.adv_weight]).float()
        weighted_adversarial_loss = adversarial_loss * self.adv_weight
        losses["adversarial_loss_weighted"] = weighted_adversarial_loss
        losses["loss"] /= weighted_adversarial_loss

        self.manual_backward(losses["loss"])
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)
        labels_hat = self.discriminator(z.detach())
        d_loss = self.adversarial_loss(labels_hat, labels)
        losses["discriminator_loss"] = d_loss

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.log_dict(
            {key: val.item() for key, val in losses.items()}, sync_dist=True
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), self.LR, weight_decay=self.weight_decay
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            self.LR,
            weight_decay=self.weight_decay,
        )
        return [opt_g, opt_d], []

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        (x, _), (y, y_mask), (labels, _) = batch
        self.curr_device = x.device

        log_probs, mu, log_var, z = self.generator.forward(x, labels=labels)
        val_loss = self.generator.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=y,
            weights=y_mask,
            kld_weight=self.kld_weight,
            duration_weight=self.duration_weight,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def predict_step(self, batch):
        return self.generator.predict_step(batch)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, output_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.activation = nn.Softmax(dim=-1)  # TODO!!

    def forward(self, x):
        x = self.block(x)
        x = self.activation(x)
        return x


class CVAESeqLSTMNudger(Base):
    def __init__(self, *args, **kwargs):
        """
        RNN based encoder and decoder with encoder embedding layer and conditionality.
        Normalises attributes size at decoder to match latent size.
        Adds latent layer to decoder instead of concatenating.
        """
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
            raise UserWarning(
                "ConditionalLSTM requires labels_size, please check you have configures a compatible encoder and labels"
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
            input_size=self.labels_size,
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
        self.fc_labels = nn.Linear(self.labels_size, flat_size_encode)
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_attributes = nn.Linear(self.labels_size, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

        if config.get("share_embed", False):
            self.decoder.embedding.weight = self.encoder.embedding.weight

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
        return [log_prob_y, mu, log_var, z]

    def encode(self, input: Tensor, labels: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        hidden = self.encoder(input)
        labels = self.fc_labels(labels)
        hidden = hidden + labels

        # Split the result into mu and var components
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(
        self, z: Tensor, labels: Tensor, target=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # encode labels
        label_mu, label_var = self.label_network(labels)
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

        return log_probs

    def predict(
        self, z: Tensor, labels: Tensor, device: int, **kwargs
    ) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(device)
        labels = labels.to(device)
        prob_samples = exp(self.decode(z=z, labels=labels, **kwargs))
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
