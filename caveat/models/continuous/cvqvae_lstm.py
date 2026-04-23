from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base, utils
from caveat.models.embed import CustomDurationEmbeddingConcat


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.mse_loss = nn.MSELoss()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, inputs):
        # Compute distances
        inputs_shape = inputs.shape  # [B, L, D]
        inputs_flat = inputs.view(-1, self.D)  # [BL, D]
        distances = (
            torch.sum(inputs_flat**2, dim=1, keepdim=True)
            - 2 * torch.matmul(inputs_flat, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight**2, dim=1)
        )  # [BL, K]

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = nn.functional.one_hot(encoding_indices, self.K).type(
            inputs.dtype
        )
        encoding_indices = encoding_indices.view(inputs_shape[0], -1)  # [B, L]

        # Quantized vectors: [B*K, D]
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(inputs_shape)  # [B, K, D]

        # Losses
        e_latent_loss = self.mse_loss(quantized.detach(), inputs)
        q_latent_loss = self.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class ConditionalPriorRNN(nn.Module):
    def __init__(
        self,
        len_embeddings,
        num_embeddings,
        embedding_dim,
        labels_dim,
        hidden_size=128,
        hidden_layers=2,
    ):
        super().__init__()
        self.L = len_embeddings
        self.K = num_embeddings
        self.D = embedding_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        flat_size = 2 * hidden_layers * hidden_size
        self.hidden_fc = nn.Linear(labels_dim, flat_size)
        self.token_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_size, hidden_layers, batch_first=True
        )
        self.output = nn.Linear(hidden_size, num_embeddings)
        self.activation = nn.LogSoftmax(dim=-1)

    def hidden_init(self, labels):
        h1, h2 = (
            self.hidden_fc(labels)
            .unflatten(1, (2 * self.hidden_layers, self.hidden_size))
            .permute(1, 0, 2)
            .split(self.hidden_layers)
        )
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        return (h1, h2)

    def forward(self, labels, batch_size, device, target=None):
        """
        Autoregressive training.
        Returns logits: [B, H, num_embeddings]
        """
        logprobs = []
        hidden = self.hidden_init(labels)

        token = torch.zeros(
            batch_size, 1, device=device
        ).int()  # start token [B, 1]

        for t in range(self.L):
            logprob, hidden = self.forward_step(
                token, hidden
            )  # [B, num_embeddings], [num_layers, B, hidden_dim]
            logprobs.append(logprob)  # [B, num_embeddings]
            if target is not None:
                token = target[:, t].unsqueeze(1)  # [B, 1]
            else:
                prob = torch.exp(logprob)  # [B, num_embeddings]
                token = torch.multinomial(prob, num_samples=1)  # [B, 1]
        return torch.stack(logprobs, dim=1)  # [B, H, num_embeddings]

    def forward_step(self, token, hidden):
        """
        Autoregressive generation step.
        token: [B, 1]
        hidden: [num_layers, B, hidden_dim]
        Returns logits: [B, num_embeddings], hidden: [num_layers, B, hidden_dim]
        """
        x = self.token_embedding(token)  # [B, embedding_dim]
        out, hidden = self.rnn(
            x, hidden
        )  # [B, 1, hidden_dim], [num_layers, B, hidden_dim]
        logit = self.output(out.squeeze(1))  # [B, num_embeddings]
        logprobs = self.activation(logit)  # [B, num_embeddings]
        return logprobs, hidden

    def generate(self, labels, batch_size, device, **kwargs):
        """
        Autoregressive generation.
        Returns logits: [B, H, num_embeddings]
        """
        samples = []
        hidden = self.hidden_init(labels)
        token = torch.zeros(
            batch_size, 1, device=device
        ).int()  # start token [B, 1]

        for t in range(self.L):
            logprob, hidden = self.forward_step(
                token, hidden
            )  # [B, num_embeddings], [num_layers, B, hidden_dim]
            prob = torch.exp(logprob)  # [B, num_embeddings]
            token = torch.multinomial(prob, num_samples=1)  # [B, 1]
            samples.append(token)
        return torch.stack(samples, dim=1)  # [B, L]


class LabelEncoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size):
        """Label Encoder using token embedding.
        Embedding outputs are the same size but use different weights so that they can be different sizes.
        Each embedding is then stacked and summed to give single encoding."""
        super(LabelEncoder, self).__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(s, hidden_size) for s in label_embed_sizes]
        )
        # self.fc = nn.Linear(hidden_size, hidden_size)
        # self.activation = nn.ReLU()

    def forward(self, x):
        x = torch.stack(
            [embed(x[:, i]) for i, embed in enumerate(self.embeds)], dim=-1
        ).sum(dim=-1)
        return x


class CVQVAEContLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
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
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length, _ = self.in_shape
        self.head_hidden_size = config.get("head_hidden_size", self.hidden_size)
        self.head_depth = config.get("head_depth", 2)
        self.num_embeddings = config.get("num_embeddings", 10)
        self.embedding_dim = config.get("embedding_dim", 64)
        self.labels_hidden_size = config.get(
            "labels_hidden_size", self.hidden_size
        )

        self.label_encoder = LabelEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.labels_hidden_size,
        )

        self.prior = ConditionalPriorRNN(
            len_embeddings=self.latent_dim,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            labels_dim=self.labels_hidden_size,
            hidden_size=self.hidden_size,
            hidden_layers=3,
        )

        self.encoder = Encoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.vq_block = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=config.get("commitment_cost", 0.25),
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
        self.fc_encode = nn.Linear(
            flat_size_encode, self.latent_dim * self.embedding_dim
        )
        self.fc_decode = nn.Linear(
            self.latent_dim * self.embedding_dim, flat_size_encode
        )
        self.prior_loss = nn.NLLLoss()

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
        B = x.size(0)
        labels_hidden = (
            self.label_encoder(labels) if labels is not None else None
        )
        z = self.encode(x, labels_hidden)  # [N, latent]
        q, vq_loss, indices = self.vq_block(z)
        log_probs_x = self.decode(q, labels=labels_hidden, target=target)
        if target is not None and torch.rand(1) < self.teacher_forcing_ratio:
            indices_logits = self.prior(
                labels_hidden, batch_size=B, device=x.device, target=indices
            )  # [B, H-1, num_embeddings]
        else:
            indices_logits = self.prior(
                labels_hidden, batch_size=B, device=x.device, target=None
            )  # [B, H-1, num_embeddings]
        prior_loss = self.prior_loss(
            indices_logits.view(-1, self.num_embeddings), indices.view(-1)
        )  # predict next token
        return [log_probs_x, prior_loss, vq_loss, indices]

    def predict(
        self, z: Tensor, device: int, labels: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        labels_hidden = (
            self.label_encoder(labels) if labels is not None else None
        )
        z = self.prior.generate(
            labels_hidden, batch_size=z.size(0), device=device
        )
        z = self.vq_block.embeddings(z)  # [N, latent_dim, embedding_dim]
        prob_samples = exp(self.decode(z, **kwargs))
        return prob_samples

    def loss_function(
        self,
        log_probs: Tensor,
        mu: Tensor,  # actually indices_loss
        log_var: Tensor,  # actually vq_loss
        target: Tensor,
        weights: Tuple[Tensor, Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]] = (None, None),
        z: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
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

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # prior loss
        # mu is actually indices_loss
        prior_loss = mu

        # regularisation loss
        scheduled_kld_weight = self.beta * self.scheduled_kld_weight
        w_vq_loss = scheduled_kld_weight * log_var

        # final loss
        loss = w_recons_loss + w_vq_loss + prior_loss

        return {
            "loss": loss,
            "vq_loss": w_vq_loss.detach(),
            "prior_loss": prior_loss.detach(),
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "vq_weight": torch.tensor([scheduled_kld_weight]).float(),
            "act_weight": torch.tensor([act_weight]).float(),
            "dur_weight": torch.tensor([dur_weight]).float(),
        }

    def encode(self, input: Tensor, labels: Optional[Tensor]) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # [N, L, C]
        hidden = self.encoder(input)
        hidden = self.fc_encode(hidden)
        return hidden.view(-1, self.latent_dim, self.embedding_dim)

    def decode(self, z: Tensor, target=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        z = z.view(-1, self.latent_dim * self.embedding_dim)
        h = self.fc_decode(z)

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
            log_probs = utils.normalise_log_durations(
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
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            outputs.append(decoder_output.squeeze(-2))

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        log_prob_outputs = torch.log(outputs)
        # TODO: remove logs, includes utils, losses, etc

        return log_prob_outputs

    def forward_step(self, x, hidden):
        # [N, 1, 2]
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)

        act_prediction = self.act_fc(output)
        act_probs = self.activity_prob_activation(act_prediction)

        durations = self.duration_fc(output)
        durations = self.duration_activation(durations)

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
        duration = self.duration_activation(duration)
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
        norm: bool = True,
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
