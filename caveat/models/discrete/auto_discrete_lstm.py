from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base


class AutoDiscLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer and conditionality."""
        super().__init__(*args, **kwargs)
        if self.labels_size is None:
            raise UserWarning(
                "ConditionalLSTM requires labels_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = 1  # dummy value for the predict dataloader
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length = self.in_shape[0]
        bidirectional = config.get("bidirectional", False)
        top_sampler = config.get("top_sampler", False)

        # self.unflattened_shape = (2 * self.hidden_layers, self.hidden_size)
        if bidirectional:
            flat_size_encode = self.hidden_n * self.hidden_size * 2 * 2
            self.adjusted_layers = self.hidden_n * 2
            self.unflatten_shape = (2 * 2 * self.hidden_n, self.hidden_size)
        else:
            flat_size_encode = self.hidden_n * self.hidden_size * 2
            self.adjusted_layers = self.hidden_n
            self.unflatten_shape = (2 * self.hidden_n, self.hidden_size)

        self.label_encoder = LabelEncoder(
            label_embed_sizes=self.label_embed_sizes,
            hidden_size=self.hidden_size,
            output_size=flat_size_encode,
        )

        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            top_sampler=top_sampler,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> List[Tensor]:
        log_probs = self.decode(z=x, labels=labels, target=target)
        return [log_probs, Tensor([]), Tensor([]), Tensor([])]

    def loss_function(
        self, log_probs: Tensor, target: Tensor, weights: Tensor, **kwargs
    ) -> dict:
        """Loss function for discretized encoding [N, L]."""
        # activity loss
        recon_act_nlll = self.NLLL(log_probs.squeeze().permute(0, 2, 1), target)

        # loss
        loss = recon_act_nlll

        return {"loss": loss, "recon_loss": recon_act_nlll}

    def encode(self, input: Tensor):
        return None

    def decode(
        self, z: None, labels: Tensor, target: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        h = self.label_encoder(labels)

        # initialize hidden state
        hidden = h.unflatten(1, self.unflatten_shape).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.adjusted_layers
        )  # ([hidden, N, layers, [hidden, N, layers]])
        batch_size = z.shape[0]

        log_probs = self.decoder(
            batch_size=batch_size,
            hidden=hidden,
            target=target,
            teacher_forcing_ratio=(
                self.teacher_forcing_ratio if target is not None else 0.0
            ),
        )

        return log_probs

    def predict(
        self, z: Tensor, labels: Tensor, device: int, **kwargs
    ) -> Tensor:
        z = z.to(device)
        labels = labels.to(device)
        return exp(self.decode(z=z, labels=labels, kwargs=kwargs))


class LabelEncoder(nn.Module):
    def __init__(self, label_embed_sizes, hidden_size, output_size):
        """Optimised Label Encoder using a single combined embedding layer.
        Eliminates the Python for-loop and multiple GPU kernel calls.
        """
        super().__init__()

        # 1. Calculate offsets for each label type
        # If label sizes are, offsets will be [0, 3, 7]
        offsets = [0]
        for s in label_embed_sizes[:-1]:
            offsets.append(offsets[-1] + s)

        # Register as a buffer so it moves to GPU/CPU automatically with the module
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # 2. Create one single combined embedding matrix
        total_vocab_size = sum(label_embed_sizes)
        self.combined_embed = nn.Sequential(
            CastToLong(), NormalizedEmbedding(total_vocab_size, hidden_size)
        )

        # 3. Final feed-forward network remains the same
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x shape: [Batch_Size, Num_Labels]

        # Apply the precomputed offsets to shift the indices of each column
        # self.offsets (shape: [Num_Labels]) broadcasts across the batch dimension
        shifted_x = x + self.offsets

        # Perform exactly ONE embedding lookup
        # Output shape: [Batch_Size, Num_Labels, Hidden_Size]
        embedded = self.combined_embed(shifted_x)

        # Sum across the labels dimension (dim=1)
        # Output shape: [Batch_Size, Hidden_Size]
        x_summed = embedded.sum(dim=1)

        return self.ff(x_summed)


class CastToLong(nn.Module):
    def forward(self, x):
        return x.long()


class NormalizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.orthogonal_(self.embed.weight)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(self.embed(x))


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
        bidirectional: bool = False,
    ):
        """LSTM Decoder with teacher forcing.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
            sos (int): start of sequence token. Defaults to 0.
            top_sampler (bool): top1 sampling. Defaults to False.
            bidirectional (bool): bidirectional lstm. Defaults to False.
        """
        super(Decoder, self).__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            print("Using bidirectional LSTM")
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        if top_sampler:
            print("Using topk sampling")
            self.sample = self.topk
        else:
            print("Using multinomial sampling")
            self.sample = self.multinomial

    def forward(
        self,
        batch_size,
        hidden,
        target=None,
        teacher_forcing_ratio=0.0,
        **kwargs,
    ):
        hidden, cell = hidden
        decoder_input = torch.full(
            (batch_size, 1), self.sos, dtype=torch.long, device=hidden.device
        )
        decoder_hidden = (hidden.contiguous(), cell.contiguous())
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            outputs.append(decoder_output)

            if target is not None and teacher_forcing_ratio > 0:
                sampled = self.sample(decoder_output)  # [N, 1]
                # per-sample decision: [N, 1] bool
                use_teacher = (
                    torch.rand(batch_size, 1, device=hidden.device)
                    < teacher_forcing_ratio
                )
                decoder_input = torch.where(
                    use_teacher, target[:, i : i + 1], sampled
                )
            else:
                decoder_input = self.sample(decoder_output)

        acts_logits = torch.cat(outputs, dim=1)
        return self.activity_logprob_activation(acts_logits)

    def forward_step(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden

    def multinomial(self, x):
        # [N, 1, encodings]
        acts = torch.multinomial(self.activity_prob_activation(x.squeeze()), 1)
        # DETACH?
        return acts

    def topk(self, x):
        _, topi = x.topk(1)
        acts = topi.squeeze(-2).detach()  # detach from history as input
        # DETACH?
        return acts
