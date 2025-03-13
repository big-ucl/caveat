import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, exp, nn

from caveat.models import Base
from caveat.models.embed import (
    CustomDurationEmbeddingAddNorm,
    CustomDurationEmbeddingConcat,
)


class VAEContXAtt(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        self.hidden_size = config["hidden_size"]
        self.ffwd_size = config.get("ffwd_size", self.hidden_size)
        self.heads = config["heads"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config.get("dropout", 0.0)
        self.length, _ = self.in_shape
        self.sampling = config.get("sampling", "top")
        self.embedding = config.get("embedding", "concat")
        print(f"Embedding: {self.embedding}")
        self.position_embedding = config.get("position_embedding", "learnt")
        print(f"Positional embedding: {self.position_embedding}")
        self.time_embedding = config.get("time_embedding", "none")
        print(f"Time embedding: {self.time_embedding}")
        self.latent_context = config.get("latent_context", "xattention")
        print(f"Latent context: {self.latent_context}")

        self.encoder = AttentionEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            ffwd_size=self.ffwd_size,
            length=self.length,
            n_head=self.heads,
            n_layer=self.hidden_n,
            dropout=self.dropout,
            embedding=self.embedding,
            position_embedding=self.position_embedding,
            time_embedding=self.time_embedding,
        )
        self.decoder = AttentionDecoder(
            input_size=self.encodings,
            output_size=self.encodings + 1,
            hidden_size=self.hidden_size,
            ffwd_size=self.ffwd_size,
            num_heads=self.heads,
            num_layers=self.hidden_n,
            length=self.length,
            dropout=self.dropout,
            embedding=self.embedding,
            position_embedding=self.position_embedding,
            time_embedding=self.time_embedding,
            latent_context=self.latent_context,
        )
        self.unflattened_shape = (self.length, self.hidden_size)
        flat_size_encode = self.length * self.hidden_size
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

        if config.get("share_embed", False):
            print("Sharing embeddings")
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(
        self, x: Tensor, target=None, input_mask=None, **kwargs
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        # if input_mask is not None:
        #     mask = torch.zeros_like(input_mask)
        #     mask[input_mask > 0] = 1.0
        #     mask = mask[:, None, :]
        # else:
        #     mask = None
        mask = None

        mu, log_var = self.encode(x, conditionals=None, mask=mask)
        z = self.reparameterize(mu, log_var)

        if target is not None:  # training
            log_prob_y = self.decode(z, context=x, mask=mask)
            return [log_prob_y, mu, log_var, z]

        # no target so assume generating
        log_prob = self.predict_sequences(z, current_device=z.device)
        return [log_prob, mu, log_var, z]

    def encode(
        self,
        input: Tensor,
        conditionals: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> list[Tensor]:
        """Encodes the input by passing through the encoder network.

        Args:
            input (tensor): Input sequence batch [N, steps, acts].

        Returns:
            list[tensor]: Latent layer input (means and variances) [N, latent_dims].
        """
        # [N, L, C]
        hidden = self.encoder(input, mask)
        # [N, flatsize]

        # Split the result into mu and var components
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(
        self, z: Tensor, context: Tensor, mask: Optional[Tensor], **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        hidden = self.fc_hidden(z)
        hidden = hidden.unflatten(1, self.unflattened_shape)
        log_probs = self.decoder(hidden, context, mask)

        return log_probs

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        log_prob_samples = self.predict_sequences(z, device)
        return exp(log_prob_samples)

    def predict_sequences(
        self, z: Tensor, current_device: int, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        z = z.to(current_device)
        B = z.shape[0]
        log_outputs = []
        sequence = torch.zeros(B, self.length, 2, device=z.device)
        sequence[:, :, 0] = self.sos  # all sos with duration 0
        for i in range(self.length):
            # get the predictions
            logits = self.decode(z, context=sequence, mask=None)
            # focus only on the last time step
            logits = logits[:, i, :]  # becomes (B, C)
            log_outputs.append(logits.unsqueeze(1))
            prediction = self.sample(logits)
            # append sampled index to the running sequence
            sequence[:, i, :] = prediction

        log_probs = torch.cat(log_outputs, dim=1)

        return log_probs

    def sample(self, logits):
        acts, duration = torch.split(logits, [self.encodings, 1], dim=-1)
        if self.sampling == "sample":
            # sample from the distribution
            act = torch.multinomial(torch.exp(logits), num_samples=1)  # (B, 1)
        elif self.sampling == "top":
            _, topi = logits.topk(1)
            act = (
                topi.squeeze(-1).detach().unsqueeze(-1)
            )  # detach from history as input?
        else:
            raise ValueError(
                f"Sampling method {self.sampling} not recognized, use 'sample' or 'top'"
            )
        # [N, 1, encodings+1]

        _, topi = acts.topk(1)
        act = (
            topi.squeeze(-1).detach().unsqueeze(-1)
        )  # detach from history as input
        duration = self.decoder.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs

    def infer(
        self,
        x: Tensor,
        device: int,
        input_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Given an encoder input, return reconstructed output and z samples.

        Args:
            x (tensor): [N, steps, acts].

        Returns:
            (tensor: [N, steps, acts], tensor: [N, latent_dims]).
        """
        log_probs_x, _, _, z = self.forward(x, input_mask=input_mask, **kwargs)
        prob_samples = exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        z = z.to(device)
        return prob_samples, z

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        """Override the validation step to include the target during validation.
        This is required for self attention.
        """

        (x, _), (y, y_weights), (labels, _) = batch
        self.curr_device = x.device

        log_probs, mu, log_var, z = self.forward(
            x, conditionals=labels, target=y
        )
        val_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=y,
            weights=y_weights,
            kld_weight=self.kld_loss_weight,
            duration_weight=self.duration_loss_weight,
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
        self.log("hp_metric", val_loss["loss"])


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        ffwd_size,
        length,
        n_head,
        n_layer,
        dropout: float = 0.0,
        embedding: str = "concat",
        position_embedding: str = "learnt",
        time_embedding: str = "none",
    ):
        """Encoder with self-attention layers.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            ffwd_size (int): Number of hidden units in the feedforward layer.
            length (int): Length of the sequence.
            n_head (int): Number of heads in the multi-head attention.
            n_layer (int): Number of layers in the encoder.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            embedding (str, optional): Type of embedding. Defaults to "concat".
            position_embedding (str, optional): Type of positional embedding. Defaults to "learnt".
            time_embedding (str, optional): Type of time embedding. Defaults to "none".
        """
        super(AttentionEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layer

        if embedding.lower() == "concat":
            self.embedding = CustomDurationEmbeddingConcat(
                input_size, hidden_size, dropout=dropout
            )
        elif embedding.lower() == "add":
            self.embedding = CustomDurationEmbeddingAddNorm(
                input_size, hidden_size, dropout=dropout
            )
        else:
            raise ValueError(
                f"Embedding must be either 'concat' or 'add', got {embedding}"
            )

        if position_embedding.lower() == "none":
            self.position_embedding = None
        elif position_embedding.lower() == "learnt":
            self.position_embedding = LearntPositionalEncoding(
                d_model=hidden_size, dropout=0.0, length=length
            )
        elif position_embedding.lower() == "fixed":
            self.position_embedding = FixedPositionalEncoding(
                d_model=hidden_size, dropout=0.0, length=length
            )
        else:
            raise ValueError(
                f"Positional embedding must be either 'none', 'learnt' or 'fixed', got {position_embedding}"
            )

        if time_embedding.lower() == "none":
            self.time_embedding = None
        elif time_embedding.lower() == "start":
            self.time_embedding = StartTimePositionEncoding(dropout=0.0)
        elif time_embedding.lower() == "remaining":
            self.time_embedding = RemainingTimePositionEncoding(dropout=0.0)
        else:
            raise ValueError(
                f"Time embedding must be either 'none', 'start' or 'remaining', got {time_embedding}"
            )

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_size,
                    n_head=n_head,
                    dropout=dropout,
                    ffwd_size=ffwd_size,
                )
                for _ in range(n_layer)
            ]
        )
        # better init?
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, mask=None):
        # idx and targets are both (B,T) tensor of integers
        x = self.embedding(x)  # (B,T,C)
        if self.position_embedding is not None:
            x = self.position_embedding(x)  # (B,T,C)
        if self.time_embedding is not None:
            x = self.time_embedding(x)
        for block in self.blocks:
            x = block(x, mask=mask)  # (B,T,C)
        x = x.flatten(1)

        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        ffwd_size,
        num_heads,
        num_layers,
        length,
        dropout: float = 0.0,
        embedding: str = "concat",
        position_embedding: str = "learnt",
        time_embedding: str = "none",
        latent_context: str = "xattention",
        sos: int = 0,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.max_length = length
        self.sos = sos

        if embedding.lower() == "concat":
            self.embedding = CustomDurationEmbeddingConcat(
                input_size, hidden_size, dropout=dropout
            )
        elif embedding.lower() == "add":
            self.embedding = CustomDurationEmbeddingAddNorm(
                input_size, hidden_size, dropout=dropout
            )
        else:
            raise ValueError(
                f"Embedding must be either 'concat' or 'add', got {embedding}"
            )

        if position_embedding.lower() == "none":
            self.position_embedding = None
        elif position_embedding.lower() == "learnt":
            self.position_embedding = LearntPositionalEncoding(
                d_model=hidden_size, dropout=dropout, length=length
            )
        elif position_embedding.lower() == "fixed":
            self.position_embedding = FixedPositionalEncoding(
                d_model=hidden_size, dropout=dropout, length=length
            )
        else:
            raise ValueError(
                f"Positional embedding must be either 'none', 'learnt' or 'fixed', got {position_embedding}"
            )

        if time_embedding.lower() == "none":
            self.time_embedding = None
        elif time_embedding.lower() == "start":
            self.time_embedding = StartTimePositionEncoding(dropout=0.0)
        elif time_embedding.lower() == "remaining":
            self.time_embedding = RemainingTimePositionEncoding(dropout=0.0)
        else:
            raise ValueError(
                f"Time embedding must be either 'none', 'start' or 'remaining', got {time_embedding}"
            )

        if (
            latent_context.lower() == "xattention"
            or latent_context.lower() == "xatt"
            or latent_context.lower() == "cross_attention"
            or latent_context.lower() == "cross_att"
        ):
            print("Using cross attention for latent context")
            self.blocks = nn.ModuleList(
                [
                    DecoderBlockXAttention(
                        hidden_size,
                        n_head=num_heads,
                        dropout=dropout,
                        block_size=length,
                        ffwd_size=ffwd_size,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif latent_context.lower() == "add":
            print("Using addition for latent context")
            self.blocks = nn.ModuleList(
                [
                    DecoderBlockAddAttention(
                        hidden_size,
                        n_head=num_heads,
                        dropout=dropout,
                        block_size=length,
                        ffwd_size=ffwd_size,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(
                f"Latent context must be either 'xattention' or 'addattention', got {latent_context}"
            )

        self.lm_head = nn.Linear(hidden_size, output_size)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, hidden, target, mask=None):
        # idx and targets are both (B,T) tensor of integers
        outputs = self.embedding(target)  # (B,T,C)
        if self.position_embedding is not None:
            outputs = self.position_embedding(outputs)  # (B,T,C)
        if self.time_embedding is not None:
            outputs = self.time_embedding(outputs)
        for layer in self.blocks:
            outputs = layer(hidden, outputs, mask)
        outputs = self.lm_head(outputs)
        acts_logits, durations = torch.split(
            outputs, [self.output_size - 1, 1], dim=-1
        )
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        durations = torch.log(durations)

        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)

        return log_prob_outputs


class EncoderBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, dropout, ffwd_size: int = None):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, ffwd_size=ffwd_size)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class AttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd=10, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size=head_size, n_embd=n_embd)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedAttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size), diagonal=0)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadMaskedAttention(nn.Module):
    """Multiple heads of masked self-attention in parallel"""

    def __init__(self, num_heads, head_size, block_size, n_embd, dropout=0.0):
        super().__init__()
        self.masked_heads = nn.ModuleList(
            [
                MaskedAttentionHead(
                    head_size=head_size, n_embd=n_embd, block_size=block_size
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.masked_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class CrossAttentionHead(nn.Module):
    """one head of x-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_encode, x_decode, mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x_encode)  # (B,T,hs)
        q = self.query(x_decode)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x_encode)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadCrossAttention(nn.Module):
    """multiple heads of masked x-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd=10, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CrossAttentionHead(head_size=head_size, n_embd=n_embd)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_encode, x, mask=None):
        out = torch.cat([h(x_encode, x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.0, ffwd_size=None):
        super().__init__()
        if ffwd_size is None:
            ffwd_size = n_embd * 2
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffwd_size),
            nn.GELU(),
            nn.Linear(ffwd_size, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlockXAttention(nn.Module):
    def __init__(
        self, n_embd, n_head, block_size, dropout, ffwd_size: int = None
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadMaskedAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.cross_attention = MultiHeadCrossAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, ffwd_size=ffwd_size)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)
        self.ln3 = nn.RMSNorm(n_embd)
        self.ln4 = nn.RMSNorm(n_embd)

    def forward(self, hidden, target, mask=None):
        target = target + self.self_attention(self.ln1(target), mask)
        target = target + self.cross_attention(
            self.ln2(hidden), self.ln3(target), mask
        )
        target = target + self.ffwd(self.ln4(target))
        return target


class DecoderBlockAddAttention(nn.Module):
    def __init__(
        self, n_embd, n_head, block_size, dropout, ffwd_size: int = None
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadMaskedAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, ffwd_size=ffwd_size)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def forward(self, hidden, target, mask=None):
        target = target + self.self_attention(self.ln1(target), mask)
        target = target + hidden
        target = target + self.ffwd(self.ln2(target))
        return target


class LearntPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, length: int = 144):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.arange(0, length, dtype=torch.long)  # (T)
        self.register_buffer("pe", pe)
        self.embedding = nn.Embedding(length, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        _, L, _ = x.shape  # (B,T,C)

        pos_emb = self.embedding(self.pe[:L]).unsqueeze(0)  # (1,L,C)
        x = x + pos_emb  # (B,L,C)
        return self.dropout(x)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, length: int = 144):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(length) / d_model)
        )
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        _, T, _ = x.shape
        x = x + self.pe[:T, :]
        return self.dropout(x)


class StartTimePositionEncoding(nn.Module):
    def __init__(self, dropout: float = 0.0, *args, **kwargs):
        """Positional encoding of start times, replaces dim [:, :, -2].
        Assumes durations are at [:, :, -1]"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        durations = x[:, :, -1]
        start_times = torch.cumsum(durations, dim=-1) - durations
        # start_times = (
        #     start_times - start_times.mean(dim=-1)[:, None]
        # )  # normalize
        x[:, :, -1] = start_times
        return self.dropout(x)


class RemainingTimePositionEncoding(nn.Module):
    def __init__(self, dropout: float = 0.0, *args, **kwargs):
        """Positional encoding for remaining duration, replaces dim [:, :, -2].
        Assumes durations are at [:, :, -1]"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        durations = x[:, :, -1]
        remaining = torch.ones_like(durations) - (
            torch.cumsum(durations, dim=-1) - durations
        )
        # remaining = remaining - remaining.mean(dim=-1)[:, None]  # normalize
        x[:, :, -1] = remaining
        return self.dropout(x)
