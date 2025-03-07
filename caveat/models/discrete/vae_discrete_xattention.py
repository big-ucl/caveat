import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, exp, nn

from caveat.models.base import Base


class VAEDiscXTrans(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer."""
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.latent_dim = config["latent_dim"]
        print(f"Latent dim: {self.latent_dim}")
        self.hidden_size = config["hidden_size"]
        print(f"Hidden size: {self.hidden_size}")
        self.heads = config["heads"]
        print(f"Heads: {self.heads}")
        self.hidden_n = config["hidden_n"]
        print(f"Hidden n: {self.hidden_n}")
        self.dropout = config["dropout"]
        print(f"Dropout: {self.dropout}")
        self.length = self.in_shape[0]
        print(f"Length: {self.length}")
        self.position_embedding = config.get("position_embedding", "learnt")
        print(f"Positional embedding: {self.position_embedding}")
        self.sampling = config.get("sampling", False)
        print(f"Sampling: {self.sampling}")

        self.encoder = AttentionEncoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            length=self.length,
            n_head=self.heads,
            n_layer=self.hidden_n,
            dropout=self.dropout,
            position_embedding=self.position_embedding,
        )
        self.decoder = AttentionDecoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_heads=self.heads,
            num_layers=self.hidden_n,
            length=self.length,
            dropout=self.dropout,
            position_embedding=self.position_embedding,
        )
        self.unflattened_shape = (self.length, self.hidden_size)
        flat_size_encode = self.length * self.hidden_size
        self.fc_mu = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_var = nn.Linear(flat_size_encode, self.latent_dim)
        self.fc_hidden = nn.Linear(self.latent_dim, flat_size_encode)

        if config.get("share_embed", False):
            print("Sharing embeddings")
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, x: Tensor, target=None, **kwargs) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        mu, log_var = self.encode(x, labels=None)
        z = self.reparameterize(mu, log_var)

        if target is not None:  # training
            log_prob_y = self.decode(z, context=x)
            return [log_prob_y, mu, log_var, z]

        # no target so assume generating
        log_prob = self.predict_sequences(z, current_device=z.device)
        return [log_prob, mu, log_var, z]

    def decode(
        self, z: Tensor, context=None, **kwargs
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
        log_probs = self.decoder(hidden, context)

        return log_probs

    def loss_function(
        self,
        log_probs: Tensor,
        mu: Tensor,
        log_var: Tensor,
        target: Tensor,
        mask: Tensor,
        **kwargs,
    ) -> dict:
        return self.discretized_loss(
            log_probs, mu, log_var, target, mask, **kwargs
        )

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
        sequences = torch.zeros(B, 1, device=z.device)
        for _ in range(self.length):
            # get the predictions
            log_probs = self.decode(z, context=sequences)
            # focus only on the last time step
            last_log_probs = log_probs[:, -1, :]  # becomes (B, C)
            log_outputs.append(last_log_probs.unsqueeze(1))
            if self.sampling:
                # sample from the distribution
                next = torch.multinomial(
                    torch.exp(last_log_probs), num_samples=1
                )  # (B, 1)
            else:
                _, next = last_log_probs.topk(1)
            # append sampled index to the running sequence
            sequences = torch.cat((sequences, next), dim=1)  # (B, T+1)

        log_probs = torch.cat(log_outputs, dim=1)

        return log_probs


class AttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
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

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedAttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd=10, block_size=128, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadMaskedAttention(nn.Module):
    """Multiple heads of masked self-attention in parallel"""

    def __init__(
        self, num_heads, head_size, block_size, n_embd=10, dropout=0.0
    ):
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

    def forward(self, x):
        out = torch.cat([h(x) for h in self.masked_heads], dim=-1)
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

    def forward(self, x_encode, x_decode):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x_encode)  # (B,T,hs)
        q = self.query(x_decode)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
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

    def forward(self, x_encode, x):
        out = torch.cat([h(x_encode, x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 1 * n_embd),
            nn.GELU(),
            nn.Linear(1 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadMaskedAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ca = MultiHeadCrossAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, x_encode, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x_encode), self.ln3(x))
        x = x + self.ffwd(self.ln4(x))
        return x


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


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        length,
        n_head,
        n_layer,
        dropout: float = 0.0,
        position_embedding: str = "learnt",
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        if position_embedding == "learnt":
            self.position_embedding = LearntPositionalEncoding(
                d_model=hidden_size, dropout=0.0, length=length
            )
        elif position_embedding == "fixed":
            self.position_embedding = FixedPositionalEncoding(
                d_model=hidden_size, dropout=0.0, length=length
            )
        else:
            raise ValueError(
                f"Positional embedding must be either 'learnt' or 'fixed', got {position_embedding}"
            )

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(hidden_size, n_head=n_head, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)  # final layer norm
        # self.lm_head = nn.Linear(hidden_size, input_size)

        # better init
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

    def forward(self, x):
        # idx and targets are both (B,T) tensor of integers
        x = self.embedding(x.long())  # (B,T,C)
        x = self.position_embedding(x)  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        x = x.flatten(1)

        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_heads,
        num_layers,
        length,
        dropout: float = 0.0,
        position_embedding: str = "learnt",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        if position_embedding == "learnt":
            self.position_embedding = LearntPositionalEncoding(
                d_model=hidden_size, dropout=dropout, length=length
            )
        elif position_embedding == "fixed":
            self.position_embedding = FixedPositionalEncoding(
                d_model=hidden_size, dropout=dropout, length=length
            )
        else:
            raise ValueError(
                f"Positional embedding must be either 'learnt' or 'fixed', got {position_embedding}"
            )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size,
                    n_head=num_heads,
                    dropout=dropout,
                    block_size=length,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, output_size)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
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

    def forward(self, x_encode, x):
        # idx and targets are both (B,T) tensor of integers
        x = self.embedding(x.long())  # (B,T,C)
        x = self.position_embedding(x)  # (B,T,C)

        for layer in self.blocks:
            x = layer(x_encode, x)

        x = self.ln_f(x)  # (B,T,C)
        x = self.lm_head(x)
        # todo get ride of this ^, needs to be done for cnn too

        acts_log_probs = self.activity_logprob_activation(x)

        return acts_log_probs
