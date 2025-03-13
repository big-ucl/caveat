import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, exp, nn

from caveat.models import Base
from caveat.models.embed import CustomDurationEmbeddingConcat


class AutoContAtt(Base):
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
        self.sampling = config.get("sampling", False)
        self.position_embedding = config.get("position_embedding", "fixed")

        self.decoder = AttentionDecoder(
            input_size=self.encodings,
            output_size=self.encodings + 1,
            hidden_size=self.hidden_size,
            ffwd_size=self.ffwd_size,
            num_heads=self.heads,
            num_layers=self.hidden_n,
            length=self.length,
            dropout=self.dropout,
            position_embedding=self.position_embedding,
        )

    def forward(
        self, x: Tensor, target=None, input_mask=None, **kwargs
    ) -> List[Tensor]:
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        if input_mask is not None:
            mask = torch.zeros_like(input_mask)
            mask[input_mask > 0] = 1.0
            mask = mask[:, None, :]
        else:
            mask = None

        if target is not None:  # training
            log_prob = self.decode(context=x, mask=mask)
            return [
                log_prob,
                torch.zeros_like(log_prob),
                torch.zeros_like(log_prob),
                torch.zeros_like(log_prob),
            ]

        # no target so assume generating
        log_prob = self.predict_sequences(current_device=self.curr_device)
        return [
            log_prob,
            torch.zeros_like(log_prob),
            torch.zeros_like(log_prob),
            torch.zeros_like(log_prob),
        ]

    def decode(
        self, context: Tensor, mask: Optional[Tensor], **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        # initialize hidden state as inputs
        log_probs = self.decoder(context, mask)
        return log_probs

    def predict(self, z, device: int, **kwargs) -> Tensor:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        log_prob_samples = self.predict_sequences(device)
        return exp(log_prob_samples)

    def predict_sequences(
        self, current_device: int, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Given samples from the latent space, return the corresponding decoder space map.

        Args:
            z (tensor): [N, latent_dims].
            current_device (int): Device to run the model.

        Returns:
            tensor: [N, steps, acts].
        """
        B = 1024  # todo?
        log_outputs = []
        sequence = torch.zeros(B, self.length, 2, device=current_device)
        sequence[:, :, 0] = self.sos  # all sos with duration 0
        for i in range(self.length):
            # get the predictions
            logits = self.decode(context=sequence, mask=None)
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
        if self.sampling:
            # sample from the distribution
            act = torch.multinomial(torch.exp(logits), num_samples=1)  # (B, 1)
        else:
            _, topi = logits.topk(1)
            act = (
                topi.squeeze(-1).detach().unsqueeze(-1)
            )  # detach from history as input?
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
        log_probs_x, _, _, _ = self.forward(x, input_mask=input_mask, **kwargs)
        prob_samples = exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        return prob_samples, torch.zeros_like(prob_samples)

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
            target=y,
            mask=y_weights,
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

    def loss_function(self, log_probs, target, mask, **kwargs) -> dict:
        """Loss function for sequence encoding [N, L, 2]."""
        # unpack act probs and durations
        target_acts, target_durations = self.unpack_encoding(target)
        pred_acts, pred_durations = self.unpack_encoding(log_probs)
        pred_durations = torch.exp(pred_durations)

        # normalise mask weights
        mask = mask / mask.mean(-1).unsqueeze(-1)
        duration_mask = mask.clone()
        duration_mask[:, 0] = 0.0
        duration_mask[
            torch.arange(duration_mask.shape[0]),
            (mask != 0).cumsum(-1).argmax(1),
        ] = 0.0

        # activity loss
        recon_act_nlll = self.base_NLLL(
            pred_acts.view(-1, self.encodings), target_acts.view(-1).long()
        )
        act_recon = (recon_act_nlll * mask.view(-1)).mean()
        scheduled_act_weight = (
            self.activity_loss_weight * self.scheduled_act_weight
        )
        w_act_recon = scheduled_act_weight * act_recon

        # duration loss
        recon_dur_mse = self.MSE(pred_durations, target_durations)
        recon_dur_mse = (recon_dur_mse * duration_mask).mean()
        scheduled_dur_weight = (
            self.duration_loss_weight * self.scheduled_dur_weight
        )
        w_dur_recon = scheduled_dur_weight * recon_dur_mse

        # reconstruction loss
        w_recons_loss = w_act_recon + w_dur_recon

        # final loss
        loss = w_recons_loss

        return {
            "loss": loss,
            "recon_loss": w_recons_loss.detach(),
            "act_recon": w_act_recon.detach(),
            "dur_recon": w_dur_recon.detach(),
            "act_weight": torch.tensor([scheduled_act_weight]).float(),
            "dur_weight": torch.tensor([scheduled_dur_weight]).float(),
        }


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
        position_embedding: str = "learnt",
        sos: int = 0,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.max_length = length
        self.sos = sos
        self.embedding = CustomDurationEmbeddingConcat(
            input_size, hidden_size, dropout=dropout
        )
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
                DecoderBlockMAskedSelfAttention(
                    hidden_size,
                    n_head=num_heads,
                    dropout=dropout,
                    block_size=length,
                    ffwd_size=ffwd_size,
                )
                for _ in range(num_layers)
            ]
        )
        # self.ln_f = nn.LayerNorm(hidden_size)
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

    def forward(self, target, mask=None):
        # idx and targets are both (B,T) tensor of integers
        outputs = self.embedding(target)  # (B,T,C)
        outputs = self.position_embedding(outputs)  # (B,T,C)
        for layer in self.blocks:
            outputs = layer(outputs, mask)

        # outputs = self.ln_f(outputs)  # (B,T,C)
        outputs = self.lm_head(outputs)

        acts_logits, durations = torch.split(
            outputs, [self.output_size - 1, 1], dim=-1
        )
        acts_log_probs = self.activity_logprob_activation(acts_logits)
        durations = self.duration_activation(durations)
        durations = torch.log(durations)

        log_prob_outputs = torch.cat((acts_log_probs, durations), dim=-1)
        return log_prob_outputs


class DecoderBlockMAskedSelfAttention(nn.Module):
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

    def forward(self, target, mask=None):
        target = target + self.self_attention(self.ln1(target), mask)
        target = target + self.ffwd(self.ln2(target))
        return target


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
