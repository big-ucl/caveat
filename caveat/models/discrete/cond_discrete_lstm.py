from typing import List, Optional, Tuple

import torch
from torch import Tensor, exp, nn

from caveat import current_device
from caveat.models import Base


class CondDiscLSTM(Base):
    def __init__(self, *args, **kwargs):
        """RNN based encoder and decoder with encoder embedding layer and conditionality."""
        super().__init__(*args, **kwargs)
        if self.conditionals_size is None:
            raise UserWarning(
                "Model requires conditionals_size, please check you have configures a compatible encoder and condition attributes"
            )

    def build(self, **config):
        self.latent_dim = 1  # dummy value for the predict dataloader
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.dropout = config["dropout"]
        length = self.in_shape[0]
        bidirectional = config.get("bidirectional", False)

        self.decoder = Decoder(
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            output_size=self.encodings,
            num_layers=self.hidden_n,
            max_length=length,
            dropout=self.dropout,
            sos=self.sos,
            bidirectional=bidirectional,
        )
        if bidirectional:
            flat_size_encode = self.hidden_n * self.hidden_size * 2 * 2
            self.adjusted_layers = self.hidden_n * 2
            self.unflatten_shape = (2 * 2 * self.hidden_n, self.hidden_size)
        else:
            flat_size_encode = self.hidden_n * self.hidden_size * 2
            self.adjusted_layers = self.hidden_n
            self.unflatten_shape = (2 * self.hidden_n, self.hidden_size)
        self.fc_hidden = nn.Linear(self.conditionals_size, flat_size_encode)
        self.fc_x = nn.Linear(self.conditionals_size, self.hidden_size)

    def forward(
        self,
        x: Tensor,
        conditionals: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        **kwargs,
    ) -> List[Tensor]:

        log_probs = self.decode(z=x, conditionals=conditionals, target=target)
        return [log_probs, Tensor([]), Tensor([]), Tensor([])]

    def loss_function(
        self, log_probs: Tensor, target: Tensor, mask: Tensor, **kwargs
    ) -> dict:
        """Loss function for discretized encoding [N, L]."""
        # activity loss
        recon_act_nlll = self.NLLL(log_probs.squeeze().permute(0, 2, 1), target)

        # loss
        loss = recon_act_nlll

        return {"loss": loss, "recon_act_nlll_loss": recon_act_nlll}

    def encode(self, input: Tensor):
        return None

    def decode(
        self, z: None, conditionals: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Decode latent sample to batch of output sequences.

        Args:
            z (tensor): Latent space batch [N, latent_dims].

        Returns:
            tensor: Output sequence batch [N, steps, acts].
        """
        h = self.fc_hidden(conditionals)
        x = self.fc_x(conditionals).unsqueeze(-2)

        # initialize hidden state
        hidden = h.unflatten(1, self.unflatten_shape).permute(
            1, 0, 2
        )  # ([2xhidden, N, layers])
        hidden = hidden.split(
            self.adjusted_layers
        )  # ([hidden, N, layers, [hidden, N, layers]])

        log_probs = self.decoder(hidden=hidden, x=x, target=None)

        return log_probs

    def predict(
        self, z: Tensor, conditionals: Tensor, device: int, **kwargs
    ) -> Tensor:
        z = z.to(device)
        conditionals = conditionals.to(device)
        return exp(self.decode(z=z, conditionals=conditionals, kwargs=kwargs))


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
        bidirectional: bool = True,
    ):
        """LSTM Decoder.
        No teacher forcing.
        LSTM unit input is conditionals.

        Args:
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            max_length (int): max length of sequences.
            dropout (float): dropout probability. Defaults to 0.
            sos (int): start of sequence token. Defaults to 0.
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
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)

    def forward(self, hidden, x, **kwargs):
        hidden, cell = hidden
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for _ in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                x, decoder_hidden
            )
            outputs.append(decoder_output)

        acts_logits = torch.cat(outputs, dim=1)  # [N, L, C]
        acts_log_probs = self.activity_logprob_activation(acts_logits)

        return acts_log_probs

    def forward_step(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, hidden
