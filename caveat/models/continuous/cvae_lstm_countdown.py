from typing import Optional

import torch
from torch import Tensor, nn

from caveat import current_device
from caveat.models.continuous.cvae_lstm import CVAEContLSTM
from caveat.models.embed import CustomDurationEmbeddingConcat


class CVAEContLSTMCountdown(CVAEContLSTM):
    def build_decoder(self, config):
        decoder_conditionality = config.get("decoder_conditionality", "none")
        if decoder_conditionality == "none":
            print("Decoder conditionality is 'none'")
            return Decoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings,
                num_layers=self.hidden_n,
                max_length=self.length,
                dropout=self.dropout,
                sos=self.sos,
                eos=self.eos,
                act_head_depth=self.act_head_depth,
                act_hidden_size=self.act_hidden_size,
                dur_head_depth=self.dur_head_depth,
                dur_hidden_size=self.dur_hidden_size,
                budget_depth=self.budget_depth,
                budget_hidden_size=self.budget_hidden_size,
            )
        elif decoder_conditionality in {"add", "inputs_add"}:
            print("Decoder conditionality is 'inputs'")
            return InputsAddConditionalDecoder(
                input_size=self.encodings,
                hidden_size=self.hidden_size,
                output_size=self.encodings,
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
                output_size=self.encodings,
                num_layers=self.hidden_n,
                max_length=self.length,
                labels_size=self.labels_hidden_size,
                dropout=self.dropout,
                sos=self.sos,
            )
        raise ValueError(
            "Decoder conditionality must be 'none', 'add/inputs_add' or 'concat/inputs_concat'"
        )


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
        act_head_depth: int = 2,
        act_hidden_size: int = 16,
        dur_head_depth: int = 2,
        dur_hidden_size: int = 16,
        budget_depth: int = 1,
        budget_hidden_size: int = 16,
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
            act_head_depth (int): number of hidden layers in the activity head. Defaults to 2.
            act_hidden_size (int): hidden size of the activity head. Defaults to 16.
            dur_head_depth (int): number of hidden layers in the duration head. Defaults to 2.
            dur_hidden_size (int): hidden size of the duration head. Defaults to
            budget_depth (int): number of hidden layers in the budget head. Defaults to 1.
            budget_hidden_size (int): hidden size of the budget head. Defaults to 16.
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
        self.budget_head = FeedForwards(
            input_size=1,
            hidden_size=budget_hidden_size,
            output_size=hidden_size,
            depth=budget_depth,
            dropout=dropout,
        )
        self.act_head = FeedForwards(
            input_size=hidden_size,
            hidden_size=act_hidden_size,
            output_size=output_size,
            depth=act_head_depth,
            dropout=dropout,
        )
        self.duration_head = FeedForwards(
            input_size=hidden_size,
            hidden_size=dur_hidden_size,
            output_size=1,  # single duration output
            depth=dur_head_depth,
            dropout=dropout,
        )
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()
        self.budget_activation = nn.Sigmoid()

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        budget = torch.ones(batch_size, 1, 1, device=hidden.device)
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, budget
            )
            outputs.append(decoder_output.squeeze(-2))

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

            # remove durations from budget
            budget = (budget - decoder_input[:, :, 1:]).detach()

            # ensure budget > 0
            budget = self.budget_activation(budget)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        log_prob_outputs = torch.log(outputs)
        # TODO: remove logs, includes utils, losses, etc

        return log_prob_outputs

    def forward_step(self, x, hidden, budget):
        # [N, 1, 2]
        embedded = self.embedding(x)
        embedded_budget = self.budget_head(budget)

        output, hidden = self.lstm(embedded, hidden)
        output = output + embedded_budget  # add budget to output

        act_prediction = self.act_head(output)
        act_probs = self.activity_prob_activation(act_prediction)

        durations = self.duration_head(output)
        durations = self.duration_activation(durations)
        durations = durations * budget

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
        # duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs


class InputsAddConditionalDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        flat_size_encode: int,
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
            flat_size_encode,
            num_layers,
            max_length,
            dropout,
            sos,
        )
        self.labels_ff = FeedForwards(
            input_size=labels_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            depth=1,
            dropout=dropout,
        )

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        budget = torch.ones(batch_size, 1, 1, device=hidden.device)
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        labels_hidden = self.labels_ff(labels).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, budget, labels_hidden
            )
            outputs.append(decoder_output.squeeze(-2))

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

            # remove durations from budget
            budget = (budget - decoder_input[:, :, 1:]).detach()

            # ensure budget > 0
            budget = self.budget_activation(budget)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        log_prob_outputs = torch.log(outputs)
        # TODO: remove logs, includes utils, losses, etc

        return log_prob_outputs

    def forward_step(self, x, hidden, budget, labels):
        # [N, 1, 2]
        embedded = self.embedding(x)
        embedded_budget = self.budget_head(budget)

        output, hidden = self.lstm(embedded, hidden)
        output = output + embedded_budget + labels

        act_prediction = self.act_head(output)
        act_probs = self.activity_prob_activation(act_prediction)

        durations = self.duration_head(output)
        durations = self.duration_activation(durations)
        durations = durations * budget

        prediction = torch.cat((act_probs, durations), dim=-1)
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
        eos: int = 1,
        act_head_depth: int = 2,
        act_hidden_size: int = 16,
        dur_head_depth: int = 2,
        dur_hidden_size: int = 16,
        budget_depth: int = 1,
        budget_hidden_size: int = 16,
        conditional_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.current_device = current_device()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.sos = sos
        self.eos = eos

        if conditional_hidden_size is None:
            conditional_hidden_size = hidden_size
        else:
            conditional_hidden_size = conditional_hidden_size

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
        self.budget_head = FeedForwards(
            input_size=1,
            hidden_size=budget_hidden_size,
            output_size=hidden_size,
            depth=budget_depth,
            dropout=dropout,
        )
        self.act_head = FeedForwards(
            input_size=hidden_size + conditional_hidden_size,
            hidden_size=act_hidden_size,
            output_size=output_size,
            depth=act_head_depth,
            dropout=dropout,
        )
        self.duration_head = FeedForwards(
            input_size=hidden_size + conditional_hidden_size,
            hidden_size=dur_hidden_size,
            output_size=1,  # single duration output
            depth=dur_head_depth,
            dropout=dropout,
        )
        self.activity_prob_activation = nn.Softmax(dim=-1)
        self.activity_logprob_activation = nn.LogSoftmax(dim=-1)
        self.duration_activation = nn.Sigmoid()
        self.budget_activation = nn.Sigmoid()

        self.labels_ff = FeedForwards(
            input_size=labels_size,
            hidden_size=hidden_size,
            output_size=conditional_hidden_size,
            depth=1,
            dropout=dropout,
        )

    def forward(self, batch_size, hidden, labels, target=None, **kwargs):
        hidden, cell = hidden
        decoder_input = torch.zeros(batch_size, 1, 2, device=hidden.device)
        decoder_input[:, :, 0] = self.sos  # set as SOS
        budget = torch.ones(batch_size, 1, 1, device=hidden.device)
        hidden = hidden.contiguous()
        cell = cell.contiguous()
        decoder_hidden = (hidden, cell)
        outputs = []

        labels_hidden = self.labels_ff(labels).unsqueeze(1)

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, budget, labels_hidden
            )
            outputs.append(decoder_output.squeeze(-2))

            if target is not None:
                # teacher forcing for next step
                decoder_input = target[:, i : i + 1, :]
            else:
                # no teacher forcing use decoder output
                decoder_input = self.pack(decoder_output)

            # remove durations from budget
            budget = (budget - decoder_input[:, :, 1:]).detach()

            # ensure budget > 0
            budget = self.budget_activation(budget)

        outputs = torch.stack(outputs).permute(1, 0, 2)  # [N, steps, acts]
        log_prob_outputs = torch.log(outputs)
        # TODO: remove logs, includes utils, losses, etc

        return log_prob_outputs

    def forward_step(self, x, hidden, budget, labels):
        # [N, 1, 2]
        embedded = self.embedding(x)
        embedded_budget = self.budget_head(budget)

        output, hidden = self.lstm(embedded, hidden)
        output = output + embedded_budget
        output = torch.cat((output, labels), dim=-1)

        act_prediction = self.act_head(output)
        act_probs = self.activity_prob_activation(act_prediction)

        durations = self.duration_head(output)
        durations = self.duration_activation(durations)
        durations = durations * budget

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
        # duration = self.duration_activation(duration)
        outputs = torch.cat((act, duration), dim=-1)
        # [N, 1, 2]
        return outputs


class FeedForwards(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int = 1,
        dropout: float = 0.0,
        norm: bool = False,
        init: Optional[str] = None,
    ):
        """Linear head for VAE decoder.

        Args:
            input_size (int): input size.
            hidden_size (int): hidden layer size.
            output_size (int): output size.
            depth (int): number of hidden layers.
            dropout (float): dropout probability.
            norm (bool): whether to apply layer normalization. Defaults to False.
            init (Optional[str]): initialization method. Defaults to None.
        """
        super(FeedForwards, self).__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError(
                "Input, hidden, and output sizes must be positive integers"
            )
        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")

        if depth == 0:
            # return a vector of zeros
            layers = [Zeros(output_size)]
        elif depth == 1:
            layers = [nn.Linear(input_size, output_size)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if norm:
                layers.append(nn.LayerNorm(hidden_size))
        else:
            layers = []
            in_features = input_size
            for _ in range(depth - 1):
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.LeakyReLU())
                in_features = hidden_size
            layers.append(nn.Linear(hidden_size, output_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if norm:
                layers.append(nn.LayerNorm(hidden_size))
        self.block = nn.Sequential(*layers)

        if init is not None:
            if init == "xavier_normal":
                self.init_xavier_normal()
            elif init == "xavier_uniform":
                self.init_xavier_uniform()
            elif init == "kaiming_normal":
                self.init_kaiming_normal()
            elif init == "kaiming_uniform":
                self.init_kaiming_uniform()

    def init_xavier_normal(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def init_xavier_uniform(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def init_kaiming_normal(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def init_kaiming_uniform(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity="leaky_relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear head.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor with activity probabilities.
        """
        return self.block(x)


class Zeros(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.par = nn.Parameter(torch.rand(1, 1, output_size))

    def forward(self, batch: Tensor) -> Tensor:
        return self.par.expand(batch.size(0), -1, -1) * batch.sum() * 0.0
