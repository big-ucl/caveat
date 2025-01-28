from typing import List, Tuple

from torch import Tensor, nn

from caveat.models import CustomDurationEmbedding
from caveat.models.schedule2label.experiment import LabelExperiment


class Schedule2LabelFeedForward(LabelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, **config):
        self.hidden_size = config["hidden_size"]
        self.hidden_n = config["hidden_n"]
        self.label_hidden_size = config.get(
            "label_hidden_size", self.hidden_size
        )
        self.label_hidden_layers = config.get(
            "label_hidden_layers", self.hidden_n
        )
        self.dropout = config["dropout"]
        length, _ = self.in_shape

        self.encoder = Encoder(
            length=length,
            input_size=self.encodings,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_n,
            dropout=self.dropout,
        )
        self.decoder = AttributeDecoder(
            attribute_embed_sizes=self.attribute_embed_sizes,
            hidden_size=self.label_hidden_size,
        )
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(x)
        return self.decode(z)

    def encode(self, input: Tensor) -> Tensor:
        # [N, L, C]
        return self.encoder(input)

    def decode(self, z: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        return self.decoder(z)

    def loss_function(
        self, probs: Tensor, target: Tensor, mask: Tensor, **kwargs
    ) -> dict:
        """Calculate the loss function for the model.

        Args:
            log_probs ((tensor, tensor)): Log probabilities for the output sequence.
            mu (tensor): Mean of the latent space.
            log_var (tensor): Log variance of the latent space.

        Returns:
            dict: Loss dictionary.
        """
        logs = {}
        # attributes loss
        loss = 0
        for i, y in enumerate(probs):
            t = target[:, i].long()
            weight = mask[:, i]
            weight = weight / weight.mean()  # average weight to 1
            nll = self.loss(y, t)
            logs[f"nll_{i}"] = nll.mean()
            weighted_nll = nll * weight
            logs[f"weighted_nll_{i}"] = weighted_nll.mean()
            loss += weighted_nll.mean()
        loss = loss / len(probs)
        scheduled_label_weight = (
            self.scheduled_label_weight * self.label_loss_weight
        )
        weighted_loss = scheduled_label_weight * loss

        logs.update(
            {
                "loss": weighted_loss,
                "weight": Tensor([scheduled_label_weight]).float(),
            }
        )
        return logs

    def predict(self, x: Tensor, device: int, **kwargs) -> Tensor:
        x = x.to(device)
        logits_y = self.forward(x=x, **kwargs)
        probs_y = [nn.functional.softmax(y, dim=-1) for y in logits_y]
        return probs_y


class Encoder(nn.Module):
    def __init__(
        self,
        length: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """LSTM Encoder.

        Args:
            length (int): length of sequences.
            input_size (int): lstm input size.
            hidden_size (int): lstm hidden size.
            num_layers (int): number of lstm layers.
            dropout (float): dropout. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = CustomDurationEmbedding(
            input_size, hidden_size, dropout=dropout
        )
        input_size = hidden_size * length
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        self.ffs = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.ffs(embedded.flatten(1))
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return hidden


class AttributeDecoder(nn.Module):
    def __init__(self, attribute_embed_sizes, hidden_size):
        super(AttributeDecoder, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.attribute_nets = nn.ModuleList(
            [nn.Linear(hidden_size, s) for s in attribute_embed_sizes]
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        log_probs = [net(x) for net in self.attribute_nets]
        return log_probs
