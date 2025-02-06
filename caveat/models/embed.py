import torch
from torch import nn


class OneHotEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity onehot embedding and duration."""
        super().__init__()
        if hidden_size != input_size + 1:
            raise ValueError("Hidden size must be equal to input size plus 1.")
        self.classes = input_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(
            nn.functional.one_hot(embedded.int(), self.classes)
        )
        embedded = torch.cat((embedded, durations), dim=-1)
        return embedded


class OneHotPlusLinearEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity onehot embedding and duration and linear layer."""
        super().__init__()
        self.classes = input_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(
            nn.functional.one_hot(embedded.int(), self.classes)
        )
        embedded = torch.cat((embedded, durations), dim=-1)
        embedded = self.fc(embedded)
        return embedded


class CustomDurationEmbeddingConcat(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration."""
        super().__init__()
        if hidden_size < 2:
            raise ValueError("Hidden size must be greater than 1.")
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)
        return embedded


class CustomDurationEmbeddingConcatNorm(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration."""
        super().__init__()
        if hidden_size < 2:
            raise ValueError("Hidden size must be greater than 1.")
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        durations = (
            durations - durations.mean(dim=-2)[:, :, None]
        )  # lazy normalization
        embedded = torch.cat((embedded, durations), dim=-1)
        return embedded


class CustomDurationEmbeddingAddNorm(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration."""
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        durations = (
            durations - durations.mean(dim=-2)[:, :, None]
        )  # lazy normalization
        embedded = embedded + durations
        return embedded


class CustomDurationModeDistanceEmbedding(nn.Module):
    def __init__(
        self,
        act_embeddings,
        mode_embeddings,
        hidden_act_size,
        hidden_mode_size,
        dropout: float = 0.1,
    ):
        """Embedding that combines activity embedding layer and duration and mode distance."""
        super().__init__()
        self.act_embedding = nn.Embedding(act_embeddings, hidden_act_size)
        self.mode_embedding = nn.Embedding(mode_embeddings, hidden_mode_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        acts, durations, modes, distances = torch.split(x, [1, 1, 1, 1], dim=-1)
        acts = self.dropout(self.act_embedding(acts.int())).squeeze(-2)
        modes = self.dropout(self.mode_embedding(modes.int())).squeeze(-2)
        embedded = torch.cat((acts, durations, modes, distances), dim=-1)
        return embedded


class CustomCombinedEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration and end time."""
        super().__init__()
        if hidden_size < 3:
            raise ValueError("Hidden size must be at least 3.")
        self.embedding = nn.Embedding(input_size, hidden_size - 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        ends = torch.cumsum(durations, dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations, ends), dim=-1)
        return embedded


class CustomLinearEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.1):
        """Embedding that combines activity embedding layer and duration using a linear layer."""
        super().__init__()
        if hidden_size < 2:
            raise ValueError("Hidden size must be greater than 1.")
        self.embedding = nn.Embedding(input_size, hidden_size - 1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded, durations = torch.split(x, [1, 1], dim=-1)
        embedded = self.dropout(self.embedding(embedded.int())).squeeze(-2)
        embedded = torch.cat((embedded, durations), dim=-1)
        embedded = self.fc(embedded)
        return embedded
