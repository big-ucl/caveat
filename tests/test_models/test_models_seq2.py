import torch

from caveat.models.seq2score.lstm import Seq2ScoreLSTM
from caveat.models.seq2seq.lstm import Seq2SeqLSTM


def test_seq2seq_forward():
    N = 10
    L = 16
    A = 5
    M = 5
    C = A + M + 2
    x = torch.randn(N, L, C)
    weights = torch.ones((N, L))
    acts, durations, modes, distances = x.split([A, 1, M, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    modes_max = modes.argmax(dim=-1).unsqueeze(-1)
    conditionals = torch.randn(N, 4)
    x_encoded = torch.cat([acts_max, durations, modes_max, distances], dim=-1)
    model = Seq2SeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=(A, M),
        encoding_weights=None,
        conditionals_size=4,
        **{"hidden_n": 1, "hidden_size": 12, "latent_dim": 2, "dropout": 0.1},
    )
    log_probs = model(x_encoded, conditionals=conditionals)
    assert log_probs.shape == (N, L, C)
    losses = model.loss_function(log_probs, x_encoded, mask=weights)
    assert "loss" in losses
    assert "recon_loss" in losses


def test_seq2score_forward():
    N = 10
    L = 16
    A = 5
    M = 5
    C = A + M + 2
    x = torch.randn(N, L, C)
    weights = torch.ones((N, L))
    acts, durations, modes, distances = x.split([A, 1, M, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    modes_max = modes.argmax(dim=-1).unsqueeze(-1)
    conditionals = torch.randn(N, 4)
    target = torch.randn(N)
    x_encoded = torch.cat([acts_max, durations, modes_max, distances], dim=-1)
    model = Seq2ScoreLSTM(
        in_shape=x_encoded[0].shape,
        encodings=(A, M),
        encoding_weights=None,
        conditionals_size=4,
        **{"hidden_n": 1, "hidden_size": 12, "latent_dim": 2, "dropout": 0.1},
    )
    score = model(x_encoded, conditionals=conditionals)
    assert score.shape == (N, 1)
    losses = model.loss_function(score, target, mask=weights)
    assert "loss" in losses
