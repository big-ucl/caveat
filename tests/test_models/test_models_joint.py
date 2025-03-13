import torch

from caveat.models.joint_vaes.jvae_continuous import JVAEContLSTM


def test_jvae_lstm_forward():
    # schedules
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)

    # attributes
    labels = torch.Tensor([[1, 0], [1, 5], [0, 0]]).long()
    label_weights = torch.ones((3, 2))

    model = JVAEContLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=None,
        labels_size=10,
        **{
            "hidden_n": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
            "label_embed_sizes": [2, 6],
        },
    )
    (log_prob_x, log_prob_y), mu, log_var, z = model(
        x_encoded, conditionals=labels
    )
    assert log_prob_x.shape == (3, 10, 6)
    assert len(log_prob_y) == 2
    assert log_prob_y[0].shape == (3, 2)
    assert log_prob_y[1].shape == (3, 6)
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=(log_prob_x, log_prob_y),
        mu=mu,
        log_var=log_var,
        targets=(x_encoded, labels),
        masks=(weights, label_weights),
    )
    assert "loss" in losses
