import pytest
import torch

from caveat.models.continuous.auto_lstm import AutoContLSTM
from caveat.models.continuous.cond_lstm import CondContLSTM
from caveat.models.continuous.cvae_lstm import CVAEContLSTM
from caveat.models.continuous.cvae_lstm_nudger import CVAEContLSTMNudger
from caveat.models.continuous.cvae_lstm_nudger_adversarial import (
    CVAEContLSTMNudgerAdversarial,
    Discriminator,
)
from caveat.models.continuous.vae_cnn1d import VAEContCNN1D
from caveat.models.continuous.vae_cnn2d import VAEContCNN2D
from caveat.models.continuous.vae_fc import VAEContFC
from caveat.models.continuous.vae_lstm import VAEContLSTM


def test_auto_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    labels = torch.randn(3, 10)  # (batch, channels)
    label_weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    model = AutoContLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        labels_size=10,
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, _, _, _ = model(x_encoded, labels=labels)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(
        log_probs=log_prob_y,
        target=x_encoded,
        weights=weights,
        label_weights=label_weights,
    )
    assert "loss" in losses


def test_conditional_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)

    label_a = torch.randn(3, 5).argmax(dim=-1)
    label_b = torch.randn(3, 2).argmax(dim=-1)
    labels = torch.concat((label_a[:, None], label_b[:, None]), dim=-1)

    label_weights = (torch.ones((3, 2)), torch.ones((3, 1)))

    model = CondContLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        labels_size=10,
        **{
            "label_embed_sizes": [5, 2],
            "hidden_n": 1,
            "hidden_size": 8,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, _, _, _ = model(x_encoded, labels=labels)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(
        log_probs=log_prob_y,
        target=x_encoded,
        weights=weights,
        label_weights=label_weights,
    )
    assert "loss" in losses


testdata = [
    ("none", "concat", "none"),
    ("none", "add", "none"),
    ("hidden", "concat", "none"),
    ("hidden", "add", "none"),
    ("inputs_add", "concat", "none"),
    ("inputs_add", "add", "none"),
    ("inputs_concat", "concat", "none"),
    ("inputs_concat", "add", "none"),
    ("both_add", "concat", "none"),
    ("both_add", "add", "none"),
    ("both_concat", "concat", "none"),
    ("both_concat", "add", "none"),
    ("none", "concat", "inputs_add"),
    ("none", "add", "inputs_add"),
    ("hidden", "concat", "inputs_add"),
    ("hidden", "add", "inputs_add"),
    ("inputs_add", "concat", "inputs_add"),
    ("inputs_add", "add", "inputs_add"),
    ("inputs_concat", "concat", "inputs_add"),
    ("inputs_concat", "add", "inputs_add"),
    ("both_add", "concat", "inputs_add"),
    ("both_add", "add", "inputs_add"),
    ("both_concat", "concat", "inputs_add"),
    ("both_concat", "add", "inputs_add"),
    ("none", "concat", "inputs_concat"),
    ("none", "add", "inputs_concat"),
    ("hidden", "concat", "inputs_concat"),
    ("hidden", "add", "inputs_concat"),
    ("inputs_add", "concat", "inputs_concat"),
    ("inputs_add", "add", "inputs_concat"),
    ("inputs_concat", "concat", "inputs_concat"),
    ("inputs_concat", "add", "inputs_concat"),
    ("both_add", "concat", "inputs_concat"),
    ("both_add", "add", "inputs_concat"),
    ("both_concat", "concat", "inputs_concat"),
    ("both_concat", "add", "inputs_concat"),
]


@pytest.mark.parametrize("encoder,latent,decoder", testdata)
def test_cvae_lstm_forward(encoder, latent, decoder):
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    label_a = torch.randn(3, 5).argmax(dim=-1)
    label_b = torch.randn(3, 2).argmax(dim=-1)
    labels = torch.concat((label_a[:, None], label_b[:, None]), dim=-1)
    label_weights = (torch.ones((3, 2)), torch.ones((3, 1)))
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = CVAEContLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        labels_size=2,
        **{
            "label_embed_sizes": [5, 2],
            "hidden_n": 1,
            "hidden_size": 8,
            "labels_hidden_size": 4,
            "latent_dim": 2,
            "dropout": 0.1,
            "encoder_conditionality": encoder,
            "latent_conditionality": latent,
            "decoder_conditionality": decoder,
        },
    )
    log_prob_y, mu, log_var, z = model(x_encoded, labels=labels)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
        label_weights=label_weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_cvae_lstm_nudger_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    labels = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = CVAEContLSTMNudger(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        labels_size=10,
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_encoded, labels=labels)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_adv_discriminator_forward():
    z = torch.randn(3, 2)
    model = Discriminator(latent_dim=2, hidden_size=2, output_size=10)
    probs = model(z)
    assert probs.shape == (3, 10)


def test_cvae_adv_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    labels = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    batch = (x_encoded, weights), (x_encoded, weights), (labels, None)
    model = CVAEContLSTMNudgerAdversarial(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        labels_size=10,
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    x_out, preds, zs, labels_out = model.predict_step(batch)
    assert x_out.shape == (3, 10, 2)
    assert preds.shape == (3, 10, 6)
    assert labels_out.shape == (3, 10)
    assert zs.shape == (3, 2)


def test_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAEContLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_encoded)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_cnn_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAEContCNN2D(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_layers": [16, 8], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_encoded)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


@pytest.mark.parametrize(
    "length,encodings,kernel,stride,padding",
    [
        (10, 6, 2, 2, 1),
        (10, 6, 2, 2, 0),
        (10, 6, 3, 2, 1),
        (10, 6, 3, 2, 0),
        (11, 6, 2, 2, 1),
        (11, 6, 2, 2, 0),
        (11, 6, 3, 2, 1),
        (11, 6, 3, 2, 0),
    ],
)
def test_cnn1d_forward(length, encodings, kernel, stride, padding):
    N = 3
    x = torch.randn(N, length, encodings + 1)  # (batch, steps, acts+1)
    weights = (torch.ones((N, length)), torch.ones((N, 1)))
    acts, durations = x.split([encodings, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAEContCNN1D(
        in_shape=x_encoded[0].shape,
        encodings=encodings,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": [16, 8],
            "latent_dim": 2,
            "dropout": 0.1,
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
        },
    )
    log_prob_y, mu, log_var, z = model(x_encoded)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_fc_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = (torch.ones((3, 10)), torch.ones((3, 1)))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAEContFC(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_layers": [16, 8], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_encoded)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        weights=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses
