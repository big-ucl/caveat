import pytest
import torch

from caveat.models.sequence.auto_sequence_lstm import AutoSeqLSTM
from caveat.models.sequence.cond_sequence_lstm import CondSeqLSTM
from caveat.models.sequence.cvae_sequence_lstm import CVAESeqLSTM
from caveat.models.sequence.cvae_sequence_lstm_nudger import CVAESeqLSTMNudger
from caveat.models.sequence.cvae_sequence_lstm_nudger_adversarial import (
    CVAESeqLSTMNudgerAdversarial,
    Discriminator,
)
from caveat.models.sequence.vae_sequence_cnn import VAESeqCNN
from caveat.models.sequence.vae_sequence_fc import VAESeqFC
from caveat.models.sequence.vae_sequence_lstm import VAESeqLSTM


def test_auto_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    conditionals = torch.randn(3, 10)  # (batch, channels)
    model = AutoSeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, _, _, _ = model(x_encoded, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(
        log_probs=log_prob_y, target=x_encoded, mask=weights
    )
    assert "loss" in losses


def test_conditional_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    conditionals = torch.randn(3, 10)  # (batch, channels)
    model = CondSeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, _, _, _ = model(x_encoded, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(
        log_probs=log_prob_y, target=x_encoded, mask=weights
    )
    assert "loss" in losses


testdata = [
    ("none", "concat", "none"),
    ("none", "add", "none"),
    ("hidden", "concat", "none"),
    ("hidden", "add", "none"),
    ("inputs", "concat", "none"),
    ("inputs", "add", "none"),
    ("hidden_and_inputs", "concat", "none"),
    ("hidden_and_inputs", "add", "none"),
    ("none", "concat", "inputs"),
    ("none", "add", "inputs"),
    ("hidden", "concat", "inputs"),
    ("hidden", "add", "inputs"),
    ("inputs", "concat", "inputs"),
    ("inputs", "add", "inputs"),
    ("hidden_and_inputs", "concat", "inputs"),
    ("hidden_and_inputs", "add", "inputs"),
]


@pytest.mark.parametrize("encoder,latent,decoder", testdata)
def test_cvae_lstm_forward(encoder, latent, decoder):
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    conditionals = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = CVAESeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
            "encoder_conditionality": encoder,
            "latent_conditionality": latent,
            "decoder_conditionality": decoder,
        },
    )
    log_prob_y, mu, log_var, z = model(x_encoded, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        mask=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_cvae_lstm_nudger_forward():
    x = torch.randn(3, 10, 6)  # (batch, channels, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    conditionals = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = CVAESeqLSTMNudger(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, mu, log_var, z = model(x_encoded, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y,
        mu=mu,
        log_var=log_var,
        target=x_encoded,
        mask=weights,
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
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    labels = torch.randn(3, 10)  # (batch, channels)
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    batch = (x_encoded, weights), (x_encoded, weights), (labels, None)
    model = CVAESeqLSTMNudgerAdversarial(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    x_out, preds, zs, conditionals_out = model.predict_step(batch)
    assert x_out.shape == (3, 10, 2)
    assert preds.shape == (3, 10, 6)
    assert conditionals_out.shape == (3, 10)
    assert zs.shape == (3, 2)


def test_lstm_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAESeqLSTM(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": 1,
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
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
        mask=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_cnn_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAESeqCNN(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": [16, 8],
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
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
        mask=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_fc_forward():
    x = torch.randn(3, 10, 6)  # (batch, steps, acts+1)
    weights = torch.ones((3, 10))
    acts, durations = x.split([5, 1], dim=-1)
    acts_max = acts.argmax(dim=-1).unsqueeze(-1)
    durations = durations
    x_encoded = torch.cat([acts_max, durations], dim=-1)
    model = VAESeqFC(
        in_shape=x_encoded[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_layers": [16, 8],
            "hidden_size": 2,
            "latent_dim": 2,
            "dropout": 0.1,
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
        mask=weights,
    )
    assert "loss" in losses
    assert "recon_loss" in losses
