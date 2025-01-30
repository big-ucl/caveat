import pytest
import torch

from caveat.models.discrete.auto_discrete_lstm import AutoDiscLSTM
from caveat.models.discrete.cond_discrete_conv import CondDiscCNN2D
from caveat.models.discrete.cond_discrete_lstm import CondDiscLSTM
from caveat.models.discrete.vae_discrete_cnn1d import VAEDiscCNN1D
from caveat.models.discrete.vae_discrete_cnn2d import VAEDiscCNN2D
from caveat.models.discrete.vae_discrete_fc import VAEDiscFC
from caveat.models.discrete.vae_discrete_lstm import VAEDiscLSTM
from caveat.models.discrete.vae_discrete_transformer import VAEDiscTrans


def test_discrete_auto_lstm_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    conditionals = torch.randn(3, 10)  # (batch, channels)
    model = AutoDiscLSTM(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob, _, _, _ = model(x_max, conditionals=conditionals)
    assert log_prob.shape == torch.Size([3, 144, 5])
    losses = model.loss_function(log_probs=log_prob, target=x_max, mask=None)
    assert "loss" in losses


def test_discrete_conditional_conv_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    conditionals = torch.randn(3, 10)  # (batch, channels)
    model = CondDiscCNN2D(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{"hidden_layers": [1], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, _, _, _ = model(x_max, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(log_probs=log_prob_y, target=x_max, mask=None)
    assert "loss" in losses


def test_discrete_conditional_lstm_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    conditionals = torch.randn(3, 10)  # (batch, channels)
    model = CondDiscLSTM(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        conditionals_size=10,
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, _, _, _ = model(x_max, conditionals=conditionals)
    assert log_prob_y.shape == x.shape
    losses = model.loss_function(log_probs=log_prob_y, target=x_max, mask=None)
    assert "loss" in losses


def test_discrete_vae_conv_forward():
    x = torch.randn(3, 1, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = VAEDiscCNN2D(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_layers": [1], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_max)
    assert log_prob_y.shape == torch.Size([3, 144, 5])
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y, mu=mu, log_var=log_var, target=x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


@pytest.mark.parametrize(
    "length,encodings,kernel,stride,padding",
    [
        (144, 6, 2, 2, 1),
        (144, 6, 2, 2, 0),
        (144, 6, 3, 2, 1),
        (144, 6, 3, 2, 0),
        (145, 6, 2, 2, 1),
        (145, 6, 2, 2, 0),
        (145, 6, 3, 2, 1),
        (145, 6, 3, 2, 0),
    ],
)
def test_discrete_vae_conv1d_forward(
    length, encodings, kernel, stride, padding
):
    x = torch.randn(3, 1, length, encodings)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = VAEDiscCNN1D(
        in_shape=x_max[0].shape,
        encodings=encodings,
        encoding_weights=torch.ones((encodings)),
        **{
            "hidden_layers": [16, 8],
            "latent_dim": 2,
            "dropout": 0.1,
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
        },
    )
    log_prob_y, mu, log_var, z = model(x_max)
    assert log_prob_y.shape == torch.Size([3, length, encodings])
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y, mu=mu, log_var=log_var, target=x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_discrete_vae_fc_forward():
    length = 144
    encodings = 10
    x = torch.randn(3, 1, length, encodings)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = VAEDiscFC(
        in_shape=x_max[0].shape,
        encodings=encodings,
        encoding_weights=torch.ones((encodings)),
        **{"hidden_layers": [16, 8], "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_max)
    assert log_prob_y.shape == torch.Size([3, length, encodings])
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y, mu=mu, log_var=log_var, target=x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_discrete_vae_lstm_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = VAEDiscLSTM(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{"hidden_n": 1, "hidden_size": 2, "latent_dim": 2, "dropout": 0.1},
    )
    log_prob_y, mu, log_var, z = model(x_max)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y, mu=mu, log_var=log_var, target=x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses


def test_discrete_transformer_forward():
    x = torch.randn(3, 144, 5)  # (batch, channels, steps, acts)
    x_max = x.argmax(dim=-1).squeeze()
    model = VAEDiscTrans(
        in_shape=x_max[0].shape,
        encodings=5,
        encoding_weights=torch.ones((5)),
        **{
            "hidden_n": 1,
            "hidden_size": 2,
            "heads": 1,
            "latent_dim": 2,
            "dropout": 0.1,
        },
    )
    log_prob_y, mu, log_var, z = model(x_max)
    assert log_prob_y.shape == x.shape
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert z.shape == (3, 2)
    losses = model.loss_function(
        log_probs=log_prob_y, mu=mu, log_var=log_var, target=x_max, mask=None
    )
    assert "loss" in losses
    assert "recon_loss" in losses
