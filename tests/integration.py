from caveat.jrunners import jrun_command
from caveat.runners import (
    batch_command,
    ngen_command,
    nrun_command,
    run_command,
)


def test_run_conv(config_discrete_conv):
    run_command(config_discrete_conv)


def test_run_vae_lstm(config_vae_lstm):
    run_command(config_vae_lstm)


def test_run_c_lstm(config_c_lstm):
    run_command(config_c_lstm)


def test_run_cvae_lstm(config_cvae_lstm):
    run_command(config_cvae_lstm)


def test_batch_multi_model(batch_config):
    batch_command(batch_config)


def test_nrun(config_vae_lstm):
    nrun_command(config_vae_lstm, n=2)


def test_nsample(config_vae_lstm):
    ngen_command(config_vae_lstm, n=2)


def test_jrun(config_jvae_lstm):
    jrun_command(config=config_jvae_lstm)
