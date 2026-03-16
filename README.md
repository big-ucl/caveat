<!--- the "--8<--" html comments define what part of the README to add to the index page of the documentation -->
<!--- --8<-- [start:docs] -->
![caveat](resources/logos/title.png)

Deep learning for modelling human activity schedules.

[![Daily CI Build](https://github.com/big-ucl/caveat/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/big-ucl/caveat/actions/workflows/daily-scheduled-ci.yml)
[![Documentation](https://github.com/big-ucl/caveat/actions/workflows/pages/pages-build-deployment/badge.svg)](https://big-ucl.github.io/caveat)

Caveat is for building and evaluating models of human activity schedules. Activity schedules are a useful representation of human behaviours, used for modelling transport, energy and epidemiological systems.

Activity scheduling is required component of activity-based models. But Caveat also has application for (i) diverse upsampling, (ii) bias correction, (iii) simple forecasting.

## Motivation

The core motivation or philosophy behind Caveat is that people behave in diverse ways, well beyond what is currently represented with existing modelling approaches. If you often grumble about existing human behavioral models being *too complex*, *slow* or *unrealistic* then Caveat may be of interest. We introduce generative modelling ideas (stollen perhaps from the fancy genAI domain but scaled down and carefully crafted for human modelling) so that we can better represent real human behavioral diversity.

We also add focus on the distribution of outputs rather than simple key metrics or expectations. If you think distributions matter, that everyone should be fairly represented, then you might like our work.

Also, with access to even a *somewhat* modern GPU, our models can be trained in minutes and take seconds to run for national scale populations.

## Research

Caveat is part of an ongoing research project. Overview of presented work can be found in the [papers](https://github.com/big-ucl/caveat/blob/main/papers/README.md) module, including key information allowing presented results to be **reproduced**.

We currently have published work using Caveat for generative modelling/synthesis of activity schedules - **Synthesising activity participations and scheduling with deep generative machine learning**, available [here](https://www.sciencedirect.com/science/article/pii/S0968090X25002773) or [here](https://arxiv.org/abs/2501.10221). Please consider referencing us:

```
@article{SHONE,
title = {Synthesising activity participations and scheduling with deep generative machine learning},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {179},
pages = {105273},
year = {2025},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2025.105273},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X25002773},
author = {Fred Shone and Tim Hillel},
}
```

## Example Applications

**Generation** for up-sampling or anonymisation an observed population of activity schedules:
- `caveat run configs/example_run.yaml` - train a generative model using an observed population of schedules, sample a new synthetic population of schedules, and evaluate it's quality.

**Conditional Generation** for bias correction and simple forecasting or modelling from an observed population of schedules and attributes:
- `caveat run configs/example_run_conditional.yaml` - train a model using an observed population of schedules with attributes, sample a new population conditional on new person attributes, and evaluate it's quality.

## Quick Start

Once [installed](#installation) get started using `caveat --help`.

Caveat provides various commands to facilitate rapid and reproducible experiments. Outputs are written to the (default) `logs` directory.

### Run

`caveat run --help`

Train and evaluate a model. The run data, encoder, model and other parameters are controlled using a run config. For example:

- `caveat run configs/example_run.yaml`
- `caveat run configs/example_run_conditional.yaml` (conditional generation)

### Batch

`caveat batch --help`

Batch allows training and comparison of multiple models and/or hyper-params as per a batch config. For example:

- `caveat batch configs/example_batch.yaml`
- `caveat batch configs/example_batch_conditional.yaml` (conditional generation)

### Nrun

`caveat nrun --help`

Nrun is a simplified version of batch used to repeat the same model training and evaluation. This is intended to test for variance in model training and sampling. For example, run and evaluate the variance of _n=3_ of the same experiment using:

- `caveat nrun configs/example_run.yaml --n 3`

The config is as per a regular run config but `seed` is ignored.

### Ngen

`caveat ngen --help`

As per nrun but only assesses variance from the sampling/generative process (not model training):

- `caveat ngen configs/example_run.yaml --n 3`

### Evaluate

`caveat eval --help`

Evaluate the outputs of an existing run (or batch using `-b`):

- `caveat eval configs/example_run.yaml`
- `caveat eval configs/example_batch_conditional.yaml -b` (conditional evaluation of batch)

### Tune

`caveat tune --help`

Carry out hyper-parameter tuning using optuna. See the example config `config/example_tune.yaml`. Tuning runs are recorded in `optuna.db` and can be reviewed via [optuna-dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html).

### Label, joint and multi-model models

We have some fancy model variations for predicting labels from schedules, joint generation of schedules and associated labels, and multi-model runs (segmenting and training a model per label).

`caveat lrun --help`

`caveat jrun --help`

`caveat jbatch --help`

`caveat mmrun --help`

### Logging

Caveat writes tensorboard logs to a (default) `logs/` directory. Monitor or review training progress using `tensorboard --logdir=logs`.

## Configuration

For help with configuration refer to our [documentation](https://big-ucl.github.io/caveat/latest/configuration).

## Installation

To install caveat, we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager:

<!--- --8<-- [start:docs-install-dev] -->
``` shell
git clone git@github.com:big-ucl/caveat.git
cd caveat
mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch --file requirements/base.txt --file requirements/dev.txt
mamba activate caveat
pip install -r requirements/pip.txt
pip install --no-deps -e .
```

Caveat is in development, hence an "editable" (`-e`) install is recommended.

### Jupyter Notebooks

To run the example notebooks you will need to add a ipython kernel into the mamba environemnt: `ipython kernel install --user --name=caveat`.

### GPU?
Feeling slow? Maybe you're not utilising your GPU. Torch is no longer keeping its conda channel up to date! So it's possible that you need to pip install.

First check if your GPU is available to pytorch, for example:

'''{python}
import torch
assert torch.cuda.is_available()
'''

If not then you can pip install (but do so after activating the mamba env) using commands from [here](https://pytorch.org/get-started/locally/).

### Windows Installation with CUDA

Based on the above the following may be redundant now.

If you want to get a cuda enabled windows install you can try the following mamba create:
```
mamba create -n caveat -c conda-forge -c city-modelling-lab -c pytorch -c nvidia --file requirements/cuda_base.txt --file requirements/dev.txt
```
Or lake a look [here](https://pytorch.org/get-started/locally/). Note that you need to have the right version of python.
<!--- --8<-- [end:docs-install-dev] -->
For more detailed instructions, see our [documentation](https://big-ucl.github.io/caveat/latest/installation/).

### Optuna
Optuna is also a bit finickety. Specifically it seems to be ahead of the grpcio version available on conda-forge. Breaking the mamba build. Current work around is to pip install within the mamba env `pip install grpcio==1.70.0`. To make sure that this version is used, make sure to use the dev build which includes a conda-forge pip, ensuring everything is discovered correctly.

## Contributing

Note that caveat is work in progress, we maintain example config files but these can be affected by breaking changes. If you find an issue with an example config please report it.

See our [documentation](https://big-ucl.github.io/caveat/latest/contributing/) for additional guides on contribution inclusing reporting problems.

## Building the documentation

If you are unable to access the online documentation, you can build the documentation locally.
First, [install a development environment of caveat](https://big-ucl.github.io/caveat/latest/installation/), then deploy the documentation using [mike](https://github.com/jimporter/mike):

```
mike deploy develop
mike serve
```

Then you can view the documentation in a browser at http://localhost:8000/.


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [arup-group/cookiecutter-pypackage](https://github.com/arup-group/cookiecutter-pypackage) project template.