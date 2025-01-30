import copy
import datetime
from pathlib import Path

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.random import seed as seeder

from caveat import cuda_available, runners
from caveat.callbacks import LinearLossScheduler


def tune_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
) -> None:
    """
    Tune the hyperparameters of the model using optuna.

    Args:
        config (dict): The configuration dictionary.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        gen (bool, optional): Whether to generate synthetic data. Defaults to True.
        test (bool, optional): Whether to test the model. Defaults to False.
        infer (bool, optional): Whether to infer the model. Defaults to True.

    Returns:
        None

    """
    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    base_logger = runners.initiate_logger(log_dir, name)
    base_dir = base_logger.log_dir
    seed = config.pop("seed", seeder())

    # load data
    input_schedules, input_attributes, synthetic_attributes = runners.load_data(
        config
    )

    trials = config.get("tune", {}).get("trials", 20)
    prune = config.get("tune", {}).get("prune", True)
    timeout = config.get("tune", {}).get("timeout", 600)

    def objective(trial: optuna.Trial) -> float:

        torch.manual_seed(seed)
        if cuda_available():
            torch.set_float32_matmul_precision("medium")
        torch.cuda.empty_cache()

        trial_config, hyperparameters = build_config(trial, config)

        trial_name = build_trial_name(trial.number, hyperparameters)
        logger = runners.initiate_logger(base_dir, trial_name)

        # encode data
        label_encoder, encoded_labels, label_weights = (
            runners.encode_input_attributes(
                logger.log_dir, input_attributes, trial_config
            )
        )

        _, encoded_schedules, data_loader = runners.encode_schedules(
            logger.log_dir,
            input_schedules,
            encoded_labels,
            label_weights,
            trial_config,
        )

        # build model
        ckpt_path = trial_config.get("ckpt_path", None)
        if ckpt_path is not None:
            model = runners.load_model(ckpt_path, trial_config)
        else:
            label_kwargs = label_encoder.label_kwargs if label_encoder else {}
            model = runners.build_model(
                encoded_schedules, trial_config, test, gen, label_kwargs
            )

        trainer = runners.build_trainer(logger, trial_config)
        trainer.logger.log_hyperparams(hyperparameters)

        trial.set_user_attr("config", trial_config)

        trainer.fit(model, datamodule=data_loader)

        return trainer.callback_metrics["val_loss"].item()

    if prune:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    db_name = f"sqlite:///{base_dir}/optuna.sqlite3"
    print(f"Study logging to {db_name}")
    study = optuna.create_study(
        storage=db_name,
        study_name=name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
    )
    study.optimize(
        objective, n_trials=trials, timeout=timeout, callbacks=[best_callback]
    )

    config = study.user_attrs["config"]
    config["logging_params"]["log_dir"] = base_dir
    config["logging_params"]["name"] = "best_trial"

    best_trial = study.best_trial
    print("Best params:", best_trial.params)
    print("=============================================")

    runners.run_command(
        config, verbose=verbose, gen=gen, test=test, infer=infer
    )

    print("Best params:", best_trial.params)


def build_trail_trainer(
    trial: optuna.Trial, logger: TensorBoardLogger, config: dict
) -> Trainer:
    trainer_config = config.get("trainer_params", {})
    patience = trainer_config.pop("patience", 5)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=2,
        save_weights_only=False,
    )
    loss_scheduling = trainer_config.pop("loss_scheduling", {})
    custom_loss_scheduler = LinearLossScheduler(loss_scheduling)
    return Trainer(
        logger=logger,
        callbacks=[
            PyTorchLightningPruningCallback(
                trial, monitor="val_loss", mode="min"
            ),
            EarlyStopping(
                monitor="val_loss", patience=patience, stopping_threshold=0.0
            ),
            LearningRateMonitor(),
            checkpoint_callback,
            custom_loss_scheduler,
        ],
        **trainer_config,
    )


def best_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr("config", trial.user_attrs["config"])


def build_config(trial: optuna.Trial, config: dict) -> dict:
    """Iterate through the config leaves and parse the values"""
    new_config = copy.deepcopy(config)
    suggestions = {}
    new_config = build_suggestions(trial, new_config, suggestions)
    return new_config, suggestions


def build_trial_name(
    number: int, suggestions: dict, include_kvs: bool = True
) -> str:
    number_str = str(number).zfill(4)
    if include_kvs:
        kv_str = "_".join(
            [f"{skey(k)}>{svalue(v)}" for k, v in suggestions.items()]
        )
        number_str = f"{number_str}_{kv_str}"
    return number_str


def skey(key: str) -> str:
    ks = key.split("_")
    if len(ks) > 1:
        return "".join([k[0].upper() for k in ks])
    length = len(key)
    if length > 3:
        return key[:4]
    return key


def svalue(value) -> str:
    if isinstance(value, float):
        return f"{value:.2e}"
    return str(value)


def build_suggestions(trial: optuna.Trial, config: dict, suggestions: dict):
    for k, v in config.copy().items():
        if isinstance(v, dict):
            config[k] = build_suggestions(trial, v, suggestions)
        else:
            name, suggestion = parse_suggestion(trial, v)
            if name is not None:
                suggestions[name] = suggestion
                config.pop(k)
                config[k] = suggestion
    return config


def parse_suggestion(trial, value: str):
    """Parse the value and return a tuple of the name and the suggested value.
    Or return Nones if not a suggestion.
    """
    if not isinstance(value, str):
        return (None, None)
    if not value.startswith("suggest"):
        return (None, None)
    if value.startswith("suggest_int("):
        return suggest_int(trial, value)
    if value.startswith("suggest_float("):
        return suggest_float(trial, value)
    if value.startswith("suggest_categorical("):
        return suggest_categorical(trial, value)
    raise ValueError(f"Unknown suggestion type: {value}")


def suggest_int(trial: optuna.Trial, value: str):
    args = (
        value.strip().removeprefix("suggest_int(").removesuffix(")").split(",")
    )
    name = parse_name(args[0])
    low = int(args[1])
    high = int(args[2])
    kwargs = {}
    for kv in args[3:]:
        key, value = kv.split("=")
        kwargs[parse_value(key)] = parse_value(value)
    return (name, trial.suggest_int(name, low, high, **kwargs))


def suggest_float(trial: optuna.Trial, value: str):
    args = (
        value.strip()
        .removeprefix("suggest_float(")
        .removesuffix(")")
        .split(",")
    )
    name = parse_name(args[0])
    low = float(args[1])
    high = float(args[2])
    kwargs = {}
    for kv in args[3:]:
        key, value = kv.split("=")
        kwargs[parse_value(key)] = parse_value(value)
    return (name, trial.suggest_float(name, low, high, **kwargs))


def suggest_categorical(trial: optuna.Trial, value: str):
    args = (
        value.strip()
        .removeprefix("suggest_categorical(")
        .removesuffix(")")
        .split(",", 1)
    )
    name = parse_name(args[0])
    choices = args[1]
    choices = choices.strip().removeprefix("[").removesuffix("]").split(",")
    choices = [parse_value(c) for c in choices]
    return (name, trial.suggest_categorical(name, choices))


def parse_name(name: str):
    return name.strip().removeprefix('"').removesuffix('"')


def parse_value(value: str):
    cleaned = value.strip()
    if cleaned.isnumeric():
        return int(cleaned)
    if cleaned.replace(".", "", 1).isnumeric():
        return float(value)
    if cleaned == "True":
        return True
    if cleaned == "False":
        return False
    return cleaned
