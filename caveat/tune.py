import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from pandas import DataFrame
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from caveat import data, encoding, label_encoding, models
from caveat.data.module import DataModule
from caveat.encoding import BaseDataset, BaseEncoder
from caveat.evaluate import evaluate

# def tune_command(
#     config: dict,
#     verbose: bool = False,
#     gen: bool = True,
#     test: bool = False,
#     infer=True,
# ) -> None:
#     """
#     Runs the training and reporting process using the provided configuration.

#     https://docs.ray.io/en/latest/train/examples/lightning/lightning_cola_advanced.html?_gl=1*1fbd0te*_up*MQ..*_ga*MTYxMjAzMDY5NC4xNzM3NzI2NDU2*_ga_0LCWHW1N3S*MTczNzcyNjQ1Ni4xLjAuMTczNzcyNjQ1Ni4wLjAuMA..

#     Args:
#         config (dict): A dictionary containing the configuration parameters.

#     Returns:
#         None
#     """
#     logger_params = config.get("logging_params", {})
#     log_dir = Path(logger_params.get("log_dir", "logs"))
#     name = str(
#         logger_params.get(
#             "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         )
#     )
#     logger = initiate_logger(log_dir, name)
#     seed = config.pop("seed", seeder())

#     # load data
#     input_schedules, input_attributes, synthetic_attributes = load_data(config)

#     # encode data
#     label_encoder, encoded_labels, label_weights = encode_input_attributes(
#         logger.log_dir, input_attributes, config
#     )

#     schedule_encoder, encoded_schedules, data_loader = encode_schedules(
#         logger.log_dir, input_schedules, encoded_labels, label_weights, config
#     )

#     # train

#     def train_func(tune_config):

#         torch.manual_seed(seed)

#         config["experiment_params"]["LR"] = tune_config["LR"]
#         config["loader_params"]["train_batch_size"] = tune_config["batch_size"]
#         config["loader_params"]["val_batch_size"] = tune_config["batch_size"]

#         label_kwargs = label_encoder.label_kwargs if label_encoder else {}
#         model = build_model(encoded_schedules, config, test, gen, label_kwargs)
#         trainer = build_trainer(logger=logger, config=config)
#         trainer = prepare_trainer(trainer)
#         trainer.fit(model, datamodule=data_loader)

#     def tune_func(search_space):

#         # The maximum training epochs
#         num_epochs = 100
#         # Number of sampls from parameter space
#         num_samples = 10
#         scheduler = ASHAScheduler(
#             max_t=num_epochs, grace_period=1, reduction_factor=2
#         )

#         scaling_config = ScalingConfig(
#             num_workers=1,
#             use_gpu=True,
#             resources_per_worker={"CPU": 1, "GPU": 1},
#         )

#         run_config = RunConfig(
#             checkpoint_config=CheckpointConfig(
#                 num_to_keep=2,
#                 checkpoint_score_attribute="val_loss",
#                 checkpoint_score_order="min",
#             )
#         )
#         # Define a TorchTrainer without hyper-parameters for Tuner
#         ray_trainer = TorchTrainer(
#             train_func, scaling_config=scaling_config, run_config=run_config
#         )

#         tuner = tune.Tuner(
#             ray_trainer,
#             param_space={"train_loop_config": search_space},
#             tune_config=tune.TuneConfig(
#                 metric="val_loss",
#                 mode="min",
#                 num_samples=num_samples,
#                 scheduler=scheduler,
#             ),
#         )

#         tuner.fit()

#     search_space = {
#         "LR": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),
#     }

#     results = tune_func(search_space=search_space)
#     print(results.get_best_result(metric="val_loss", mode="max"))

# if test:
#     # test the model
#     run_test(
#         trainer=trainer,
#         schedule_encoder=schedule_encoder,
#         write_dir=Path(logger.log_dir),
#         seed=seed,
#     )

# if infer:
#     test_infer_path = Path(f"{logger.log_dir}/test_inference")
#     test_infer_path.mkdir(exist_ok=True, parents=True)

#     test_inference(
#         trainer=trainer,
#         schedule_encoder=schedule_encoder,
#         attribute_encoder=label_encoder,
#         write_dir=test_infer_path,
#         seed=seed,
#     )

# if gen:
#     # prepare synthetic attributes
#     if synthetic_attributes is not None:
#         synthetic_population, _ = label_encoder.encode(synthetic_attributes)
#     else:
#         synthetic_population = input_schedules.pid.nunique()

#     # generate synthetic schedules
#     synthetic_schedules, _, _ = generate(
#         trainer=trainer,
#         population=synthetic_population,
#         schedule_encoder=schedule_encoder,
#         attribute_encoder=label_encoder,
#         config=config,
#         write_dir=Path(logger.log_dir),
#         seed=seed,
#     )

#     # evaluate synthetic schedules
#     evaluate_synthetics(
#         synthetic_schedules={name: synthetic_schedules},
#         synthetic_attributes={name: synthetic_attributes},
#         default_eval_schedules=input_schedules,
#         default_eval_attributes=input_attributes,
#         write_path=Path(logger.log_dir),
#         eval_params=config.get("evaluation_params", {}),
#         stats=False,
#         verbose=verbose,
#     )


def load_labels(path):
    synthetic_attributes = pd.read_csv(path)
    print(f"> Loaded {len(synthetic_attributes)} synthetic labels from {path}")
    if synthetic_attributes.empty:
        raise UserWarning(f"No labels found in {path}.")
    return synthetic_attributes


def batch_eval_command(
    batch_config: dict,
    schedules_name: str = "synthetic_schedules.csv",
    labels_name: Optional[str] = None,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    """
    Runs a batch of evaluation based on the provided configuration.

    Args:
        batch_config (dict[dict]): A dictionary containing the configuration for each training job.

    Returns:
        None
    """
    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params")
    name = str(logger_params.get("name"))
    batch_dir = Path(logger_params.get("log_dir"), name)

    synthetic_schedules_all = {}
    synthetic_labels_all = {}

    for name, config in batch_config.items():
        name = str(name)
        print(f"\n======= Loading {name} =======")
        log_dir = batch_dir / name

        # build config for this run
        combined_config = global_config.copy()
        combined_config.update(config)

        # load data
        input_schedules, input_attributes, synthetic_labels = load_data(
            combined_config, verbose=verbose
        )
        print(
            f"> Loaded {input_schedules.pid.nunique()} target schedules for evaluation"
        )
        print(
            f"> Loaded {input_attributes.pid.nunique()} target attributes for evaluation"
        )

        # get most recent version
        version = sorted([d for d in log_dir.iterdir() if d.is_dir()])[-1]
        outputs_dir = log_dir / version.name
        schedules_path = outputs_dir / schedules_name
        synthetic_schedules_all[log_dir.name] = (
            data.load_and_validate_schedules(schedules_path)
        )
        print(
            f"> Loaded {synthetic_schedules_all[log_dir.name].pid.nunique()} synthetic schedules from {schedules_path}"
        )

        synthetic_labels_path = (
            outputs_dir / "synthetic_labels.csv"
        )  # todo: make this consistent across all models
        synthetic_attributes_path = outputs_dir / "synthetic_attributes.csv"
        if labels_name is not None:
            synthetic_labels_path = outputs_dir / labels_name
            synthetic_labels = load_labels(synthetic_labels_path)

        elif synthetic_labels_path.exists():
            synthetic_labels = load_labels(synthetic_labels_path)

        elif synthetic_attributes_path.exists():
            synthetic_labels = load_labels(synthetic_attributes_path)

        synthetic_labels_all[log_dir.name] = synthetic_labels

    # evaluate synthetic schedules
    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules_all,
        synthetic_attributes=synthetic_labels_all,
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_attributes,
        write_path=batch_dir,
        eval_params=global_config.get("evaluation_params", {}),
        stats=stats,
        verbose=verbose,
    )


def report_command(
    observed_path: Path,
    log_dir: Path,
    name: str = "synthetic_schedules.csv",
    verbose: bool = False,
    head: int = 10,
    batch: bool = False,
    stats: bool = True,
):
    observed_path = Path(observed_path)
    log_dir = Path(log_dir)
    observed = data.load_and_validate_schedules(observed_path)
    synthetic_schedules = {}
    if batch:
        paths = [p for p in log_dir.iterdir() if p.is_dir()]
    else:
        paths = [log_dir]

    for experiment in paths:
        # get most recent version
        version = sorted([d for d in experiment.iterdir() if d.is_dir()])[-1]
        path = experiment / version.name / name
        synthetic_schedules[experiment.name] = data.load_and_validate_schedules(
            path
        )

    reports = evaluate.evaluate(
        target_schedules=observed,
        synthetic_schedules=synthetic_schedules,
        report_stats=stats,
    )
    evaluate.report(reports, log_dir=log_dir, head=head, verbose=verbose)


def load_data(
    config: dict, verbose: bool = False
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    # load schedules data
    schedules_path = Path(config["schedules_path"])
    schedules = data.load_and_validate_schedules(schedules_path)
    if verbose:
        print(
            f"Loaded {schedules.pid.nunique()} schedules from {schedules_path}"
        )

    # load attributes data (conditional case)
    attributes, synthetic_attributes = data.load_and_validate_attributes(
        config, schedules, verbose=verbose
    )
    return schedules, attributes, synthetic_attributes


def encode_schedules(
    log_dir: Path,
    schedules: DataFrame,
    attributes: Optional[Tensor],
    label_weights: Optional[Tensor],
    config: dict,
) -> Tuple[BaseEncoder, BaseDataset, DataModule]:

    # encode schedules
    schedule_encoder = build_encoder(config)
    encoded_schedules = schedule_encoder.encode(
        schedules=schedules, labels=attributes, label_weights=label_weights
    )
    data_loader = build_dataloader(config, encoded_schedules)

    pickle.dump(schedule_encoder, open(f"{log_dir}/schedule_encoder.pkl", "wb"))

    return (schedule_encoder, encoded_schedules, data_loader)


def encode_input_attributes(
    log_dir: Path, input_attributes: Optional[DataFrame], config: dict
) -> Tuple[BaseEncoder, BaseDataset, DataModule, Tensor]:
    attribute_encoder = None
    # optionally encode attributes
    encoded_attributes = None
    weights = None
    if input_attributes is not None:
        conditionals_config = config.get("conditionals", None)
        if conditionals_config is None:
            raise UserWarning("Config must contain conditionals configuration.")

        encoder_name = config.get("attribute_encoder", "onehot")
        attribute_encoder = label_encoding.library[encoder_name](
            conditionals_config
        )
        encoded_attributes, weights = attribute_encoder.encode(input_attributes)

    pickle.dump(
        attribute_encoder, open(f"{log_dir}/attribute_encoder.pkl", "wb")
    )

    return (attribute_encoder, encoded_attributes, weights)


# def train(
#     name: str,
#     data_loader: DataModule,
#     encoded_schedules: BaseDataset,
#     config: dict,
#     test: bool,
#     gen: bool,
#     logger: TensorBoardLogger,
#     seed: Optional[int] = None,
#     ckpt_path: Optional[Path] = None,
#     label_encoder: Optional[BaseLabelEncoder] = None,
# ) -> Tuple[Trainer, encoding.BaseEncoder]:
#     """
#     Trains a model on the observed data. Return model trainer (which includes model) and encoder.

#     Args:
#         name (str): The name of the experiment.
#         schedules (pandas.DataFrame): The "observed" population data to train the model on.
#         conditionals (pandas.DataFrame): The "conditionals" data to train the model on.
#         config (dict): A dictionary containing the configuration parameters for the experiment.
#         logger (TensorBoardLogger): Logger.

#     Returns:
#         Tuple(pytorch.Trainer, BaseEncoder).
#     """
#     print(f"\n======= Training {name} =======")

#     torch.manual_seed(seed)

#     if cuda_available():
#         torch.set_float32_matmul_precision("medium")

#     torch.cuda.empty_cache()
#     if ckpt_path is not None:
#         experiment = load_model(ckpt_path, config)
#     else:
#         label_kwargs = label_encoder.label_kwargs if label_encoder else {}
#         experiment = build_model(
#             encoded_schedules, config, test, gen, label_kwargs
#         )
#     trainer = build_trainer(logger, config)
#     trainer.fit(experiment, datamodule=data_loader)
#     return trainer


def run_test(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    print("\n======= Testing =======")
    if ckpt_path is None:
        ckpt_path = "best"
    trainer.test(ckpt_path=ckpt_path, datamodule=trainer.datamodule)
    (test_in, test_target, conditionals, predictions) = zip(
        *list(
            trainer.predict(
                ckpt_path="best",
                dataloaders=trainer.datamodule.test_dataloader(),
            )
        )
    )
    test_in = torch.concat(test_in)
    test_target = torch.concat(test_target)
    conditionals = torch.concat(conditionals)
    predictions = torch.concat(predictions)

    test_in = schedule_encoder.decode_input(schedules=test_in)
    data.validate_schedules(test_in)
    test_in.to_csv(write_dir / "test_input.csv")

    test_target = schedule_encoder.decode_target(schedules=test_target)
    test_target.to_csv(write_dir / "test_target.csv")

    predictions = schedule_encoder.decode_output(schedules=predictions)
    predictions.to_csv(write_dir / "pred.csv")

    return test_in, test_target, predictions


def test_inference(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    if ckpt_path is None:
        ckpt_path = "best"

    print("\n======= Testing Inference =======")
    inference = trainer.predict(
        ckpt_path=ckpt_path, dataloaders=trainer.datamodule.test_dataloader()
    )
    input_schedules, inferred_schedules, zs, conditionals = zip(*inference)

    input_schedules = torch.concat(input_schedules)
    inferred_schedules = torch.concat(inferred_schedules)
    zs = torch.concat(zs)
    conditionals = torch.concat(conditionals)

    input_schedules = schedule_encoder.decode(input_schedules, argmax=False)
    data.validate_schedules(input_schedules)
    input_schedules.to_csv(write_dir / "input_schedules.csv")

    inferred_schedules = schedule_encoder.decode(inferred_schedules)
    data.validate_schedules(inferred_schedules)
    inferred_schedules.to_csv(write_dir / "inferred_schedules.csv")

    DataFrame(zs.cpu().numpy()).to_csv(
        Path(write_dir, "zs.csv"), index=False, header=False
    )

    if attribute_encoder is not None:
        attributes = attribute_encoder.decode(conditionals)
        attributes.to_csv(write_dir / "input_attributes.csv")


def generate(
    trainer: Trainer,
    population: Union[int, Tensor],
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
) -> DataFrame:
    torch.manual_seed(seed)
    if ckpt_path is None:
        ckpt_path = "best"
    latent_dims = config.get("model_params", {}).get("latent_dim")
    if latent_dims is None:
        latent_dims = config.get("experiment_params", {}).get("latent_dims", 2)
        # default of 2
    batch_size = config.get("generator_params", {}).get("batch_size", 256)

    if isinstance(population, int):
        print(f"\n======= Sampling {population} new schedules =======")
        synthetic_schedules, zs = generate_n(
            trainer,
            n=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
        )
        synthetic_attributes = None
    elif isinstance(population, Tensor):
        print(
            f"\n======= Sampling {len(population)} new schedules from synthetic attributes ======="
        )
        synthetic_attributes, synthetic_schedules, zs = (
            generate_from_attributes(
                trainer,
                attributes=population,
                batch_size=batch_size,
                latent_dims=latent_dims,
                seed=seed,
                ckpt_path=ckpt_path,
            )
        )
        synthetic_attributes = attribute_encoder.decode(synthetic_attributes)
        synthetic_attributes.to_csv(write_dir / "synthetic_attributes.csv")

    synthetic_schedules = schedule_encoder.decode(schedules=synthetic_schedules)
    data.validate_schedules(synthetic_schedules)
    synthetic_schedules.to_csv(write_dir / "synthetic_schedules.csv")
    DataFrame(zs.cpu().numpy()).to_csv(
        Path(write_dir, "synthetic_zs.csv"), index=False, header=False
    )
    return synthetic_schedules, synthetic_attributes, zs


def generate_n(
    trainer: Trainer,
    n: int,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
) -> torch.Tensor:
    torch.manual_seed(seed)
    dataloaders = data.build_latent_dataloader(n, latent_dims, batch_size)
    synth = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    _, synthetic_schedules, zs = zip(*synth)
    synthetic_schedules = torch.concat(synthetic_schedules)
    zs = torch.concat(zs)
    return synthetic_schedules, zs


def generate_from_attributes(
    trainer: Trainer,
    attributes: Tensor,
    batch_size: int,
    latent_dims: int,
    seed: int,
    ckpt_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    dataloaders = data.build_latent_conditional_dataloader(
        attributes, latent_dims, batch_size
    )
    synth = trainer.predict(ckpt_path=ckpt_path, dataloaders=dataloaders)
    synthetic_attributes, synthetic_schedules, zs = zip(*synth)
    synthetic_attributes = torch.concat(synthetic_attributes)
    synthetic_schedules = torch.concat(synthetic_schedules)
    zs = torch.concat(zs)
    return synthetic_attributes, synthetic_schedules, zs


def evaluate_synthetics(
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    default_eval_schedules: DataFrame,
    default_eval_attributes: DataFrame,
    write_path: Path,
    eval_params: dict,
    stats: bool = True,
    verbose: bool = False,
) -> None:
    print("\n======= Evaluating synthetic schedules =======")
    head = eval_params.get("head", 10)

    eval_schedules_path = eval_params.get("schedules_path", None)
    if eval_schedules_path:
        eval_schedules = data.load_and_validate_schedules(eval_schedules_path)
        print(
            f"<!> Loaded {len(eval_schedules)} schedules for evaluation from {eval_schedules_path}"
        )
    else:
        eval_schedules = default_eval_schedules
        print("Evaluating synthetic schedules against target schedules")

    split_on = eval_params.get("split_on", [])
    if split_on:
        print(f"Splitting evaluation on {split_on}")
        eval_attributes_path = eval_params.get("attributes_path", None)
        if eval_attributes_path:
            eval_attributes = data.load_and_validate_attributes(
                eval_params, eval_schedules
            )
            print(
                f"<!> Loaded {len(eval_attributes)} attributes for evaluation from {eval_attributes_path}"
            )
        else:
            eval_attributes = default_eval_attributes

        sub_reports = evaluate.evaluate_subsampled(
            synthetic_schedules=synthetic_schedules,
            synthetic_attributes=synthetic_attributes,
            target_schedules=eval_schedules,
            target_attributes=eval_attributes,
            split_on=split_on,
            report_stats=stats,
        )
        evaluate.report_splits(
            sub_reports,
            log_dir=write_path,
            head=head,
            verbose=verbose,
            suffix="_subs",
            ranking=len(synthetic_schedules) > 1,
        )

    else:
        reports = evaluate.evaluate(
            target_schedules=eval_schedules,
            synthetic_schedules=synthetic_schedules,
            report_stats=stats,
        )
        evaluate.report(
            reports,
            log_dir=write_path,
            head=head,
            verbose=verbose,
            ranking=len(synthetic_schedules) > 1,
        )


def conditional_sample(
    trainer: Trainer,
    population_size: int,
    data_encoder: encoding.BaseEncoder,
    config: dict,
    write_dir: Path,
    seed: int,
) -> DataFrame:
    torch.manual_seed(seed)
    print("\n======= Sampling =======")
    predict_loader = data.build_latent_dataloader(
        population_size, config["model_params"]["latent_dim"], 256
    )
    predictions = trainer.predict(ckpt_path="best", dataloaders=predict_loader)
    predictions = torch.concat(predictions)  # type: ignore
    synthetic = data_encoder.decode(schedules=predictions)
    data.validate_schedules(synthetic)
    synthesis_path = write_dir / "synthetic.csv"
    synthetic.to_csv(synthesis_path)
    return synthetic


def build_encoder(config: dict) -> encoding.BaseEncoder:
    encoder_name = config["encoder_params"]["name"]
    data_encoder = encoding.library[encoder_name](**config["encoder_params"])
    return data_encoder


def build_dataloader(
    config: dict, dataset: encoding.BaseDataset
) -> data.DataModule:
    data_loader_params = config.get("loader_params", {})
    datamodule = data.DataModule(data=dataset, **data_loader_params)
    datamodule.setup()
    return datamodule


def build_model(
    dataset: encoding.BaseDataset,
    config: dict,
    test: bool,
    gen: bool,
    label_kwargs: dict,
) -> LightningModule:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    model = model(
        in_shape=dataset.shape(),
        encodings=dataset.activity_encodings,
        encoding_weights=dataset.encoding_weights,
        conditionals_size=dataset.labels_shape,
        **config["model_params"],
        test=test,
        gen=gen,
        **config.get("experiment_params", {}),
        **label_kwargs,
    )
    return model


def load_model(ckpt_path: Path, config: dict) -> LightningModule:
    model_name = config["model_params"]["name"]
    model = models.library[model_name]
    return model.load_from_checkpoint(ckpt_path)


# def build_trainer(logger: TensorBoardLogger, config: dict) -> Trainer:

#     if cuda_available():
#         torch.set_float32_matmul_precision("medium")

#     torch.cuda.empty_cache()
#     trainer_config = config.get("trainer_params", {})
#     patience = trainer_config.pop("patience", 5)
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=Path(logger.log_dir, "checkpoints"),
#         monitor="val_loss",
#         save_top_k=2,
#         save_weights_only=False,
#     )
#     loss_scheduling = trainer_config.pop("loss_scheduling", {})
#     custom_loss_scheduler = LinearLossScheduler(loss_scheduling)
#     return Trainer(
#         logger=logger,
#         devices="auto",
#         accelerator="auto",
#         strategy=RayDDPStrategy(),
#         callbacks=[
#             RayTrainReportCallback(),
#             EarlyStopping(
#                 monitor="val_loss", patience=patience, stopping_threshold=0.0
#             ),
#             LearningRateMonitor(),
#             checkpoint_callback,
#             custom_loss_scheduler,
#         ],
#         plugins=[RayLightningEnvironment()],
#         enable_progress_bar=False,
#         **trainer_config,
#     )


def initiate_logger(save_dir: Union[Path, str], name: str) -> TensorBoardLogger:
    """
    Initializes a TensorBoardLogger object for logging training progress.

    Args:
        save_dir (str): The directory where the logs will be saved.
        name (str): The name of the logger.

    Returns:
        TensorBoardLogger: The initialized TensorBoardLogger object.
    """
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(
        exist_ok=True, parents=True
    )
    Path(f"{tb_logger.log_dir}/val_z").mkdir(exist_ok=True, parents=True)
    return tb_logger
