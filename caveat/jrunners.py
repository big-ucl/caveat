import datetime
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from pandas import DataFrame
from pytorch_lightning import Trainer
from torch import Tensor
from torch.random import seed as seeder

from caveat import data, encoding, label_encoding, samplers
from caveat.runners import (
    encode_input_labels,
    encode_schedules,
    evaluate_synthetics,
    initiate_logger,
    load_data,
    run_test,
    train,
)


def jrun_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
    sample: bool = True,
    patience: int = 8,
) -> None:
    """
    Runs the training for joint-model variation.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    labels_encoder_config = config.get("labels_encoder", {})
    labels_encoder = labels_encoder_config.get("name", None)
    if labels_encoder is None or labels_encoder != "tokens":
        raise ValueError(
            "Joint model requires attribute_encoder to be configured as 'tokens'."
        )

    labels_config = labels_encoder_config.get("labels", None)
    if labels_config is None:
        raise ValueError("No labels provided in the labels_encoder config.")

    logger_params = config.get("logging_params", {})
    log_dir = Path(logger_params.get("log_dir", "logs"))
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    logger = initiate_logger(log_dir, name)
    seed = config.pop("seed", seeder())

    # load data
    input_schedules, input_labels, synthetic_labels = load_data(config)

    # encode data
    labels_encoder, encoded_labels, label_weights = encode_input_labels(
        logger.log_dir, input_labels, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        logger.log_dir, input_schedules, encoded_labels, label_weights, config
    )

    # train
    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        label_encoder=labels_encoder,
        config=config,
        test=test,
        gen=gen,
        logger=logger,
        seed=seed,
    )

    if test:
        # test the model
        run_test(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            write_dir=Path(logger.log_dir),
            seed=seed,
        )

    if infer:
        test_infer_path = Path(f"{logger.log_dir}/test_inference")
        test_infer_path.mkdir(exist_ok=True, parents=True)

        test_inference(
            trainer=trainer,
            schedule_encoder=schedule_encoder,
            attribute_encoder=labels_encoder,
            write_dir=test_infer_path,
            seed=seed,
        )

    if gen:
        # prepare target sample size
        if synthetic_labels is not None:
            synthetic_population = len(synthetic_labels)
        else:
            synthetic_population = input_schedules.pid.nunique()

        if sample:
            synthetic_labels, synthetic_schedules = target_sample(
                conditionals=list(labels_config.keys()),
                patience=patience,
                synthetic_population=synthetic_population,
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=labels_encoder,
                config=config,
                log_dir=Path(logger.log_dir),
                seed=seed,
                synthetic_labels=synthetic_labels,
                verbose=verbose,
            )
        else:
            # generate synthetic schedules
            synthetic_schedules, synthetic_labels, _ = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                attribute_encoder=labels_encoder,
                config=config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )

        # evaluate synthetic schedules
        evaluate_synthetics(
            synthetic_schedules={name: synthetic_schedules},
            synthetic_attributes={name: synthetic_labels},
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_labels,
            write_path=Path(logger.log_dir),
            eval_params=config.get("evaluation_params", {}),
            stats=False,
            verbose=verbose,
        )


def target_sample(
    conditionals: list,
    patience: int,
    synthetic_population: int,
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    attribute_encoder: label_encoding.BaseLabelEncoder,
    config: dict,
    log_dir: Path,
    seed: int,
    synthetic_labels: Optional[DataFrame] = None,
    verbose: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    """
    Sample synthetic schedules and labels for target labels.
    Repeatedly calls the model generate function until the target labels are sampled.
    Or runs out of patience.

    Args:
        conditionals (list): _description_
        patience (int): _description_
        synthetic_population (int): _description_
        trainer (Trainer): _description_
        schedule_encoder (encoding.BaseEncoder): _description_
        attribute_encoder (attribute_encoding.BaseLabelEncoder): _description_
        config (dict): _description_
        log_dir (Path): _description_
        seed (int): _description_
        synthetic_labels (Optional[DataFrame], optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[DataFrame, DataFrame]: _description_
    """

    print(
        f"\n======= Sampling {synthetic_population} new schedules for target labels ======="
    )

    sampler = samplers.TargetLabelSampler(
        target_labels=synthetic_labels, target_columns=conditionals
    )

    for _ in range(patience):
        random.seed(seed)
        seed = random_seed()

        # generate synthetic schedules
        synthetic_schedules, synthetic_labels, _ = generate(
            trainer=trainer,
            population=synthetic_population,
            schedule_encoder=schedule_encoder,
            attribute_encoder=attribute_encoder,
            config=config,
            write_dir=log_dir,
            seed=seed,
        )
        sampler.sample(synthetic_labels, synthetic_schedules)
        sampler.print(verbose=verbose)
        if sampler.is_done():
            break
    sampler.print(verbose=verbose)
    return sampler.finish()


def random_seed():
    return random.randint(1000, 1000000000)


def jbatch_command(
    batch_config: dict,
    stats: bool = False,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
    sample: bool = True,
    patience: int = 8,
) -> None:
    """
    Batch runs the training for joint-model variation.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """

    global_config = batch_config.pop("global")
    logger_params = global_config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(logger_params.get("log_dir", "logs"), name)

    synthetic_schedules_all = {}
    synthetic_attributes_all = {}

    for name, config in batch_config.items():
        name = str(name)
        logger = initiate_logger(log_dir, name)

        # build config for this run
        combined_config = global_config.copy()
        combined_config.update(config)
        seed = combined_config.pop("seed", seeder())

        attribute_encoder = combined_config.get("attribute_encoder", None)
        if attribute_encoder is None or attribute_encoder != "tokens":
            raise ValueError(
                "Joint model requires attribute_encoder to be configured as 'tokens'."
            )

        conditionals = combined_config.get("conditionals", None)
        if conditionals is None:
            raise ValueError("No conditionals provided in the config.")

        # load data
        input_schedules, input_labels, synthetic_labels = load_data(
            combined_config
        )

        # encode data
        attribute_encoder, encoded_labels, label_weights = encode_input_labels(
            logger.log_dir, input_labels, combined_config
        )

        schedule_encoder, encoded_schedules, data_loader = encode_schedules(
            logger.log_dir,
            input_schedules,
            encoded_labels,
            label_weights,
            combined_config,
        )

        # train
        trainer = train(
            name=name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            label_encoder=attribute_encoder,
            config=combined_config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
        )

        if test:
            # test the model
            run_test(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )

        if infer:
            test_infer_path = Path(f"{logger.log_dir}/test_inference")
            test_infer_path.mkdir(exist_ok=True, parents=True)

            test_inference(
                trainer=trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=test_infer_path,
                seed=seed,
            )

        if gen:
            # prepare synthetic sample size
            if synthetic_labels is not None:
                synthetic_population = len(synthetic_labels)
            else:
                synthetic_population = input_schedules.pid.nunique()

            if sample:
                synthetic_labels, synthetic_schedules = target_sample(
                    conditionals=list(conditionals.keys()),
                    patience=patience,
                    synthetic_population=synthetic_population,
                    trainer=trainer,
                    schedule_encoder=schedule_encoder,
                    attribute_encoder=attribute_encoder,
                    config=combined_config,
                    log_dir=Path(logger.log_dir),
                    seed=seed,
                    synthetic_labels=synthetic_labels,
                    verbose=verbose,
                )
            else:
                # generate synthetic schedules
                synthetic_schedules, synthetic_labels, _ = generate(
                    trainer=trainer,
                    population=synthetic_population,
                    schedule_encoder=schedule_encoder,
                    attribute_encoder=attribute_encoder,
                    config=combined_config,
                    write_dir=Path(logger.log_dir),
                    seed=seed,
                )
            synthetic_schedules_all[name] = synthetic_schedules
            synthetic_attributes_all[name] = synthetic_labels

    if gen:
        # evaluate synthetic schedules
        evaluate_synthetics(
            synthetic_schedules=synthetic_schedules_all,
            synthetic_attributes=synthetic_attributes_all,
            default_eval_schedules=input_schedules,
            default_eval_attributes=input_labels,
            write_path=Path(log_dir),
            eval_params=global_config.get("evaluation_params", {}),
            stats=stats,
            verbose=verbose,
        )


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
    input_schedules, inferred_schedules, input_labels, inferred_labels, zs = (
        zip(*inference)
    )

    input_schedules = torch.concat(input_schedules)
    inferred_schedules = torch.concat(inferred_schedules)
    input_labels = torch.concat(input_labels)
    inferred_labels = repack_labels(inferred_labels)
    zs = torch.concat(zs)

    input_schedules = schedule_encoder.decode(input_schedules, argmax=False)
    data.validate_schedules(input_schedules)
    input_schedules.to_csv(write_dir / "input_schedules.csv")

    inferred_schedules = schedule_encoder.decode(inferred_schedules)
    data.validate_schedules(inferred_schedules)
    inferred_schedules.to_csv(write_dir / "inferred_schedules.csv")

    if attribute_encoder is not None:
        attributes = attribute_encoder.decode(input_labels)
        attributes.to_csv(write_dir / "input_labels.csv")
        inferred_labels = attribute_encoder.sample_decode(inferred_labels)
        inferred_labels.to_csv(write_dir / "inferred_labels.csv")

    DataFrame(zs.cpu().numpy()).to_csv(
        Path(write_dir, "zs.csv"), index=False, header=False
    )


def generate(
    trainer: Trainer,
    population: int,
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
        synthetic_schedules, synthetic_labels, zs = generate_n(
            trainer,
            n=population,
            batch_size=batch_size,
            latent_dims=latent_dims,
            seed=seed,
            ckpt_path=ckpt_path,
        )
        synthetic_attributes = None

    synthetic_schedules = schedule_encoder.decode(schedules=synthetic_schedules)
    data.validate_schedules(synthetic_schedules)
    synthetic_schedules.to_csv(write_dir / "synthetic_schedules.csv")

    synthetic_attributes = attribute_encoder.sample_decode(synthetic_labels)
    synthetic_attributes.to_csv(write_dir / "synthetic_labels.csv")

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
    synthetic_schedules, synthetic_labels, zs = zip(*synth)
    synthetic_schedules = torch.concat(synthetic_schedules)
    synthetic_labels = repack_labels(synthetic_labels)
    zs = torch.concat(zs)
    return synthetic_schedules, synthetic_labels, zs


def repack_labels(batched_labels: Tuple[List[Tensor]]) -> List[Tensor]:
    batched_labels = list(batched_labels)
    if len(batched_labels) == 1:
        return batched_labels[0]
    else:
        unpacked_labels = batched_labels.pop(0)
        for batch in batched_labels:
            for i, labels in enumerate(batch):
                unpacked_labels[i] = torch.concat((unpacked_labels[i], labels))
        return unpacked_labels
