import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torch.random import seed as seeder

from caveat import data, encoding, label_encoding
from caveat.runners import (
    encode_input_labels,
    encode_schedules,
    initiate_logger,
    load_data,
    train,
)


def label_run_command(
    config: dict, verbose: bool = False, gen: bool = True, test: bool = False
) -> None:
    """
    Runs the training for label prediction.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """
    label_encoder = config.get("attribute_encoder", None)
    if label_encoder is None or label_encoder != "tokens":
        raise ValueError(
            "Joint model requires attribute_encoder to be configured as 'tokens'."
        )

    conditionals = config.get("conditionals", None)
    if conditionals is None:
        raise ValueError("No conditionals provided in the config.")

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
    input_schedules, input_labels, _ = load_data(config)

    # encode data
    label_encoder, encoded_labels, label_weights = encode_input_labels(
        logger.log_dir, input_labels, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        logger.log_dir, input_schedules, encoded_labels, label_weights, config
    )

    trainer = train(
        name=name,
        data_loader=data_loader,
        encoded_schedules=encoded_schedules,
        label_encoder=label_encoder,
        config=config,
        test=test,
        gen=gen,
        logger=logger,
        seed=seed,
    )

    run_test(
        trainer=trainer,
        schedule_encoder=schedule_encoder,
        label_encoder=label_encoder,
        write_dir=Path(logger.log_dir),
        seed=seed,
    )


# def label_batch_command(
#     batch_config: dict,
#     stats: bool = False,
#     verbose: bool = False,
#     gen: bool = True,
#     test: bool = False,
#     infer=True,
#     sample: bool = True,
#     patience: int = 8,
# ) -> None:
#     """
#     Batch runs the training for joint-model variation.

#     Args:
#         config (dict): A dictionary containing the configuration parameters.

#     Returns:
#         None
#     """

#     global_config = batch_config.pop("global")
#     logger_params = global_config.get("logging_params", {})
#     name = str(
#         logger_params.get(
#             "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         )
#     )
#     log_dir = Path(logger_params.get("log_dir", "logs"), name)

#     for name, config in batch_config.items():
#         name = str(name)
#         logger = initiate_logger(log_dir, name)

#         # build config for this run
#         combined_config = global_config.copy()
#         combined_config.update(config)
#         seed = combined_config.pop("seed", seeder())

#         attribute_encoder = combined_config.get("attribute_encoder", None)
#         if attribute_encoder is None or attribute_encoder != "tokens":
#             raise ValueError(
#                 "Joint model requires attribute_encoder to be configured as 'tokens'."
#             )

#         conditionals = combined_config.get("conditionals", None)
#         if conditionals is None:
#             raise ValueError("No conditionals provided in the config.")

#         # load data
#         input_schedules, input_labels, synthetic_labels = load_data(
#             combined_config
#         )

#         # encode data
#         attribute_encoder, encoded_labels, label_weights = (
#             encode_input_attributes(
#                 logger.log_dir, input_labels, combined_config
#             )
#         )

#         schedule_encoder, encoded_schedules, data_loader = encode_schedules(
#             logger.log_dir,
#             input_schedules,
#             encoded_labels,
#             label_weights,
#             combined_config,
#         )

#         # train
#         trainer = train(
#             name=name,
#             data_loader=data_loader,
#             encoded_schedules=encoded_schedules,
#             label_encoder=attribute_encoder,
#             config=combined_config,
#             test=test,
#             gen=gen,
#             logger=logger,
#             seed=seed,
#         )

#         run_test(
#             trainer=trainer,
#             label_encoder=attribute_encoder,
#             schedule_encoder=schedule_encoder,
#             write_dir=Path(logger.log_dir),
#             seed=seed,
#         )


def run_test(
    trainer: Trainer,
    schedule_encoder: encoding.BaseEncoder,
    label_encoder: label_encoding.BaseLabelEncoder,
    write_dir: Path,
    seed: int,
    ckpt_path: Optional[str] = None,
):
    torch.manual_seed(seed)
    print("\n======= Testing =======")
    if ckpt_path is None:
        ckpt_path = "best"
    trainer.test(
        ckpt_path=ckpt_path, datamodule=trainer.datamodule, verbose=True
    )
    print("\n======= Inference =======")
    inference = trainer.predict(
        ckpt_path=ckpt_path, dataloaders=trainer.datamodule.test_dataloader()
    )
    input_schedules, input_labels, inferred_labels = zip(*inference)

    input_schedules = torch.concat(input_schedules)
    input_labels = torch.concat(input_labels)
    inferred_labels = repack_labels(inferred_labels)

    input_schedules = schedule_encoder.decode(input_schedules, argmax=False)
    data.validate_schedules(input_schedules)
    input_schedules.to_csv(write_dir / "input_schedules.csv")

    attributes = label_encoder.decode(input_labels)
    attributes.to_csv(write_dir / "input_labels.csv")
    inferred_labels = label_encoder.sample_decode(inferred_labels)
    inferred_labels.to_csv(write_dir / "inferred_labels.csv")


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
