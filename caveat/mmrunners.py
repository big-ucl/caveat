import datetime
from pathlib import Path

import pandas as pd
from torch import concat as torch_concat
from torch.random import seed as seeder

from caveat.runners import (
    build_dataloader,
    encode_input_labels,
    encode_schedules,
    evaluate_synthetics,
    generate,
    initiate_logger,
    load_data,
    run_test,
    test_inference,
    train,
)


def mmrun_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
    warm_start: bool = True,
) -> None:
    """
    Runs the training for multi-model variation.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """

    conditionals = config.get("conditionals", None)

    if conditionals is None:
        raise ValueError("No conditionals provided in the config.")

    # check conditional encodings
    for cond, encoding in conditionals.items():
        if not encoding == "nominal":
            raise ValueError(
                f"{cond} encoding not supported. Only nominal encoding is supported for conditional multi-model training."
            )

    logger_params = config.get("logging_params", {})
    name = str(
        logger_params.get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_root = Path(logger_params.get("log_dir", "logs"), name)

    seed = config.pop("seed", seeder())

    # load data
    input_schedules, input_attributes, synthetic_attributes = load_data(config)

    # encode data
    base_logger = initiate_logger(logger_params.get("log_dir", "logs"), name)
    attribute_encoder, encoded_labels, label_weights = encode_input_labels(
        base_logger.log_dir, input_attributes, config
    )

    schedule_encoder, encoded_schedules, data_loader = encode_schedules(
        base_logger.log_dir,
        input_schedules.copy(),
        encoded_labels,
        label_weights,
        config,
    )

    if warm_start:

        logger = initiate_logger(log_root, "warm_start")

        # train
        warming_trainer = train(
            name=name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            label_encoder=attribute_encoder,
            config=config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
            ckpt_path=None,
        )

        if test:
            # test the model
            run_test(
                trainer=warming_trainer,
                schedule_encoder=schedule_encoder,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )

        if infer:
            test_infer_path = Path(f"{logger.log_dir}/test_inference")
            test_infer_path.mkdir(exist_ok=True, parents=True)
            test_inference(
                trainer=warming_trainer,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                write_dir=test_infer_path,
                seed=seed,
            )

    columns = list(conditionals.keys())
    attributes_filtered = filter_attributes_on_conditionals(
        input_attributes, columns
    )
    synthetic_attributes_filtered = filter_attributes_on_conditionals(
        synthetic_attributes, columns
    )

    combined_schedules = []
    combined_attributes = []
    combined_zs = []

    # loop through sub models
    for keys, sub_attributes in attributes_filtered.items():

        name = "_".join([f"{k}-{v}" for k, v in zip(columns, keys)])
        sub_schedules = filter_schedules_on_attributes(
            sub_attributes, input_schedules.copy()
        )
        if len(sub_schedules) == 0:
            raise ValueError(f"No schedules found for {name}.")
        print(f"Found {sub_schedules.pid.nunique()} schedules for {name}.")

        logger = initiate_logger(log_root, name)

        # encode data
        attribute_encoder, encoded_labels = encode_input_labels(
            logger.log_dir, sub_attributes, config
        )

        encoded_schedules = schedule_encoder.encode(
            schedules=sub_schedules, labels=encoded_labels
        )
        data_loader = build_dataloader(config, encoded_schedules)

        # train
        trainer = train(
            name=name,
            data_loader=data_loader,
            encoded_schedules=encoded_schedules,
            label_encoder=attribute_encoder,
            config=config,
            test=test,
            gen=gen,
            logger=logger,
            seed=seed,
            ckpt_path=(
                warming_trainer.checkpoint_callback.best_model_path
                if warm_start
                else None
            ),
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
            sub_synthetic_attributes = synthetic_attributes_filtered.get(
                keys, None
            )
            # prepare synthetic attributes
            if synthetic_attributes is not None:
                if sub_synthetic_attributes is None:
                    print(
                        f"No synthetic attributes found for {name}. Skipping generation."
                    )
                    continue
                synthetic_population, _ = attribute_encoder.encode(
                    sub_synthetic_attributes
                )
            else:
                synthetic_population = sub_schedules.pid.nunique()

            generated_schedules, generated_attributes, zs = generate(
                trainer=trainer,
                population=synthetic_population,
                schedule_encoder=schedule_encoder,
                attribute_encoder=attribute_encoder,
                config=config,
                write_dir=Path(logger.log_dir),
                seed=seed,
            )

            # generate synthetic schedules
            combined_schedules.append(generated_schedules)
            combined_attributes.append(generated_attributes)
            combined_zs.append(zs)

    # combine synthetic schedules
    i = 0
    for sub_schedules, sub_attributes in zip(
        combined_schedules, combined_attributes
    ):
        sub_schedules.pid = sub_schedules.pid + i
        sub_attributes.pid = sub_attributes.pid + i
        i += sub_schedules.pid.max() + 1
    synthetic_schedules = pd.concat(combined_schedules)
    synthetic_attributes = pd.concat(combined_attributes)
    synthetic_zs = torch_concat(combined_zs)

    synthetic_schedules.to_csv(
        Path(base_logger.log_dir) / "synthetic_schedules.csv"
    )
    synthetic_attributes.to_csv(
        Path(base_logger.log_dir) / "synthetic_attributes.csv"
    )
    pd.DataFrame(synthetic_zs.cpu().numpy()).to_csv(
        Path(base_logger.log_dir) / "synthetic_zs.csv",
        index=False,
        header=False,
    )

    # evaluate synthetic schedules
    evaluate_synthetics(
        synthetic_schedules={name: synthetic_schedules},
        synthetic_attributes={name: synthetic_attributes},
        default_eval_schedules=input_schedules,
        default_eval_attributes=input_attributes,
        write_path=Path(logger.log_dir),
        eval_params=config.get("evaluation_params", {}),
        stats=False,
        verbose=verbose,
    )


def filter_attributes_on_conditionals(attributes, columns) -> dict:
    if attributes is None:
        return None
    filtered = {}
    values = attributes[columns].value_counts().index
    for value in values:
        selected = attributes.copy()
        for k, v in zip(columns, value):
            selected = selected[selected[k] == v]
        if len(selected) > 0:
            filtered[value] = selected

    return filtered


def filter_schedules_on_attributes(attributes, schedules) -> dict:
    pids = attributes["pid"]
    return schedules[schedules["pid"].isin(pids)]
