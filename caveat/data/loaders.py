from pathlib import Path

import pandas as pd


def load_and_validate_schedules(data_path: Path) -> pd.DataFrame:
    """
    Load, prepare and validate schedules data from a CSV file.

    Args:
        data_path (Path): The path to the CSV file containing the schedules data.

    Returns:
        pd.DataFrame: The loaded and validated schedules data.

    Raises:
        UserWarning: If no data is found in the specified file.
    """
    data = pd.read_csv(data_path)
    if data.empty:
        raise UserWarning(f"No data found in {data_path}.")
    validate_schedules(data)
    data.act = data.act.astype("category")
    data.start = data.start.astype("int")
    data.end = data.end.astype("int")
    data["duration"] = data.end - data.start
    return data.sort_values(by=["pid", "start"])


def validate_schedules(data: pd.DataFrame):
    """
    Validate the schedules data.

    Args:
        data (pd.DataFrame): The schedules data to be validated.

    Raises:
        UserWarning: If the input data is missing required columns.
    """
    required_cols = {"pid", "act", "start", "end"}
    found = set(data.columns)
    missing = required_cols - found
    if missing:
        raise UserWarning(
            f"""
    Input data is missing required columns.
    Required: {required_cols}.
    Found: {found}.
    Please add missing: {missing}.
    """
        )


def load_and_validate_attributes(
    config: dict, schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and validate attributes data from a CSV file.

    Args:
        config (dict): The configuration settings.
        schedules (pd.DataFrame): The schedules data.

    Returns:
        pd.DataFrame: The loaded and validated attributes data.

    Raises:
        UserWarning: If no attributes are found in the specified file.
    """
    # load attributes data
    if config.get("attributes_path"):
        data_path = Path(config["attributes_path"])
        attributes = pd.read_csv(data_path)
        if attributes.empty:
            raise UserWarning(f"No attributes found in {data_path}.")
        validate_attributes(attributes, config)
        validate_attributes_index(attributes, schedules)
        attributes = attributes.sort_values(by="pid")
        print(
            f"Loaded {len(attributes)} attributes from {config['attributes_path']}"
        )
    else:
        attributes = None

    # load synthetic attributes data
    if attributes is not None:
        if config.get("synthetic_attributes_path"):
            data_path = Path(config["synthetic_attributes_path"])
            synthetic_attributes = pd.read_csv(data_path)
            if synthetic_attributes.empty:
                raise UserWarning(
                    f"No synthetic attributes found in {data_path}."
                )
            validate_attributes(synthetic_attributes, config, synthetic=True)
            print(
                f"Loaded {len(synthetic_attributes)} synthetic attributes from {config['synthetic_attributes_path']}"
            )
        else:
            synthetic_attributes = attributes
            print("Using input attributes as synthetic attributes")
    else:
        synthetic_attributes = None

    return attributes, synthetic_attributes


def validate_attributes(
    attributes: pd.DataFrame, config: dict, synthetic=False
):
    """
    Validate the attributes data.

    Args:
        attributes (pd.DataFrame): The attributes data to be validated.
        config (dict): The configuration settings.
        synthetic (bool, optional): Whether the attributes are synthetic or not. Defaults to False.

    Raises:
        UserWarning: If the attributes data is missing configured conditional columns.
    """
    required_cols = set(config.get("conditionals", {}).keys()) | {"pid"}
    found = set(attributes.columns)
    print(f"Found attributes: {found}")
    missing = required_cols - found
    text = "Synthetic attributes" if synthetic else "Attributes"
    if missing:
        raise UserWarning(
            f"""
    {text} data is missing configures conditional columns:
    Required: {required_cols}.
    Found: {found}.
    Please add missing: {missing}.
    """
        )


def validate_attributes_index(
    attributes: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Sort the attributes data based on the sequence data.

    Args:
        attributes (pd.DataFrame): The attributes data to be sorted.
        schedules (pd.DataFrame): The schedules data.

    Returns:
        pd.DataFrame: The sorted attributes data.

    Raises:
        UserWarning: If the schedules and attributes pids do not match or their datatypes do not match.
    """
    seq_index = schedules.pid
    attr_index = attributes.pid
    if not set(seq_index) == set(attr_index):
        raise UserWarning("Schedules and attributes pids do not match")
    if not seq_index.dtype == attr_index.dtype:
        raise UserWarning(
            "Schedules and attributes pid datatypes do not match, this may result in 'misalignment' of schedules and attributes."
        )
