from pathlib import Path

import pytest
import yaml

from caveat.jrunners import jrun_command
from caveat.mmrunners import mmrun_command
from caveat.runners import (
    batch_command,
    ngen_command,
    nrun_command,
    run_command,
)
from caveat.tune import tune_command

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "configs" / "examples"


def iter_configs():
    for path in EXAMPLES_DIR.iterdir():
        if path.suffix == ".yml" or path.suffix == ".yaml":
            yield path


def pre_process_config(path: Path, tmp_path) -> Path:
    config = yaml.load(path.read_text(), Loader=yaml.FullLoader)

    if "global" in config:
        config["global"]["logging_params"]["log_dir"] = str(tmp_path)

        config["global"]["schedules_path"] = str(
            ROOT / config["global"]["schedules_path"]
        )
        if config["global"].get("attributes_path"):
            config["global"]["attributes_path"] = str(
                ROOT / config["global"]["attributes_path"]
            )

    else:
        config["logging_params"] = config.get("logging_params", {})
        config["logging_params"]["log_dir"] = str(tmp_path)

        config["schedules_path"] = str(ROOT / config["schedules_path"])
        if config.get("attributes_path"):
            config["attributes_path"] = str(ROOT / config["attributes_path"])

    with open(tmp_path / path.name, "w") as f:
        yaml.dump(config, f)

    return tmp_path / path.name


CONFIGS = list(iter_configs())
NAMES = [path.name for path in CONFIGS]


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_run_configs(path, tmp_path):
    path = pre_process_config(path, tmp_path)
    name = path.name
    print(f">>> Running {name}")
    config = yaml.load(path.read_text(), Loader=yaml.FullLoader)
    if name.startswith("run"):
        run_command(config)
    elif name.startswith("batch"):
        batch_command(config)
    elif name.startswith("nrun"):
        nrun_command(config, n=2)
    elif name.startswith("ngen"):
        ngen_command(config, n=2)
    elif name.startswith("jrun"):
        jrun_command(config)
    elif name.startswith("mmrun"):
        mmrun_command(config)
    elif name.startswith("tune"):
        tune_command(config)
    else:
        raise ValueError(f"Unknown config type: {name}")
