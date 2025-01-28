"""Console script for caveat."""

from typing import Optional

import click
import yaml

from caveat.jrunners import jbatch_command, jrun_command
from caveat.label_runners import label_run_command
from caveat.mmrunners import mmrun_command
from caveat.runners import (
    batch_command,
    batch_eval_command,
    eval_command,
    ngen_command,
    nrun_command,
    report_command,
    run_command,
)


@click.version_option(package_name="caveat")
@click.group()
def cli():
    """Console script for caveat."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def run(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
):
    """Train and report on an encoder and model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run_command(
            config,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
        )


@cli.command(name="tune")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def tune(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
):
    """Train and report on an encoder and model as per the given configuration file."""
    from caveat.tune import tune_command  # noqa

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        tune_command(
            config,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
        )


@cli.command(name="jrun")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--no_sample", "-ns", is_flag=True)
@click.option("--patience", "-p", type=int, default=8)
def jrun(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
    no_sample: bool,
    patience: int,
):
    """Train and report on a joint model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        jrun_command(
            config,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
            sample=not no_sample,
            patience=patience,
        )


# @cli.command(name="jsample")
# @click.argument("config_path", type=click.Path(exists=True))
# @click.option("--patience", "-p", type=int, default=10)
# @click.option("--test", "-t", is_flag=True)
# @click.option("--no-infer", "-ni", is_flag=True)
# @click.option("--no-gen", "-ng", is_flag=True)
# @click.option("--verbose", "-v", is_flag=True)
# def jsample(
#     config_path: click.Path,
#     patience: int,
#     test: bool,
#     no_gen: bool,
#     no_infer: bool,
#     verbose: bool,
# ):
#     """Train and report on a joint model with sampling as per the given configuration file."""
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#         jsample_command(
#             config,
#             patience=patience,
#             verbose=verbose,
#             test=test,
#             gen=not no_gen,
#             infer=not no_infer,
#         )


@cli.command(name="jbatch")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--no_sample", "-ns", is_flag=True)
@click.option("--patience", "-p", type=int, default=8)
def jbatch(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
    no_sample: bool,
    patience: int,
):
    """Train and report on a batch of joint models as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        jbatch_command(
            config,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
            sample=not no_sample,
            patience=patience,
        )


@cli.command(name="mmrun")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--cool-start", "-cs", is_flag=True)
def mmrun(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
    cool_start: bool,
):
    """Multi-model variation of run command for brute-force conditionality."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        mmrun_command(
            config,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
            warm_start=not cool_start,
        )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--stats", "-s", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def batch(
    config_path: click.Path,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    stats: bool,
    verbose: bool,
):
    """Train and report on a batch of encoders and models as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        batch_command(
            config, stats=stats, verbose=verbose, test=test, gen=not no_gen
        )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
@click.option("--test", "-t", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--stats", "-s", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def nrun(
    config_path: click.Path,
    n: int,
    stats: bool,
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        nrun_command(
            config, n=n, stats=stats, test=test, gen=not no_gen, verbose=verbose
        )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n", type=int, default=5)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--stats", "-s", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def ngen(
    config_path: click.Path, n: int, no_infer: bool, stats: bool, verbose: bool
):
    """Train and report variance on n identical runs with varying seeds."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        ngen_command(
            config, n=n, stats=stats, infer=not no_infer, verbose=verbose
        )


@cli.command()
@click.argument("observed_path", type=click.Path(exists=True))
@click.argument("logs_dir", type=click.Path(exists=True))
@click.option("--name", type=str, default="synthetic_schedules.csv")
@click.option("--verbose", is_flag=True)
@click.option("--head", type=int, default=10)
@click.option("--batch", "-b", is_flag=True)
def report(
    observed_path: click.Path,
    logs_dir: click.Path,
    name: str,
    verbose: bool,
    head: int,
    batch: bool,
):
    """Report on the given observed population and logs directory."""
    report_command(observed_path, logs_dir, name, verbose, head, batch)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--schedules", type=str, default="synthetic_schedules.csv")
@click.option("--labels", type=str)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--stats", is_flag=True)
@click.option("--batch", "-b", is_flag=True)
def eval(
    config_path: click.Path,
    batch: bool,
    verbose: bool,
    stats: bool,
    schedules: str = "synthetic_schedules.csv",
    labels: Optional[str] = None,
):
    """Evaluate on the given observed population and logs directory."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        if batch:
            batch_eval_command(
                batch_config=config,
                schedules_name=schedules,
                labels_name=labels,
                verbose=verbose,
                stats=stats,
            )
        else:
            eval_command(
                config,
                schedules_name=schedules,
                labels_name=labels,
                verbose=verbose,
                stats=stats,
            )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def lrun(config_path: click.Path, verbose: bool):
    """Train and test label predicting model."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        label_run_command(config, verbose=verbose)
