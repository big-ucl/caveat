"""Tests for `caveat` CLI."""

from click.testing import CliRunner

from caveat.cli import cli


def test_command_line_interface_help():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    assert (
        "Console script for caveat.\n\nOptions:\n  "
        "--version  Show the version and exit.\n  "
        "--help     Show this message and exit.\n" in help_result.output
    )
