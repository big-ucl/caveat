"""Orchestration loop — runs the model, collects metrics, applies LLM edits, repeats."""

import argparse
import json
import sys
from pathlib import Path

import harness
import agent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LLM-assisted hyperparameter tuner: train → metrics → LLM edits → repeat"
    )
    parser.add_argument("model", help="Path to the user model file (.py)")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Number of LLM edit cycles (default: 5). Total runs = iterations + 1.",
    )
    parser.add_argument(
        "--output", "-o",
        default="run_history.json",
        help="Path to write run history JSON (default: run_history.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    history = []
    total_runs = args.iterations + 1  # run 0 is baseline

    print(f"Starting tuning loop: {total_runs} run(s), model={model_path}")
    print("-" * 60)

    for run_idx in range(total_runs):
        print(f"Run {run_idx}/{args.iterations} ...", end=" ", flush=True)

        metrics = harness.run(model_path)
        record = {"run": run_idx, **metrics}
        history.append(record)

        # Save after every run so partial results are preserved
        Path(args.output).write_text(json.dumps(history, indent=2))

        print(
            f"val_loss={metrics['best_val_loss']:.4f}  "
            f"val_acc={metrics['best_val_accuracy']:.4f}  "
            f"time={metrics['total_time_seconds']:.1f}s"
        )

        # Apply LLM edits unless this was the last run
        if run_idx < args.iterations:
            print(f"  Requesting edits from LLM ...", end=" ", flush=True)
            try:
                agent.edit_model(model_path, metrics, history)
                print("done.")
            except Exception as e:
                print(f"FAILED ({e}). Skipping edit, continuing with unchanged model.")

    print("-" * 60)
    best = min(history, key=lambda r: r["best_val_loss"])
    print(
        f"Best run: #{best['run']}  "
        f"val_loss={best['best_val_loss']:.4f}  "
        f"val_acc={best['best_val_accuracy']:.4f}"
    )
    print(f"History saved to: {args.output}")


if __name__ == "__main__":
    main()
