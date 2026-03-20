"""Single training run harness — loads a user model file and returns metrics."""

import importlib.util
import sys
import time
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning import Callback


class MetricsCallback(Callback):
    """Captures validation and training metrics across epochs."""

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.epoch_history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch_metrics = {"epoch": trainer.current_epoch}

        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            epoch_metrics["val_loss"] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        if "val_accuracy" in metrics:
            val_acc = float(metrics["val_accuracy"])
            epoch_metrics["val_accuracy"] = val_acc
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc

        if "train_loss" in metrics:
            epoch_metrics["train_loss"] = float(metrics["train_loss"])

        self.epoch_history.append(epoch_metrics)


def load_model_module(path: str):
    """Dynamically import a model file, invalidating any cached version.

    Args:
        path: Path to the Python model file.

    Returns:
        The imported module.
    """
    # Remove cached version so edits are picked up on re-import
    if "user_model" in sys.modules:
        del sys.modules["user_model"]

    spec = importlib.util.spec_from_file_location("user_model", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = module
    spec.loader.exec_module(module)
    return module


def run(model_path: str) -> dict:
    """Run a single training cycle and return metrics.

    Calls the three contract functions from the model file:
      - ``get_hyperparameters()`` → dict of hyperparameters
      - ``build_model(hp)`` → LightningModule
      - ``get_datamodule(hp)`` → LightningDataModule

    Args:
        model_path: Path to the user model file.

    Returns:
        Dict with keys: ``best_val_loss``, ``best_val_accuracy``,
        ``epoch_history``, ``hyperparameters``, ``total_time_seconds``.
    """
    start = time.time()

    module = load_model_module(model_path)
    hp = module.get_hyperparameters()
    model = module.build_model(hp)
    datamodule = module.get_datamodule(hp)

    metrics_cb = MetricsCallback()

    trainer = L.Trainer(
        max_epochs=hp.get("max_epochs", 10),
        callbacks=[metrics_cb],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)

    total_time = time.time() - start

    return {
        "best_val_loss": metrics_cb.best_val_loss,
        "best_val_accuracy": metrics_cb.best_val_accuracy,
        "epoch_history": metrics_cb.epoch_history,
        "hyperparameters": hp,
        "total_time_seconds": round(total_time, 2),
    }
