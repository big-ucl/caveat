from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class LinearLossScheduler(Callback):
    def __init__(self, config: dict) -> None:
        self.min_epochs = config.get("min_epochs", 0)
        self.kld_schedule = config.get("kld_loss_schedule", None)
        self.act_schedule = config.get("activity_loss_schedule", None)
        self.dur_schedule = config.get("duration_loss_schedule", None)
        self.end_schedule = config.get("end_loss_schedule", None)
        self.label_schedule = config.get("label_loss_schedule", None)
        self.validate_weights_schedule("KLD", self.kld_schedule)
        self.validate_weights_schedule("ACT", self.act_schedule)
        self.validate_weights_schedule("DUR", self.dur_schedule)
        self.validate_weights_schedule("END", self.end_schedule)
        self.validate_weights_schedule("ATT", self.label_schedule)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        current_epoch = trainer.current_epoch
        if self.kld_schedule is not None:
            s, e = self.kld_schedule
            if current_epoch < s:
                pl_module.scheduled_kld_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_kld_weight = 1.0
            else:
                pl_module.scheduled_kld_weight = (current_epoch - s) / (e - s)
        if self.act_schedule is not None:
            s, e = self.act_schedule
            if current_epoch < s:
                pl_module.scheduled_act_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_act_weight = 1.0
            else:
                pl_module.scheduled_act_weight = (current_epoch - s) / (e - s)
        if self.dur_schedule is not None:
            s, e = self.dur_schedule
            if current_epoch < s:
                pl_module.scheduled_dur_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_dur_weight = 1.0
            else:
                pl_module.scheduled_dur_weight = (current_epoch - s) / (e - s)
        if self.end_schedule is not None:
            s, e = self.end_schedule
            if current_epoch < s:
                pl_module.scheduled_end_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_end_weight = 1.0
            else:
                pl_module.scheduled_end_weight = (current_epoch - s) / (e - s)
        if self.label_schedule is not None:
            s, e = self.label_schedule
            if current_epoch < s:
                pl_module.scheduled_label_weight = 0.0
            elif current_epoch >= e:
                pl_module.scheduled_label_weight = 1.0
            else:
                pl_module.scheduled_label_weight = (current_epoch - s) / (e - s)

    def validate_weights_schedule(self, name, schedule):
        if schedule is None:
            return None
        s, e = schedule
        if s > e:
            raise ValueError(f"Invalid schedule for {name}: {schedule}")
        if s < 0 or e < 0:
            raise ValueError(f"Invalid schedule for {name}: {schedule}")
        if e < self.min_epochs:
            print(
                f"WARNING: {name} schedule {schedule} ends after min_epochs {self.min_epochs}"
            )
        print(
            f"Found {name} schedule: {s} -> {e}. Check that this is ok with your epochs."
        )
