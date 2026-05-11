from caveat.experiment import Experiment


class LabelExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        self.attribute_embed_sizes = kwargs.get("attribute_embed_sizes", None)
        if self.attribute_embed_sizes is None:
            raise UserWarning("ConditionalLSTM requires attribute_embed_sizes")
        if not isinstance(self.attribute_embed_sizes, list):
            raise UserWarning(
                "ConditionalLSTM requires attribute_embed_sizes to be a list of attribute embedding sizes"
            )
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (x, _), (y, y_mask), (labels, label_mask) = batch

        self.curr_device = x.device
        probs = self.forward(x)
        train_loss = self.loss_function(
            probs=probs, target=labels, mask=label_mask, batch_idx=batch_idx
        )
        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )
        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        (x, _), (y, y_weights), (labels, label_weights) = batch
        self.curr_device = x.device

        probs = self.forward(x)
        val_loss = self.loss_function(
            probs=probs,
            target=labels,
            mask=label_weights,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        (x, _), (y, y_weights), (labels, label_weights) = batch
        self.curr_device = x.device

        probs = self.forward(x)
        test_loss = self.loss_function(
            probs=probs,
            target=labels,
            mask=label_weights,
            duratio_weight=self.duration_loss_weight,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"test_{key}": val.item() for key, val in test_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def predict_step(self, batch):
        (x, _), (_, _), (target_labels, _) = batch
        preds = self.predict(x, device=self.curr_device)
        return x, target_labels, preds

    def on_validation_end(self):
        return None
