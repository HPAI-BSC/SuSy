from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np

from sklearn.metrics import accuracy_score

import json


class DetectorModule(LightningModule):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        mlp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        multiclass: bool = True,
        methods: List[str] = None,
        test_method: List[str] = None,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # self.save_hyperparameters(logger=False, ignore=["feature_extractor", "classifier"])

        self.feature_extractor = feature_extractor
        self.mlp = mlp

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.methods = methods
        self.test_method = test_method

        self.num_classes = len(methods) if multiclass else 2

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.train_predictions = []
        self.train_targets = []

        self.val_predictions = []
        self.val_targets = []

        self.test_predictions = []

        self.val_filenames = []

    def log_predictions(self, predictions, targets, filenames):

        result_dict = {
            "epoch": self.current_epoch,
            "methods": list(self.methods),
            "predictions": predictions,
            "y": targets,
            "filenames": filenames,
        }
        json_file_path = f"predictions_epoch_{self.current_epoch}.json"
        with open(json_file_path, "w") as json_file:
            json.dump(result_dict, json_file)

    def log_individual_classes(self, predictions, targets, mode):

        for class_id in range(self.num_classes):
            class_indices = targets == class_id
            class_preds = predictions[class_indices]
            class_labels = targets[class_indices]

            if class_labels.shape[0] > 0:
                class_accuracy = accuracy_score(class_labels.cpu(), class_preds.cpu())

            else:
                print(f"No samples for class {class_id} in this batch")
                class_accuracy = 0.0

            self.log(
                f"{mode}/acc_{self.methods[class_id]}", class_accuracy, prog_bar=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.feature_extractor(x)
        x = self.mlp(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, list]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, filenames = batch
        if len(x.shape) == 5:
            y = y.repeat_interleave(x.shape[1])
            filenames = [filename for filename in filenames for i in range(x.shape[1])]
            x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, filenames

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        self.train_predictions.append(preds)
        self.train_targets.append(targets)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:

        epoch_predictions = torch.cat(self.train_predictions)
        epoch_targets = torch.cat(self.train_targets)
        self.log_individual_classes(epoch_predictions, epoch_targets, "train")
        self.train_predictions = []
        self.train_targets = []
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, val_filenames = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_predictions.append(preds)
        self.val_targets.append(targets)
        self.val_filenames.append(val_filenames)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

        epoch_predictions = torch.cat(self.val_predictions)
        epoch_targets = torch.cat(self.val_targets)
        if not self.trainer.sanity_checking:
            self.log_individual_classes(epoch_predictions, epoch_targets, "val")
            # self.log_predictions(epoch_predictions, epoch_targets, self.val_filenames)
        self.val_predictions = []
        self.val_targets = []
        self.val_filenames = []

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, test_filenames = self.model_step(batch)

        self.test_predictions.append(preds)

        # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        preds = torch.cat(self.test_predictions)
        # convert to 1 or 0
        binary_preds = preds != 0
        # np array all ones
        self.log(
            f"test_{self.test_method}/acc",
            accuracy_score(binary_preds.cpu(), np.ones(binary_preds.shape)),
            prog_bar=True,
        )
        # get percent of each class in preds
        class_freq = torch.bincount(preds, minlength=self.num_classes)

        for p in range(len(class_freq)):
            m = self.methods[p]
            self.log(
                f"test_{self.test_method}_{m}/predicted",
                class_freq[p] / len(preds),
                prog_bar=True,
            )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.feature_extractor = torch.compile(self.feature_extractor)
            self.mlp = torch.compile(self.mlp)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DetectorModule(None, None, None, None)
