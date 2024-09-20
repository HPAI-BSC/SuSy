import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pathlib import Path
import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from torch.nn.functional import softmax

from src.utils import (RankedLogger, extras, get_metric_value,
                       instantiate_callbacks, instantiate_loggers,
                       log_hyperparameters, task_wrapper)

log = RankedLogger(__name__, rank_zero_only=True)

local_rank = None
def rank0_log(*args):
    if local_rank == 0:
        log.info(*args)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random

    global local_rank

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    
    rank0_log(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    rank0_log(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    rank0_log("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    rank0_log("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    rank0_log(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    local_rank = trainer.local_rank
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        rank0_log("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        rank0_log("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        rank0_log("Finished training!")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        
        model.eval()
        rank0_log("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        test_method = cfg.lou_method

        if test_method:
           
            rank0_log(f"Testing with {test_method} method")
            test_datamodule = hydra.utils.instantiate(cfg.data, methods=[test_method])
            trainer.test(model=model, datamodule=test_datamodule, ckpt_path=ckpt_path)
        else:
            rank0_log("No test method provided!")

        if cfg.get("test_folder_paths"):
            os.makedirs("predictions", exist_ok=True)
            test_folder_paths = cfg.test_folder_paths
            rank0_log("Testing on custom test folders")
            results = {}
            for test_folder_path in test_folder_paths:
                test_dataset = hydra.utils.instantiate(cfg.test_dataset, folder_path=test_folder_path)
                test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=40, pin_memory=False)
                test_dataset_predictions = []
                test_dataset_label = 0 if "original" in test_folder_path else 1
                for i, batch in enumerate(test_dataloader):
                    rank0_log(f"Processing test batch {i}")
                    rank0_log(batch.shape)
                    # [128, TOP_K, 3, 224, 224]
                    patches = batch.view(-1, 3, 224, 224)
                    rank0_log(patches.shape)
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    
                    with torch.no_grad():
                        preds = model(patches)
                        preds = softmax(preds)
                        preds = torch.argmax(preds, dim=1)

                    preds = preds.detach().cpu().numpy()
                    rank0_log(f"Predictions shape {preds.shape}")
                    test_dataset_predictions.extend(preds)
                    rank0_log("Predictions shape")
                
                rank0_log(f"Test dataset predictions shape {np.array(test_dataset_predictions).shape}")
                results[test_folder_path] = test_dataset_predictions

                # calculate accuracy, 0 is real 1 is fake
                is_fake_thresholds = [0.25, 0.5, 0.75]
                for is_fake_threshold in is_fake_thresholds:

                    binary_predictions = [0 if x < is_fake_threshold else 1 for x in test_dataset_predictions]
                    accuracy = np.mean(np.array(binary_predictions) == test_dataset_label)
                
                    rank0_log(f"Dataset label is {test_dataset_label}")
                    rank0_log(f"Patch level accuracy for {test_folder_path} is {accuracy}. Threshold is {is_fake_threshold}")
                    test_dataset_name = Path(test_folder_path).stem
                    wandb.log({f"test/{test_dataset_name}_patch_acc_{is_fake_threshold}": accuracy})
                
            with open("predictions/test_predictions.pkl", "wb") as f:
                pickle.dump(results, f)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
