import os
import pickle
from pathlib import Path
from typing import List

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (RankedLogger, extras, instantiate_loggers,
                       log_hyperparameters, repair_checkpoint, task_wrapper)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate a given checkpoint on a datamodule testset.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Tuple containing metrics dictionary and object dictionary.
    """

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")

    ckpt_path = cfg.get("ckpt_path")
    ckpt = repair_checkpoint(ckpt_path) if cfg.model.compile else torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    if ckpt_path:
        log.info(f"Loaded checkpoint from {ckpt_path}")

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    model_name = Path(ckpt_path).stem
    result_save_path = os.path.join(cfg.get("result_save_path"), model_name)
    os.makedirs(result_save_path, exist_ok=True)

    log.info(f"Testing dataset {cfg.data.dataset_name} with path: {cfg.data.data_root_dir}")
    datamodule.setup()

    methods = cfg.data.methods
    multiclass = cfg.multiclass
    top_k_patches = cfg.data.top_k_patches
    prediction_dict = {method: {} for method in methods}
    prediction_dict["method_label_mapping"] = {method: i for i, method in enumerate(methods)}

    for i, batch in enumerate(datamodule.test_dataloader()):
        log.info(f"Processing test batch {i}")

        patches, labels, filenames = batch
        labels = labels.cpu().numpy()
        if len(patches.shape) == 5: # [BATCH_SIZE, TOP_K, 3, 224, 224]
            patches = patches.view(patches.shape[0] * patches.shape[1], *patches.shape[2:])

        if i == 0:
            log.info(f"Patches shape {patches.shape}")

        if torch.cuda.is_available():
            patches = patches.cuda()

        with torch.no_grad():
            preds = model(patches)
            preds = torch.nn.functional.softmax(preds)

        preds = preds.detach().cpu().numpy()

        if i == 0:
            log.info(f"Predictions shape {preds.shape}")

        batch_size = len(preds) // 5
        preds = preds.reshape(batch_size, top_k_patches, len(methods) if multiclass else 2)

        if i == 0:
            log.info(f"Image predictions shape {preds.shape}")

        for img_preds, label, filename in zip(preds, labels, filenames):
            prediction_dict[methods[label.item()]][Path(filename).stem] = img_preds

    with open(f"{os.path.join(result_save_path, cfg.data.dataset_name)}.pkl", "wb") as f:
        pickle.dump(prediction_dict, f)

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
