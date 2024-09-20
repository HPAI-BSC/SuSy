from typing import Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import repair_checkpoint, task_wrapper

softmax = torch.nn.Softmax(dim=1)

class SuSy(torch.nn.Module):
    def __init__(self, fe, mlp):
        super(SuSy, self).__init__()
        self.feature_extractor = fe
        self.mlp = mlp
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt_path = cfg.get("ckpt_path")

    ckpt = repair_checkpoint(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["state_dict"])

    print(f"Loaded checkpoint from {ckpt_path}")

    susy = SuSy(model.feature_extractor, model.mlp)

    print("Transforming to PyTorch Model...")
    model_scripted = torch.jit.script(susy)

    print("Saving model...")
    model_scripted.save('SuSy.pt')

    print("Loading model...")
    model = torch.jit.load('SuSy.pt')

    return {}, {}

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

if __name__ == "__main__":
    main()
