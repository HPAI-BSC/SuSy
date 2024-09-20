import os
from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import load_data, save_data
from utils.image_utils import compute_patch_indices, img_to_patch_ovelapped


class ImageDataset(Dataset):
    def __init__(self, data_root_dir, split, methods, patch_size, top_k_patches, patch_selection_criterion, patch_strategy, transform, multiclass=False):
        super().__init__()

        self.dataset = data_root_dir.split("/")[-1]
        self.split = split
        self.methods = methods
        self.filenames = []
        self.labels = []
        self._setup_datasets(data_root_dir, methods)

        self.patch_size = patch_size
        self.top_k_patches = top_k_patches
        self.patch_selection_criterion = patch_selection_criterion
        self.patch_strategy = patch_strategy # ["all", "max", "min", "random"]

        self.transform_img = transforms.Compose([transforms.PILToTensor()])
        self.transform_patch = transform
        self.multiclass = multiclass

        self.top_k_indices = self._compute_top_k_indices()

        if not self.multiclass and split != "test":
            self._convert_to_binary()

        self._print_dataset_stats()

    def _print_dataset_stats(self):
        print(f"Number of images: {len(self.filenames)}")
        print(f"All labels: {set(self.labels)}")
        for label_num in set(self.labels):
            method_name = self.methods[label_num]
            print(f"Number of frames in {method_name} {label_num}: {self.labels.count(label_num)}")
    
    def _convert_to_binary(self):
        self.labels = [0 if label == 0 else 1 for label in self.labels]

    def _setup_datasets(self, data_root_dir, methods):
        for method_id, method in enumerate(methods):
            self._setup_dataset(data_root_dir, method, method_id)

    def _setup_dataset(self, data_root_dir, method, method_id):
        for image in (Path(data_root_dir) / self.split / method).iterdir():
            self.filenames.append(image)
            self.labels.append(method_id)

    def _compute_top_k_indices(self):
        top_k_indices_path = f"{self.dataset}_{self.split}_{self.patch_size}px_top{self.top_k_patches}_{self.patch_selection_criterion}_{self.patch_strategy}_indices.pk"

        if os.path.exists(top_k_indices_path):
            print("Loading precomputed top k indices")
            top_k_indices = load_data(top_k_indices_path)
        else:
            print("No precomputed top k indices found. Computing top k indices...")
            top_k_indices = dict()

        missing_filenames = [filename for filename in self.filenames if filename not in top_k_indices.keys()]

        if len(missing_filenames) > 0:
            print(f"Computing top k indices for {len(missing_filenames)} missing filenames")
            top_k_indices_missing = compute_patch_indices(self.filenames, self.patch_size, self.top_k_patches, self.patch_selection_criterion, self.patch_strategy)
            top_k_indices.update(top_k_indices_missing)

            print(f"Saving top k indices to {top_k_indices_path}")
            save_data(top_k_indices, top_k_indices_path)

        return top_k_indices

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])

        # Patches shape: torch.Size([n_patches, 3, 240, 240])
        image = self.transform_img(image)
        patches = img_to_patch_ovelapped(image.unsqueeze(0), self.patch_size, flatten_channels=False).squeeze(0)
        patch_ids = self.top_k_indices[self.filenames[idx]][:self.top_k_patches]
        selected_patches = patches[patch_ids, :]

        selected_patches = np.transpose(torch.div(selected_patches, 255.0).numpy(), (0, 2, 3, 1))
        selected_patches = torch.stack([self.transform_patch(image=patch)["image"] for patch in selected_patches])

        return selected_patches, self.labels[idx], str(self.filenames[idx])


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_root_dir: str,
        transform_train: transforms.Compose,
        transform_val: transforms.Compose,
        transform_test: transforms.Compose,
        methods: list,
        multiclass: bool,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        patch_size: int = 224,
        top_k_patches: int = 5,
        patch_selection_criterion: str = "contrast",
        patch_strategy: str = "max"
    ) -> None:
   
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        """
        return len(self.hparams.methods)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        self.train = ImageDataset(
            data_root_dir=self.hparams.data_root_dir,
            split="train",
            methods=self.hparams.methods,
            patch_size=self.hparams.patch_size,
            top_k_patches=self.hparams.top_k_patches,
            patch_selection_criterion=self.hparams.patch_selection_criterion,
            patch_strategy=self.hparams.patch_strategy,
            transform=self.hparams.transform_train,
            multiclass=self.hparams.multiclass,
            )

        self.val = ImageDataset(
            data_root_dir=self.hparams.data_root_dir,
            split="val",
            methods=self.hparams.methods,
            patch_size=self.hparams.patch_size,
            top_k_patches=self.hparams.top_k_patches,
            patch_selection_criterion=self.hparams.patch_selection_criterion,
            patch_strategy=self.hparams.patch_strategy,
            transform=self.hparams.transform_val,
            multiclass=self.hparams.multiclass,
            )

        self.test = ImageDataset(
            data_root_dir=self.hparams.data_root_dir,
            split="test",
            methods=self.hparams.methods,
            patch_size=self.hparams.patch_size,
            top_k_patches=self.hparams.top_k_patches,
            patch_selection_criterion=self.hparams.patch_selection_criterion,
            patch_strategy=self.hparams.patch_strategy,
            transform=self.hparams.transform_test,
            multiclass=self.hparams.multiclass,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = ImageDataModule()
