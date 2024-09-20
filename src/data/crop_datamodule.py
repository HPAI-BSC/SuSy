from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CropDataset(Dataset):
    def __init__(self, data_root_dir, train, transform,
                 methods=None, label=None, multiclass=False):

        self.data_root_dir = data_root_dir
        self.train = train
        self.transform = transform
        self.multiclass = multiclass
        self.label = label
        self.filenames = []
        self.labels = []
        self.methods = methods

        self.split_path = "train" if self.train else "val"

        if "original" in self.methods:
            assert self.methods[0] == "original", "original method must be the first method"

        self._setup_datasets(data_root_dir, methods)

        if not self.multiclass:
            self._convert_to_binary()

        self._print_dataset_stats()
       
    def _print_dataset_stats(self):
        print(f"Number of images: {len(self.filenames)}")
        print(f"all labels: {set(self.labels)}")
        for label_num in set(self.labels):
            method_name = self.methods[label_num]
            print(f"Number of frames in {method_name} {label_num}: {self.labels.count(label_num)}")
    
    def _convert_to_binary(self):
        self.labels = [0 if label == 0 else 1 for label in self.labels]

    def _setup_datasets(self, data_root_dir, methods):
        for ix , method in enumerate(methods):
            self._setup_dataset(data_root_dir, method, ix)

    def _setup_dataset(self, data_root_dir, dataset_path, label):
        for image in (Path(data_root_dir) / self.split_path / dataset_path).iterdir():
            self.filenames.append(image)
            self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.filenames[idx]).convert("RGB"), dtype=np.float32)
        image = self.transform(image=image)["image"]
        image /= 255.0

        label = self.labels[idx]
      
        return image, label, str(self.filenames[idx])


class CropDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        transform_train: transforms.Compose,
        transform_val: transforms.Compose,
        methods: list,
        multiclass: bool,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
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

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

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

        self.train = CropDataset(
            data_root_dir=self.hparams.data_root_dir,
            methods=self.hparams.methods,
            train=True,
            multiclass=self.hparams.multiclass,
            transform=self.hparams.transform_train,
            )

        self.val = CropDataset(
            data_root_dir=self.hparams.data_root_dir,
            methods=self.hparams.methods,
            train=False,
            multiclass=self.hparams.multiclass,
            transform=self.hparams.transform_val,
            )

        self.test = CropDataset(
            data_root_dir=self.hparams.data_root_dir,
            methods=self.hparams.methods,
            train=False,
            multiclass=self.hparams.multiclass,
            transform=self.hparams.transform_val,
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

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CropDataModule()
