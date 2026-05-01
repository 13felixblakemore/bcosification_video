import os
import time
import pytorchvideo
import torch
import torch.utils.data as data
import torchvision
from pytorchvideo.data import make_clip_sampler
from torchvision.datasets import ImageFolder, UCF101
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import bcos.settings as settings
from random import shuffle
from typing import List
from .categories import IMAGENET_CATEGORIES, IMAGENETTE_CATEGORIES, UCF101_CATEGORIES
from .sampler import RASampler
from .transforms import RandomCutmix, RandomMixup, SplitAndGrid

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

# This file is adapted from the B-Cosification repository framework:
# https://github.com/shrebox/B-cosification

# My contribution is the UCF101DataModule only, which is adapted from the existing ImageNetDataModule

__all__ = ["ImageNetDataModule", "ImageNetteDataModule", "ClassificationDataModule", "UCF101DataModule"]

def get_random_cut(dataset, cut_ratio):
    all_indices = [*range(0, len(dataset))]
    cut_idx = int(cut_ratio* len(all_indices))
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_indices) #Always shuffle the same way.
    random.setstate(state)
    return all_indices[:cut_idx], all_indices[cut_idx:]

class ClassificationDataModule(pl.LightningDataModule):
    """Base class for data modules for classification tasks."""

    NUM_CLASSES: int = None
    """Number of classes in the dataset."""
    NUM_TRAIN_EXAMPLES: int = None
    """Number of training examples in the dataset. Need not be defined."""
    NUM_EVAL_EXAMPLES: int = None
    """Number of evaluation examples in the dataset. Need not be defined."""
    CATEGORIES: List[str] = None
    """List of categories in the dataset. Need not be defined."""

    # ===================================== [ Registry stuff ] ======================================
    __data_module_registry = {}
    """Registry of data modules."""

    def __init_subclass__(cls, **kwargs):
        # check that the class attributes are defined
        super().__init_subclass__(**kwargs)
        assert cls.NUM_CLASSES is not None
        # rest don't need to be defined

        # get name and remove DataModule suffix
        name = cls.__name__
        # check if name matches XXXDataModule
        if not name.endswith("DataModule"):
            raise ValueError(
                f"Data module class name '{name}' does not end with 'DataModule'"
            )
        name = name[: -len("DataModule")]
        # check if name is already registered
        if name in cls.__data_module_registry:
            raise ValueError(f"Data module {name} already registered")
        # register the class in the registry
        cls.__data_module_registry[name] = cls

    @classmethod
    def registry(cls):
        """Returns the registry of data modules."""
        return cls.__data_module_registry

    # ===================================== [ Normal stuff ] ======================================
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        p_gridified = config.get("p_gridified", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(
            mixup_alpha, cutmix_alpha, p_gridified
        )

    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        shuffle = None if train_sampler is not None else True
        return data.DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def get_train_collate_fn(
        cls,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_gridified: float = 0.0,
    ):
        assert not (p_gridified and mixup_alpha), "For now, do not use both."

        collate_fn = None
        if p_gridified:
            gridify = SplitAndGrid(p_gridified, num_classes=cls.NUM_CLASSES)

            def collate_fn(batch):
                return gridify(*data.default_collate(batch))

            rank_zero_info(f"Gridify active for training with {p_gridified=}")

        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):  # noqa: F811
                return mixupcutmix(*data.default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler


class ImageNetDataModule(ClassificationDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 1_281_167
    NUM_EVAL_EXAMPLES: int = 50_000

    CATEGORIES: List[str] = IMAGENET_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()
        print(f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        # this way changes to the settings are reflected at function call time
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENET_PATH = settings.IMAGENET_PATH
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNet train dataset...")
            start = time.perf_counter()
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENET_PATH,
                "train",
            )
            self.train_dataset = ImageFolder(
                root=train_root,
                transform=self.config["train_transform"],
            )
            #assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNet val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, "val"),
            transform=self.config["test_transform"],
        )
        #assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class ImageNetteDataModule(ClassificationDataModule):
    NUM_CLASSES: int = 10

    NUM_TRAIN_EXAMPLES: int = 50000
    NUM_EVAL_EXAMPLES: int = 3925

    CATEGORIES: List[str] = IMAGENETTE_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()

        (f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        # this way changes to the settings are reflected at function call time
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENETTE_PATH = settings.IMAGENETTE_PATH
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNette train dataset...")
            start = time.perf_counter()
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENETTE_PATH,
                "train",
            )
            self.train_dataset = ImageFolder(
                root=train_root,
                transform=self.config["train_transform"],
            )
            #assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNette val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENETTE_PATH, "val"),
            transform=self.config["test_transform"],
        )
        #assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class UCF101DataModule(ClassificationDataModule):
    NUM_CLASSES: int = 101

    NUM_TRAIN_EXAMPLES: int = 50000 # Not sure
    NUM_EVAL_EXAMPLES: int = 3925 # Not sure

    CATEGORIES: List[str] = UCF101_CATEGORIES

    UCF101_PATH = settings.UCF101_PATH
    _TRAIN_PATH = "ucfTrainTestlist"
    _TEST_PATH = "ucfTrainTestlist"

    def setup(self, stage: str) -> None:
        train_md = torch.load("ucf101_train_metadata.pt")
        val_md = torch.load("ucf101_eval_metadata.pt")
        if stage == "fit":
            rank_zero_info("Setting up UCF101 train dataset...")
            start = time.perf_counter()
            self.train_dataset = UCF101(
                root=settings.UCF101_PATH,
                annotation_path=self._TRAIN_PATH,
                frames_per_clip=8,
                fold=2,
                transform=self.config["train_transform"],
                step_between_clips=32,
                train=True,
                _precomputed_metadata=train_md,
            )
            self.train_dataset = VideoOnlyDataset(self.train_dataset)
            #torch.save(self.train_dataset.metadata, "ucf101_train_metadata.pt")
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

        start = time.perf_counter()
        rank_zero_info("Setting up UCF101 val dataset...")
        self.eval_dataset = UCF101(
            root=settings.UCF101_PATH,
            annotation_path=self._TRAIN_PATH,
            frames_per_clip=8,
            fold=2,
            transform=self.config["test_transform"],
            step_between_clips=32,
            train=False,
            _precomputed_metadata=val_md,
        )
        self.eval_dataset = VideoOnlyDataset(self.eval_dataset)
        #torch.save(self.eval_dataset.metadata, "ucf101_eval_metadata.pt")
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

# Removes audio from videos
class VideoOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, _, label = self.dataset[idx]  # discard audio
        return video, label









