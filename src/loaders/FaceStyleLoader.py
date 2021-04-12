import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os
import torch
from pathlib import Path
from loaders.FaceStyleDataset import FaceStyleDataset
from models.mtcnn import fixed_image_standardization
import numpy as np

class FaceStyleLoader(pl.LightningDataModule):

    def __init__(self, data_root:Path =None, batch_size=128):
        """
        Init FaceStyleLoader to be a Dataloader which loads
        face images and corresponding style tokens.
        """
        super().__init__()
        assert data_root is not None
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        self.embeddings = data_root / "embeddings"
        self.images = data_root / "images"
        self.batch_size = batch_size


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = FaceStyleDataset(
                embeddings_file = self.embeddings / "train" / "embedding_list",
                image_dir = self.images / "train",
                transform = self.transform
            )
            self.val = FaceStyleDataset(
                embeddings_file = self.embeddings / "val" / "embedding_list",
                image_dir = self.images / "val",
                transform = self.transform
            )
            self.dims = self.train[0][0].shape
        elif stage == "test" or stage is None:
            self.test = FaceStyleDataset(
                embeddings_file = self.embeddings / "test" / "embedding_list",
                image_dir = self.images / "test",
                transform = self.transform
            )
            self.dims = self.test[0][0].shape

    def train_dataloader(self):
        return DataLoader(self.train,
            batch_size=self.batch_size,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(self.val, 
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(self.test, 
            batch_size=self.batch_size,
        )

