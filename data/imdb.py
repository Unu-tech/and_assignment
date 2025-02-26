"""
Lightning datamodule for imdb data
"""

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional


class IMDBDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the IMDB dataset.
    This module handles preparing the IMDB dataset for training, validation, and testing.
    """

    def __init__(
        self,
        tokenizer,
        max_position_embeddings: int,
        batch_size: int,
        num_workers: int,
    ):
        """
        Initialize the IMDB DataModule.

        Args:
            tokenizer: pretrained tokenizer object
            max_position_embeddings: Maximum sequence length
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
        """
        super().__init__()

        self.tokenizer = tokenizer

        self.max_position_embeddings = max_position_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download the IMDB dataset if not already available.
        This method is called only once upon initialization.
        """
        load_dataset("imdb")

    def setup(self, stage: Optional[str] = None):
        """
        Setup train, validation, and test datasets.

        Args:
            stage: 'fit' for training and validation, 'test' for testing
        """
        # Load the full dataset
        if stage == "fit" or stage is None:
            # Get the training dataset with 25k samples
            self.train_dataset = (
                load_dataset("imdb", split="train")
                .map(
                    lambda sample: self.tokenizer(
                        sample["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_position_embeddings,
                        return_tensors="pt",
                    )
                )
                .set_format(type="torch")
            )

            # Split test dataset and get first 10k samples for validation
            self.val_dataset = (
                load_dataset("imdb", split="test[:40%]")
                .map(
                    lambda sample: self.tokenizer(
                        sample["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_position_embeddings,
                        return_tensors="pt",
                    )
                )
                .set_format(type="torch")
            )

        if stage == "test" or stage is None:
            # Get the test dataset and get 15k samples for testing
            self.test_dataset = (
                load_dataset("imdb", split="test[-60%:]")
                .map(
                    lambda sample: self.tokenizer(
                        sample["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_position_embeddings,
                        return_tensors="pt",
                    )
                )
                .set_format(type="torch")
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
