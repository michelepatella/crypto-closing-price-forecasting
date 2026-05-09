"""src/training/train.py

Training pipeline for TMTGNN.
"""

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tmtgnn.config import (
    DiffusionConfig,
    GraphConfig,
    NormConfig,
    TMTGNNConfig,
    TransformerConfig,
)
from tmtgnn.models import TMTGNN
from torch import nn
from tqdm.auto import tqdm

from const import BEST_MODEL_PATH


class EarlyStopping:
    """Early stopping callback with model checkpoint management.

    This class monitors validation loss and stops training if no improvement is observed
    for a specified number of epochs (patience). Saves best model checkpoint.

    Attributes:
        patience (int):
            Number of epochs to wait for improvement before stopping.
        delta (float):
            Minimum change to qualify as an improvement.
        best_loss (float):
            Best validation loss observed so far.
        patience_counter (int):
            Current patience counter.
        best_epoch (int):
            Epoch with best validation loss.
    """

    def __init__(self, patience: int, delta: float) -> None:
        """Initialize EarlyStopping.

        Args:
            patience (int):
                Number of epochs to wait before stopping.
            delta (float):
                Minimum change to qualify as improvement.

        Returns:
            None
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            val_loss (float):
                Current validation loss.
            epoch (int):
                Current epoch number.

        Returns:
            bool:
                True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_epoch = epoch

            # Continue training
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            # Stop training
            return True

        # Continue training
        return False


class TimeSeriesTrainer:
    """Training pipeline with Time Series Split and Early Stopping.

    This class orchestrates the complete training pipeline including:
    - Data loading and preprocessing
    - Time Series Split cross-validation
    - Training loop with validation
    - Early stopping and checkpoint management
    - Metrics tracking and reporting

    Attributes:
        device (torch.device):
            Computation device.
        model_dir (Path):
            Directory for saving model checkpoints.
        report (dict):
            Comprehensive training report.
    """

    def __init__(self, device: str) -> None:
        """Initialize TimeSeriesTrainer.

        Args:
            device (str):
                Device to use.

        Returns:
            None
        """
        self.device = torch.device(device)
        self.model_dir = Path(BEST_MODEL_PATH)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report = {}

    def create_model(
        self,
        num_nodes: int,
        in_channels: int,
        seq_length: int,
        out_channels: int,
    ) -> TMTGNN:
        """Create TMTGNN model.

        Args:
            num_nodes (int):
                Number of nodes in graph.
            in_channels (int):
                Input feature channels.
            seq_length (int):
                Sequence length.
            out_channels (int):
                Output channels.

        Returns:
            TMTGNN:
                Configured model instance.
        """
        # Configure and initialize T-MTGNN
        tmtgnn_config = TMTGNNConfig(
            hidden_dim=16,
            num_layers=2,
            skip_dim=32,
            head_dim=16,
            dropout=0.1,
            num_forecast_steps=1,
        )

        transformer_config = TransformerConfig(
            num_heads=1,
            num_layers=1,
            dropout=0.1,
            max_sequence_length=seq_length,
            mode="temporal",
        )

        graph_config = GraphConfig(
            learning_enabled=False,
            top_k=10,
            ema_alpha=0.99,
            sigmoid_alpha=1.0,
            noise_scale=0.01,
        )

        diffusion_config = DiffusionConfig(
            diffusion_steps=1,
            residual_alpha=0.5,
            projection_bias=True,
        )

        norm_config = NormConfig(
            eps=1e-5,
            elementwise_affine=True,
        )

        model = TMTGNN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            seq_length=seq_length,
            out_channels=out_channels,
            device=self.device,
            diffusion_config=diffusion_config,
            graph_config=graph_config,
            norm_config=norm_config,
            tmtgnn_config=tmtgnn_config,
            transformer_config=transformer_config,
        ).to(self.device)

        return model

    def train_epoch(
        self,
        model: TMTGNN,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train model for one epoch.

        Args:
            model (TMTGNN):
                Model to train.
            train_loader (DataLoader):
                Training data loader.
            criterion (nn.Module):
                Loss function.
            optimizer (torch.optim.Optimizer):
                Optimizer.

        Returns:
            float:
                Average training loss.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training batches",
            leave=False,
        )

        # Iterate over training batches
        for _, (X_batch, y_batch, A_batch) in progress_bar:
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            A_batch = A_batch.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch, adj=A_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / num_batches

    def validate(
        self,
        model: TMTGNN,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Validate model on validation set.

        Args:
            model (TMTGNN):
                Model to validate.
            val_loader (DataLoader):
                Validation data loader.
            criterion (nn.Module):
                Loss function.

        Returns:
            float:
                Average validation loss.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            # Iterate over validation batches
            for X_batch, y_batch, A_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                A_batch = A_batch.to(self.device)

                # Forward pass
                y_pred = model(X_batch, adj=A_batch)
                loss = criterion(y_pred, y_batch)

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

        # Compute average loss
        avg_loss = total_loss / num_batches

        return avg_loss

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        A_val: np.ndarray,
        batch_size: int,
    ) -> tuple:
        """Create train and validation data loaders.

        Args:
            X_train (np.ndarray):
                Training feature tensor.
            y_train (np.ndarray):
                Training target tensor.
            A_train (np.ndarray):
                Training adjacency matrix.
            X_val (np.ndarray):
                Validation feature tensor.
            y_val (np.ndarray):
                Validation target tensor.
            A_val (np.ndarray):
                Validation adjacency matrix.
            batch_size (int):
                Batch size.

        Returns:
            tuple:
                A tuple containing the training and validation data loaders.
        """

        class TimeSeriesDataset(torch.utils.data.Dataset):
            """Custom dataset for time series data with adjacency matrix support.

            This class creates a PyTorch Dataset that loads time series data along
            with the corresponding adjacency matrix. It supports indexing to retrieve
            batches of data for training and validation.

            Attributes:
                X (torch.Tensor):
                    Feature tensor.
                y (torch.Tensor):
                    Target tensor.
                A (torch.Tensor):
                    Adjacency matrix.
            """

            def __init__(
                self,
                X: np.ndarray,
                y: np.ndarray,
                A: np.ndarray,
            ) -> None:
                """Initialize TimeSeriesDataset.

                Args:
                    X (np.ndarray):
                        Feature tensor.
                    y (np.ndarray):
                        Target tensor.
                    A (np.ndarray):
                        Adjacency matrix.

                Returns:
                    None
                """
                self.X = torch.from_numpy(X).float()
                self.y = torch.from_numpy(y).float()
                self.A = torch.from_numpy(A if A.ndim == 2 else A[0]).float()

            def __len__(self) -> int:
                """Return the number of samples in the dataset.

                Returns:
                    int:
                        Number of samples.
                """
                return len(self.X)

            def __getitem__(self, idx) -> tuple:
                """Retrieve a sample from the dataset.

                Args:
                    idx (int):
                        Index of the sample to retrieve.

                Returns:
                    tuple:
                        A tuple containing the feature tensor, target tensor,
                        and adjacency matrix for the given index.
                """
                return self.X[idx], self.y[idx], self.A

        def collate_batch(batch: list[tuple]) -> tuple:
            """Stack features and targets while keeping a single adjacency matrix."""
            X_batch, y_batch, A_batch = zip(*batch)
            return torch.stack(X_batch), torch.stack(y_batch), A_batch[0]

        # Create datasets for training and validation
        train_dataset = TimeSeriesDataset(X_train, y_train, A_train)
        val_dataset = TimeSeriesDataset(X_val, y_val, A_val)

        # Create data loaders for training and validation
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_batch,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_batch,
        )

        return train_loader, val_loader

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: np.ndarray,
        batch_size: int,
        val_ratio: float = 0.2,
    ) -> dict:
        """Execute complete training pipeline with a single hold-out split.

        Args:
            X_train (np.ndarray):
                Training feature tensor.
            y_train (np.ndarray):
                Training target tensor.
            A_train (np.ndarray):
                Training adjacency matrix.
            batch_size (int):
                Batch size for training.
            val_ratio (float):
                Fraction of the training set reserved for validation.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")

        split_idx = int(len(X_train) * (1.0 - val_ratio))
        if split_idx <= 0 or split_idx >= len(X_train):
            raise ValueError(
                "val_ratio produces an empty train or validation split",
            )

        X_split = X_train[:split_idx]
        y_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]

        num_nodes = X_train.shape[2]
        in_channels = X_train.shape[1]
        seq_length = X_train.shape[3]
        out_channels = 1

        model = self.create_model(
            num_nodes,
            in_channels,
            seq_length,
            out_channels,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        train_loader, val_loader = self.create_data_loaders(
            X_split,
            y_split,
            A_train,
            X_val,
            y_val,
            A_train,
            batch_size,
        )

        early_stopping = EarlyStopping(patience=5, delta=1e-4)

        train_losses = []
        val_losses = []
        best_model_state = None

        epoch_bar = tqdm(range(500), desc="Training")

        for epoch in epoch_bar:
            train_loss = self.train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            train_losses.append(train_loss)

            val_loss = self.validate(model, val_loader, criterion)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss <= early_stopping.best_loss - early_stopping.delta:
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "val_loss": val_loss,
                        "epoch": epoch,
                    },
                    BEST_MODEL_PATH,
                )

            if early_stopping(val_loss, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(
                    f"Best validation loss: {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch + 1}",
                )
                break

            epoch_bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                },
            )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        self.report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_epochs": 500,
                "batch_size": batch_size,
                "validation_ratio": val_ratio,
                "early_stopping_patience": 5,
                "early_stopping_delta": 1e-4,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
                "device": str(self.device),
            },
            "model_config": {
                "hidden_dim": 16,
                "num_layers": 2,
                "skip_dim": 32,
                "head_dim": 16,
                "dropout": 0.1,
                "transformer_heads": 1,
                "transformer_layers": 1,
                "diffusion_steps": 1,
                "graph_learning_enabled": False,
            },
            "holdout": {
                "num_train_samples": len(X_split),
                "num_val_samples": len(X_val),
                "best_epoch": early_stopping.best_epoch + 1,
                "best_val_loss": float(early_stopping.best_loss),
                "final_train_loss": float(train_losses[-1]),
                "final_val_loss": float(val_losses[-1]),
                "min_train_loss": float(min(train_losses)),
                "min_val_loss": float(min(val_losses)),
                "train_losses": [float(x) for x in train_losses],
                "val_losses": [float(x) for x in val_losses],
            },
        }

        return self.report
