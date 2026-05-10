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

from const import TARGET_COLUMNS


class SMAPELoss(nn.Module):
    """Custom implementation of SMAPE loss function."""

    def __init__(self) -> None:
        """Initialize SMAPELoss.

        Returns:
            None
        """
        super().__init__()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SMAPE loss between predictions and true values.

        Args:
            y_pred (torch.Tensor):
                Predicted values.
            y_true (torch.Tensor):
                True values.

        Returns:
            torch.Tensor:
                Computed SMAPE loss.
        """
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2

        # SMAPE formula
        smape = numerator / (denominator + 1e-10)

        return 100 * torch.mean(smape)


class EarlyStopping:
    """Early stopping callback.

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


class Trainer:
    """Training pipeline with Early Stopping.

    Attributes:
        device (torch.device):
            Computation device.
        model_config (dict):
            Configuration for the model.
        model_path (Path):
            Path for saving the best model.
        report (dict):
            Comprehensive training report.
    """

    def __init__(
        self,
        model_config: dict,
        model_path: Path,
        device: str,
    ) -> None:
        """Initialize Trainer.

        Args:
            model_config (dict):
                Configuration for the model.
            model_path (Path):
                Path for saving the best model.
            device (str):
                Device to use.

        Returns:
            None
        """
        self.device = torch.device(device)
        self.model_config = model_config
        self.model_path = model_path
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
            hidden_dim=self.model_config["tmtgnn"]["hidden_dim"],
            num_layers=self.model_config["tmtgnn"]["num_layers"],
            skip_dim=self.model_config["tmtgnn"]["skip_dim"],
            head_dim=self.model_config["tmtgnn"]["head_dim"],
            dropout=self.model_config["tmtgnn"]["dropout"],
        )

        transformer_config = TransformerConfig(
            num_heads=self.model_config["transformer"]["num_heads"],
            num_layers=self.model_config["transformer"]["num_layers"],
            dropout=self.model_config["transformer"]["dropout"],
            max_sequence_length=self.model_config["transformer"][
                "max_sequence_length"
            ],
            mode=self.model_config["transformer"]["mode"],
        )

        graph_config = GraphConfig(
            learning_enabled=self.model_config["graph"]["learning_enabled"],
            top_k=self.model_config["graph"]["top_k"],
            ema_alpha=self.model_config["graph"]["ema_alpha"],
            sigmoid_alpha=self.model_config["graph"]["sigmoid_alpha"],
            noise_scale=self.model_config["graph"]["noise_scale"],
        )

        diffusion_config = DiffusionConfig(
            diffusion_steps=self.model_config["diffusion"]["diffusion_steps"],
            residual_alpha=self.model_config["diffusion"]["residual_alpha"],
            projection_bias=self.model_config["diffusion"]["projection_bias"],
        )

        norm_config = NormConfig(
            eps=self.model_config["norm"]["eps"],
            elementwise_affine=self.model_config["norm"]["elementwise_affine"],
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

            # Reshape output: (B, num_nodes) -> (B, num_cryptos, window_size)
            num_cryptos = y_batch.shape[1]
            num_nodes = y_pred.shape[1]
            window_size = num_nodes // num_cryptos

            y_pred = y_pred.view(y_pred.shape[0], num_cryptos, window_size)

            # Get prediction for the last time step
            y_pred = y_pred[:, :, -1]

            # Compute loss
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

                # Reshape output: (B, num_nodes) -> (B, num_cryptos, window_size)
                num_cryptos = y_batch.shape[1]
                num_nodes = y_pred.shape[1]
                window_size = num_nodes // num_cryptos

                y_pred = y_pred.view(y_pred.shape[0], num_cryptos, window_size)

                # Get prediction for the last time step
                y_pred = y_pred[:, :, -1]

                # Compute loss
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
            """Custom dataset for time series data.

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
            """Stack features and targets while keeping a single adjacency matrix.

            Args:
                batch (list[tuple]):
                    A list of tuples, where each tuple contains a feature tensor,
                    target tensor, and adjacency matrix.

            Returns:
                tuple:
                    A tuple containing the stacked feature tensor, stacked target tensor,
                    and a single adjacency matrix.
            """
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
        X_val: np.ndarray,
        y_val: np.ndarray,
        A_val: np.ndarray,
        batch_size: int,
        training_epochs: int,
        optimizer_config: dict,
        scheduler_config: dict,
        early_stopping_config: dict,
    ) -> dict:
        """ "Train the model and return a comprehensive report.

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
                Batch size for training.
            training_epochs (int):
                Number of training epochs.
            optimizer_config (dict):
                Configuration for the optimizer.
            scheduler_config (dict):
                Configuration for the learning rate scheduler.
            early_stopping_config (dict):
                Configuration for early stopping.

        Returns:
            dict:
                A comprehensive training report.
        """
        # Extract dimensions from training data
        num_nodes = X_train.shape[2]
        in_channels = X_train.shape[1]
        seq_length = X_train.shape[3]
        out_channels = len(TARGET_COLUMNS)

        # Create model
        model = self.create_model(
            num_nodes,
            in_channels,
            seq_length,
            out_channels,
        )

        # Configure optimizer, loss function, and learning rate scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
            betas=optimizer_config["betas"],
        )

        criterion = SMAPELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
        )

        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train,
            y_train,
            A_train,
            X_val,
            y_val,
            A_val,
            batch_size,
        )

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_config["patience"],
            delta=early_stopping_config["delta"],
        )

        # Training loop with early stopping
        train_losses = []
        val_losses = []
        best_model_state = None
        epoch_bar = tqdm(range(training_epochs), desc="Training")
        for epoch in epoch_bar:
            # Train for one epoch
            train_loss = self.train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )

            # Validate on validation set
            val_loss = self.validate(model, val_loader, criterion)

            # Update metrics and check for early stopping
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Step the learning rate scheduler based on validation loss
            scheduler.step(val_loss)

            # Check for improvement and save best model state
            improved = (
                val_loss < early_stopping.best_loss - early_stopping.delta
            )

            if improved:
                best_model_state = copy.deepcopy(model.state_dict())

            if early_stopping(val_loss, epoch):
                break

            epoch_bar.set_postfix(
                {"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"},
            )

        # Load best model state before returning report
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Prepare comprehensive training report
        self.report = {
            "summary": {
                "best_epoch": early_stopping.best_epoch + 1,
                "best_val_loss": float(early_stopping.best_loss),
                "final_train_loss": float(train_losses[-1]),
                "final_val_loss": float(val_losses[-1]),
            },
            "history": {
                "train_losses": [float(x) for x in train_losses],
                "val_losses": [float(x) for x in val_losses],
                "min_train_loss": float(min(train_losses)),
                "min_val_loss": float(min(val_losses)),
            },
            "config": {
                "model": self.model_config,
                "optimizer": optimizer_config,
                "scheduler": scheduler_config,
                "early_stopping": early_stopping_config,
                "batch_size": batch_size,
                "epochs": training_epochs,
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "dataset": {
                    "num_train_samples": len(X_train),
                    "num_val_samples": len(X_val),
                },
            },
        }

        # Save best model checkpoint along with training report
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": self.model_config,
                "report": self.report,
            },
            self.model_path,
        )

        return self.report
