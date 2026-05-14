"""src/modeling/evaluate.py

Evaluation pipeline for TMTGNN.
"""

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
from tqdm.auto import tqdm

from const import CLOSE_COLUMN, TARGET_COLUMNS


class Evaluator:
    """Evaluation pipeline for testing TMTGNN performance.

    Attributes:
        device (torch.device):
            Computation device.
        model_path (str):
            Path to the saved model checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
    ) -> None:
        """Initialize Evaluator.

        Args:
            model_path (str):
                Path to the best model checkpoint.
            device (str):
                Device to use for inference.

        Returns:
            None
        """
        self.device = torch.device(device)
        self.model_path = model_path

    def _load_model(
        self,
        num_nodes: int,
        in_channels: int,
        seq_length: int,
        out_channels: int,
    ) -> TMTGNN:
        """Load the trained TMTGNN model from checkpoint.

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
                Model instance with loaded weights.
        """
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint["config"]

        # Reconstruct internal configs
        tmtgnn_config = TMTGNNConfig(**model_config["tmtgnn"])
        transformer_config = TransformerConfig(**model_config["transformer"])
        graph_config = GraphConfig(**model_config["graph"])
        diffusion_config = DiffusionConfig(**model_config["diffusion"])
        norm_config = NormConfig(**model_config["norm"])

        # Initialize model and load state dict
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

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _unnormalize(
        self,
        y_tensor: torch.Tensor,
        prep_report: dict | None,
    ) -> torch.Tensor:
        """Inverse transform normalized targets/predictions.

        Args:
            y_tensor (torch.Tensor):
                Normalized tensor of shape (B, num_cryptos).
            prep_report (dict | None):
                Data preparation report containing normalization stats.

        Returns:
            torch.Tensor:
                Un-normalized tensor.
        """
        if prep_report is None:
            return y_tensor

        per_crypto = prep_report.get("per_crypto_normalization")
        cryptos = prep_report.get("cryptos_order")
        if per_crypto is None or cryptos is None:
            return y_tensor

        y_un = y_tensor.detach().cpu().clone()
        for i, crypto in enumerate(cryptos):
            stats = per_crypto.get(crypto, {}).get(CLOSE_COLUMN)
            if stats is None:
                continue
            y_un[:, i] = y_un[:, i] * stats["std"] + stats["mean"]

        return y_un

    def _compute_directional_accuracy(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute directional accuracy between predictions and true values.

        This function measures the percentage of correct direction predictions
        (up/down) between consecutive time steps.

        Args:
            y_pred (np.ndarray):
                Predicted values of shape (batch_size,).
            y_true (np.ndarray):
                True values of shape (batch_size,).

        Returns:
            float:
                Directional accuracy as percentage (0-100).
        """
        if len(y_pred) < 2 or len(y_true) < 2:
            return 0.0

        # Calculate direction changes (up=1, down=-1, flat=0)
        pred_direction = np.sign(np.diff(y_pred))
        true_direction = np.sign(np.diff(y_true))

        # Count correct directions
        correct = np.sum(pred_direction == true_direction)
        total = len(pred_direction)

        return 100.0 * correct / total if total > 0 else 0.0

    def _compute_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cryptos: list | None,
    ) -> dict:
        """Compute MAE, RMSE, MAPE, and Directional Accuracy metrics per-crypto.

        Args:
            y_pred (torch.Tensor):
                Predicted values (denormalized).
            y_true (torch.Tensor):
                True values (denormalized).
            cryptos (list | None):
                List of crypto names.

        Returns:
            dict:
                Computed metrics per-crypto.
        """
        y_pred_arr = y_pred.cpu().numpy()
        y_true_arr = y_true.cpu().numpy()

        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)

        _, c = y_true_arr.shape
        per_crypto = {}

        for i in range(c):
            pred_i = y_pred_arr[:, i]
            true_i = y_true_arr[:, i]

            mae_i = np.mean(np.abs(pred_i - true_i))
            rmse_i = np.sqrt(np.mean((pred_i - true_i) ** 2))
            mape_i = 100 * np.mean(
                np.abs((true_i - pred_i) / (np.abs(true_i) + 1e-10)),
            )
            da_i = self._compute_directional_accuracy(pred_i, true_i)

            key = cryptos[i] if cryptos and i < len(cryptos) else f"crypto_{i}"
            per_crypto[key] = {
                "mae": float(mae_i),
                "rmse": float(rmse_i),
                "mape": float(mape_i),
                "directional_accuracy": float(da_i),
            }

        return per_crypto

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        A_test: np.ndarray,
        batch_size: int,
        prep_report: dict | None,
    ) -> dict:
        """Run evaluation on the test set.

        Args:
            X_test (np.ndarray):
                Testing feature tensor.
            y_test (np.ndarray):
                Testing target tensor.
            A_test (np.ndarray):
                Adjacency matrix.
            batch_size (int):
                Batch size for inference.
            prep_report (dict | None):
                Data preparation report for un-normalization.

        Returns:
            dict:
                Evaluation report with metrics.
        """
        # Data preparation
        X_tensor = torch.from_numpy(X_test).float()
        y_tensor = torch.from_numpy(y_test).float()
        A_tensor = (
            torch.from_numpy(A_test if A_test.ndim == 2 else A_test[0])
            .float()
            .to(self.device)
        )

        # Model initialization
        model = self._load_model(
            num_nodes=X_test.shape[2],
            in_channels=X_test.shape[1],
            seq_length=X_test.shape[3],
            out_channels=len(TARGET_COLUMNS),
        )

        all_predictions = []
        all_targets = []

        # Inference loop
        with torch.no_grad():
            for i in tqdm(
                range(0, len(X_tensor), batch_size),
                desc="Evaluating",
            ):
                end = min(i + batch_size, len(X_tensor))
                X_batch = X_tensor[i:end].to(self.device)
                y_batch = y_tensor[i:end].to(self.device)

                # Get predictions
                y_pred = model(X_batch, adj=A_tensor)

                # Reshaping
                num_cryptos = y_batch.shape[1]
                num_nodes = y_pred.shape[1]
                window_size = num_nodes // num_cryptos
                y_pred = y_pred.view(y_pred.shape[0], num_cryptos, window_size)

                # Target the last time step prediction
                y_pred = y_pred[:, :, -1]

                all_predictions.append(y_pred.cpu())
                all_targets.append(y_batch.cpu())

        # Metrics computation
        predictions_un = self._unnormalize(
            torch.cat(all_predictions, dim=0),
            prep_report,
        )
        targets_un = self._unnormalize(
            torch.cat(all_targets, dim=0),
            prep_report,
        )
        cryptos = prep_report.get("cryptos_order") if prep_report else None

        test_metrics = self._compute_metrics(
            predictions_un,
            targets_un,
            cryptos=cryptos,
        )

        return {
            "test_metrics": test_metrics,
            "metadata": {
                "num_test_samples": len(X_test),
                "device": str(self.device),
            },
        }
