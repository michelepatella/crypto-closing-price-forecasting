"""src/data/utils/adjacency_matrix.py

Adjacency matrix construction using lagged cross-correlations.
"""

import numpy as np


def _compute_lagged_cross_correlations(
    log_returns: dict[str, np.ndarray],
    max_lag: int,
) -> dict[tuple, list[dict]]:
    """Compute lagged cross-correlations between cryptocurrency pairs.

    Args:
        log_returns (dict[str, np.ndarray]): Log returns per crypto.
        max_lag (int): Maximum lag horizon.

    Returns:
        dict[tuple, list[dict]]: Cross-correlations indexed by (crypto_i,
            crypto_j). Each value is a list of dicts containing:
            - 'lag': lag value
            - 'corr': correlation coefficient
    """
    cross_correlations = {}
    cryptos = sorted(log_returns.keys())

    # Convert dict to matrix
    min_len = min(len(log_returns[c]) for c in cryptos)
    returns_matrix = np.array(
        [log_returns[c][:min_len] for c in cryptos],
        dtype=np.float32,
    )

    if min_len <= max_lag:
        # Not enough data for lagged correlations
        for i, crypto_i in enumerate(cryptos):
            for j, crypto_j in enumerate(cryptos):
                if i != j:
                    cross_correlations[(crypto_i, crypto_j)] = []
        return cross_correlations

    # Initialize correlation storage
    lag_correlations = {}

    for h in range(1, max_lag + 1):
        # Extract lagged and shifted windows
        lagged = returns_matrix[:, :-h]
        shifted = returns_matrix[:, h:]

        # Normalize to zero mean and unit variance
        lagged_mean = np.mean(lagged, axis=1, keepdims=True)
        lagged_std = np.std(lagged, axis=1, keepdims=True)
        lagged_std[lagged_std == 0] = 1.0

        shifted_mean = np.mean(shifted, axis=1, keepdims=True)
        shifted_std = np.std(shifted, axis=1, keepdims=True)
        shifted_std[shifted_std == 0] = 1.0

        lagged_norm = (lagged - lagged_mean) / lagged_std
        shifted_norm = (shifted - shifted_mean) / shifted_std

        # Compute correlation matrix
        T = lagged.shape[1]
        corr_matrix = np.dot(lagged_norm, shifted_norm.T) / T

        lag_correlations[h] = corr_matrix

    # Convert correlation matrices back to dict format
    for i, crypto_i in enumerate(cryptos):
        for j, crypto_j in enumerate(cryptos):
            if i == j:
                continue

            key = (crypto_i, crypto_j)
            correlations = []

            for h in range(1, max_lag + 1):
                corr_value = lag_correlations[h][i, j]

                # Handle NaN values
                if not np.isnan(corr_value):
                    correlations.append(
                        {"lag": h, "corr": abs(float(corr_value))},
                    )

            cross_correlations[key] = correlations

    return cross_correlations


def _select_top_k_correlations(
    cross_correlations: dict[tuple, list[dict]],
    top_k: int,
) -> dict[tuple, list[dict]]:
    """Select top-k strongest correlations per cryptocurrency pair.

    Args:
        cross_correlations (dict[tuple, list[dict]]): Cross-correlations
            indexed by (crypto_i, crypto_j).
        top_k (int): Number of top correlations to keep.

    Returns:
        dict[tuple, list[dict]]: Top-k correlations per pair.
    """
    top_k_correlations = {}

    for pair, correlations in cross_correlations.items():
        if not correlations:
            top_k_correlations[pair] = []
            continue

        # Sort by correlation strength (descending)
        sorted_corrs = sorted(
            correlations,
            key=lambda x: x["corr"],
            reverse=True,
        )
        top_k_correlations[pair] = sorted_corrs[:top_k]

    return top_k_correlations


def build_adjacency_matrix(
    crypto_data: dict[str, np.ndarray],
    cryptos: list[str],
    window_size: int,
    max_lag: int,
    top_k: int,
    close_col_idx: int,
) -> np.ndarray:
    """Build adjacency matrix with temporal and cross-crypto edges.

    Args:
        crypto_data (dict[str, np.ndarray]): Pre-extracted data for each crypto.
        cryptos (list[str]): Sorted list of cryptocurrency names.
        window_size (int): Size of the sliding window.
        max_lag (int): Maximum lag horizon for cross-correlations.
        top_k (int): Number of top correlations to keep per pair.
        close_col_idx (int): Column index for close price.

    Returns:
        np.ndarray: Normalized combined adjacency matrix.
    """
    num_cryptos = len(cryptos)
    num_nodes = num_cryptos * window_size

    # ===================================
    # 1. TEMPORAL EDGES
    # ===================================
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for crypto_idx in range(num_cryptos):
        for t in range(window_size - 1):
            node_t = crypto_idx * window_size + t
            node_t_plus_1 = crypto_idx * window_size + t + 1
            A[node_t, node_t_plus_1] = 1.0

    # ===================================
    # 2. CROSS-CRYPTO EDGES
    # ===================================
    # Compute log returns directly from arrays
    log_returns = {}

    for crypto in cryptos:
        prices = crypto_data[crypto][:, close_col_idx]
        returns = np.log(prices[1:] / prices[:-1])
        returns = np.concatenate([[0.0], returns])
        log_returns[crypto] = returns

    # Compute lagged cross-correlations
    cross_correlations = _compute_lagged_cross_correlations(
        log_returns,
        max_lag=max_lag,
    )

    # Select top-k correlations
    top_k_correlations = _select_top_k_correlations(
        cross_correlations,
        top_k=top_k,
    )

    # Create mapping from crypto name to index
    crypto_to_idx = {crypto: idx for idx, crypto in enumerate(cryptos)}

    # Fill adjacency matrix with cross-crypto edges
    for (crypto_i, crypto_j), correlations in top_k_correlations.items():
        i = crypto_to_idx[crypto_i]
        j = crypto_to_idx[crypto_j]

        for corr_info in correlations:
            h = corr_info["lag"]
            corr = corr_info["corr"]

            # For each timestamp t in the window (except last window_size-h)
            for t in range(window_size - h):
                node_i = i * window_size + t
                node_j = j * window_size + (t + h)

                if node_j < num_nodes:
                    A[node_i, node_j] += corr

    # ===================================
    # 3. NORMALIZATION
    # ===================================
    # Normalize combined matrix by row sums
    D = np.sum(A, axis=1, keepdims=True)
    D[D == 0] = 1.0
    A_normalized = A / D

    return A_normalized
