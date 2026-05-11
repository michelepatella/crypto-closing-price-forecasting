# ===================================
# REPRODUCIBILITY
# ===================================

seed = 42
"""
The random seed for reproducibility (> 0).
"""

# ===================================
# DATA
# ===================================

window_size = 24
"""
The size of the model input window, representing the number of past time
steps used for prediction (> 0).
"""

sequence_length = 24
"""
The temporal context length provided to the transformer for each timestamp (> 0).
"""

train_ratio = 0.8
"""
The training split ratio (0.0-1.0).
"""

valid_ratio = 0.1
"""
The validation split ratio (0.0-1.0).
"""

# ===================================
# TRAINING
# ===================================

training_batch_size = 2
"""
The batch size for training (> 0).
"""

training_device = "mps"
"""
The device to use for training ("cpu", "cuda", "mps").
"""

training_epochs = 500
"""
The number of training epochs (> 0).
"""

# ===================================
# MODEL
# ===================================

model_config = {
    "tmtgnn": {
        "hidden_dim": 16,
        "num_layers": 2,
        "skip_dim": 32,
        "head_dim": 16,
        "dropout": 0.1,
    },
    "transformer": {
        "num_heads": 1,
        "num_layers": 1,
        "dropout": 0.1,
        "max_sequence_length": sequence_length,
        "mode": "temporal",
    },
    "graph": {
        "learning_enabled": False,
        "top_k": 10,
        "ema_alpha": 0.99,
        "sigmoid_alpha": 1.0,
        "noise_scale": 0.01,
    },
    "diffusion": {
        "diffusion_steps": 1,
        "residual_alpha": 0.5,
        "projection_bias": True,
    },
    "norm": {
        "eps": 1e-5,
        "elementwise_affine": True,
    },
}
"""
Model configuration dictionary.
"""

# ===================================
# OPTIMIZER
# ===================================

optimizer_config = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999),
    "max_norm": 1.0,
}
"""
Optimizer configuration dictionary.
"""

# ===================================
# SCHEDULER
# ===================================

scheduler_config = {
    "mode": "min",
    "factor": 0.5,
    "patience": 10,
    "min_lr": 1e-6,
}
"""
Scheduler configuration dictionary.
"""

# ===================================
# EARLY STOPPING
# ===================================

early_stopping_config = {
    "patience": 10,
    "delta": 1e-4,
}
"""
Early stopping configuration dictionary.
"""

# ===================================
# LOSS
# ===================================

loss_alpha = 0.4
"""
The alpha parameter for the combined loss function (0.0-1.0).
"""

# ===================================
# INFERENCE
# ===================================

inference_device = "mps"
"""
The device to use for inference ("cpu", "cuda", "mps").
"""

inference_batch_size = 2
"""
The batch size for inference (> 0).
"""
