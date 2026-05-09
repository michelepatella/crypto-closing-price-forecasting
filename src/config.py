# ===================================
# REPRODUCIBILITY
# ===================================
SEED = 42
"""
The random seed for reproducibility.
"""

# ===================================
# DATA
# ===================================

WINDOW_SIZE = 24
"""
The size of the model input window, representing the number of past time
steps used for prediction (> 0).
"""

TRAIN_RATIO = 0.8
"""
The training split ratio (0.0-1.0).
"""

# ===================================
# RESOURCES
# ===================================

TRAINING_DEVICE = "mps"
"""
The device to use for training ("cpu", "cuda", "mps").
"""
