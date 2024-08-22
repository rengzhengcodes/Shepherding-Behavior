# Debug mode
DEBUG = False

import numpy as np
from numba import jit, config
config.DISABLE_JIT = DEBUG

# Experiment parameters.
MORPHOLOGY = False
MODE = 0

# Target parameters.
TARGET_X = 400
TARGET_Y = 400
TARGET_SIZE = 125  # radius
TARGET = (TARGET_X, TARGET_Y, TARGET_SIZE)
FENCE = True
K_FENCE = np.inf
# In radians
FENCE_MIDDLE_ANGLE = 0
GATE_ANGULAR_WIDTH = 1

# Number of threads each experiment thread can use to draw.
DRAW_THREADS = -1

# Cannot have a fence without a target.
assert not FENCE or not MORPHOLOGY