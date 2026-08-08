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
FENCE = False
K_FENCE = 1e3
# In radians
FENCE_MIDDLE_ANGLE = 0
GATE_ANGULAR_WIDTH = 1

# Rendering cadence/playback parameters. DRAW_INTERVAL = record + render
# cadence consumed by experiments/test.py (render every Nth herding tick);
# DRAW_FPS = mp4 playback rate consumed by basic/drawing/video.py.
DRAW_INTERVAL = 100  # record + render every Nth herding tick
DRAW_FPS = 10  # playback fps; 1 rendered frame per DRAW_INTERVAL ticks

# Cannot have a fence without a target.
assert not FENCE or not MORPHOLOGY