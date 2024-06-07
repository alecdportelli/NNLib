"""
OptimizerBase.py

Description: Base class for optimizers

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np


class OptimizerBase:
    def __init__(self, learning_rate=1, decay=0):
        self.learning_rate = learning_rate
        self.decay = decay
