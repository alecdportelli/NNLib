"""
LossBase.py

Description: Test file for trying out nn features

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss