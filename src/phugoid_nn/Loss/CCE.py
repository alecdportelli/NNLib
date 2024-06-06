"""
CCE.py

Description: Categorical Cross Entropy Loss

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from .LossBase import Loss


class CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Prevents division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        neg_log_probs = -np.log(confidences)
        return neg_log_probs
    
    def backward(self, derivatives, y_true):
        samples = len(derivatives)
        labels = len(derivatives[0]) # First row

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.derivatives = -y_true / derivatives
        self.derivative_inputs = self.derivatives / samples