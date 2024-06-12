"""
ModelBase.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np


class ModelBase:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        
