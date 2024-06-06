import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

from src.phugoid_nn.Layers.DenseLayer import Dense
from src.phugoid_nn.Activations.ReLU import ReLU
from src.phugoid_nn.Activations.Softmax_Loss_CCE import Softmax_Loss_CCE as SLCEE

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(num_inputs=2, num_neurons=3, activation="relu")
activation1 = ReLU()

dense2 = Dense(num_inputs=3, num_neurons=3, activation="relu")
loss_activation = SLCEE()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print(f"Loss {loss}")

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print(f"accuracy: {accuracy}")

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.derivative_inputs)
activation1.backward(dense2.derivative_inputs)
dense1.backward(activation1.derivative_inputs)