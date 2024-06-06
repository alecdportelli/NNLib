import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

from src.phugoid_nn.Layers.DenseLayer import Dense
from src.phugoid_nn.Activations.ReLU import ReLU
from src.phugoid_nn.Activations.Softmax_Loss_CCE import Softmax_Loss_CCE as SLCEE
from src.phugoid_nn.Optimizers.SGD import SGD


# Create the dataset
X, y = spiral_data(samples=100, classes=3)

# Create the layers
dense1 = Dense(num_inputs=2, num_neurons=64, activation="relu")
dense2 = Dense(num_inputs=64, num_neurons=3, activation="relu")

# Create the activation functions
activation1 = ReLU()
loss_activation = SLCEE()

# Create the optimizer
# NOTE: Default learning rate is 1
optimizer = SGD(learning_rate=1, decay=1e-3, momentum=0.9)

# Set the number of epochs
NUM_EPOCHS = 10000

# Lists for plotting 
epochs = []
losses = []
accuracies = []
lrs = []

# Training loop
for epoch in range(NUM_EPOCHS):
    # Perform the forward pass on first layer
    dense1.forward(X)

    # Activation function from first layer
    activation1.forward(dense1.output)

    # Repeat for second layer 
    dense2.forward(activation1.output)

    # Calculate loss from second layer and 'y' which
    # is the acutal data from the function
    loss = loss_activation.forward(dense2.output, y)

    # Calculate predictions (axis 1 is row)
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} ---- Loss:" 
              f"{loss:.3f} ---- "
              f"Accuracy: {accuracy:.3f}"
              f"--- LR: {optimizer.current_learning_rate}")
    
    # Append for plotting
    epochs.append(epoch)
    losses.append(loss)
    accuracies.append(accuracy)
    lrs.append(optimizer.current_learning_rate)

    # Perform the back propogation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.derivative_inputs)
    activation1.backward(dense2.derivative_inputs)
    dense1.backward(activation1.derivative_inputs)

    # Use optimizer to update weights and biases
    optimizer.before_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.after_update_params()

import matplotlib.pyplot as plt

# Create figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 2 rows, 1 column

# Plot data on the first subplot
ax1.plot(epochs, losses, color='blue', label='Losses')
ax1.set_title('Loss Over Epochs')
ax1.legend()

# Plot data on the second subplot
ax2.plot(epochs, accuracies, color='red', label='Accuracies')
ax2.set_title('Accuracies Over Epochs ')
ax2.legend()

# Plot data on thirs
ax3.plot(epochs, lrs, color='green', label='Learning Rates')
ax3.set_title('LR Over Epochs')
ax3.legend()

# Add shared x-axis label
fig.suptitle('Training Results')

# Show the plot
plt.show()