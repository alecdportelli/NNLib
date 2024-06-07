import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

from src.phugoid_nn.Layers.DenseLayer import Dense
from src.phugoid_nn.Activations.ReLU import ReLU
from src.phugoid_nn.Activations.Softmax_Loss_CCE import Softmax_Loss_CCE as SLCEE
from src.phugoid_nn.Optimizers.SGD import SGD
from src.phugoid_nn.Optimizers.Adagrad import Adagrad
from src.phugoid_nn.Optimizers.RMS import RMS
from src.phugoid_nn.Optimizers.Adam import Adam


# Create the dataset
X, y = spiral_data(samples=100, classes=3)

# Create the layers
dense1 = Dense(
    num_inputs=2, 
    num_neurons=64, 
    activation="relu", 
    weight_regularizer_L2=5e-4, 
    bias_regularizer_L2=5e-4
    )
dense2 = Dense(num_inputs=64, num_neurons=3, activation="relu")

# Create the activation functions
activation1 = ReLU()
loss_activation = SLCEE()

# Create the optimizer
# NOTE: Default learning rate is 1
optimizer = Adam(learning_rate=0.05, decay=5e-7)

# Set the number of epochs
NUM_EPOCHS = 80001

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
    data_loss = loss_activation.forward(dense2.output, y)

    # Data regularization for penalties of high weight values
    loss_regularization = \
        loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)
    
    # Total loss
    loss = loss_regularization + data_loss

    # Calculate predictions (axis 1 is row)
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} ---- Loss:" 
              f"{loss:.3f} ---- "
              f"Data loss: {data_loss} ---- "
              f"Reg loss: {loss_regularization:.3f} ---- "
              f"Accuracy: {accuracy:.3f} ---- "
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





# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


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