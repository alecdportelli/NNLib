import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

from src.phugoid_nn.Layers.DenseLayer import Dense
from src.phugoid_nn.Layers.Dropout import Dropout
from src.phugoid_nn.Activations.ReLU import ReLU
from src.phugoid_nn.Activations.Sigmoid import Sigmoid
from src.phugoid_nn.Activations.Softmax_Loss_CCE import Softmax_Loss_CCE as SLCEE
from src.phugoid_nn.Optimizers.SGD import SGD
from src.phugoid_nn.Optimizers.Adagrad import Adagrad
from src.phugoid_nn.Optimizers.RMS import RMS
from src.phugoid_nn.Optimizers.Adam import Adam
from src.phugoid_nn.Loss.BinaryCrossEntropy import BinaryCrossEntropy


# Create dataset
X, y = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)

dense1 = Dense(2, 64, "Relu", weight_regularizer_L2=5e-4,
                            bias_regularizer_L2=5e-4)

activation1 = ReLU()

dense2 = Dense(64, 1, "Relu")

activation2 = Sigmoid()

loss_function = BinaryCrossEntropy()

# Create optimizer
optimizer = Adam(decay=5e-7)

# Train in loop
for epoch in range(10001):

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)

    # Calculate the data loss
    data_loss = loss_function.calculate(activation2.output, y)
    # Calculate regularization penalty
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.derivative_inputs)
    dense2.backward(activation2.derivative_inputs)
    activation1.backward(dense2.derivative_inputs)
    dense1.backward(activation1.derivative_inputs)

    # Update weights and biases
    optimizer.before_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.after_update_params()


# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)


# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
