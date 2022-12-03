from typing import *

import numpy as np
import matplotlib.pyplot as plt

from data import Dataset


def forward(x, W1, W2) -> Tuple[np.ndarray, np.ndarray]:
    h1 = W1 @ np.append(x, [[[1]]], axis=1)
    # Apply ReLU.
    h1 *= h1 > 0
    y = W2 @ np.append(h1, [[[1]]], axis=1)
    return h1, y

def backpropagate(loss: float, x: np.ndarray, h1: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate gradients for the second layer.
    G2 = 2 * loss * np.append(h1, [[[1]]], axis=1).transpose((0, 2, 1))

    # Calculate gradients for the first layer.
    x = np.append(x, [[[1]]], axis=1).transpose((0, 2, 1))
    G1 = 2 * loss * (h1 > 0) * W2[:, :, :4].transpose((0, 2, 1)) * x

    return G1, G2

def mse(prediction, label):
    return np.mean((prediction - label) ** 2)

def gradient_descent(W1, W2, G1, G2, learning_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    W1 = W1 - learning_rate * G1
    W2 = W2 - learning_rate * G2
    return W1, W2

def main(epochs: int, learning_rate: float, batch_size: int):
    # Initialize weights.
    W1 = (np.random.rand(1, 4, 3) - 0.5*0) * 1e-4
    W2 = (np.random.rand(1, 1, 5) - 0.5*0) * 1e-4

    # Create dataset.
    function = lambda x: (x[:, 0:1, :] + x[:, 1:2, :])
    dataset_size = 6400
    dataset = Dataset(function=function, input_size=2, dataset_size=dataset_size, input_range=[0, 1])

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}')

        total_loss = 0
        for batch in range(0, dataset_size, batch_size):
            # Get the input and label data for the current batch, with shapes (batch, ..., 1).
            x = dataset.x[batch:batch+batch_size, ...]
            label = dataset.y[batch:batch+batch_size, ...]
            
            # Make predictions with the model.
            h1, y = forward(x, W1, W2)

            # Calculate loss.
            loss = mse(y, label)
            # Calculate gradients for the weights.
            G1, G2 = backpropagate(loss, x, h1, W1, W2)
            # Update weights using gradient descent.
            W1, W2 = gradient_descent(W1, W2, G1, G2, learning_rate)

            print(f'Loss: {loss:,.2e}', end='\r')
        print()


if __name__ == '__main__':
    main(epochs=10, learning_rate=1e-5, batch_size=1)