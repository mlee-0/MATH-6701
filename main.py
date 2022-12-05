from typing import *

import numpy as np
import matplotlib.pyplot as plt

from data import Dataset


def forward(x, W1, W2) -> Tuple[np.ndarray, np.ndarray]:
    """Make a prediction using the model with the given inputs."""

    # Layer 1.
    h1 = W1 @ np.append(x, np.ones((x.shape[0], 1, x.shape[2])), axis=1)
    # ReLU.
    h1 *= h1 > 0

    # Layer 2.
    y = W2 @ np.append(h1, np.ones((h1.shape[0], 1, h1.shape[2])), axis=1)

    return h1, y

def backpropagate(y: np.ndarray, label: np.ndarray, x: np.ndarray, h1: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate gradients."""

    # Calculate gradients for the second layer.
    G2 = 2 * (y - label) * np.append(h1, np.ones((h1.shape[0], 1, h1.shape[2])), axis=1).transpose((0, 2, 1))
    G2 = np.mean(G2, axis=0)

    # Calculate gradients for the first layer.
    x = np.append(x, np.ones((x.shape[0], 1, x.shape[2])), axis=1).transpose((0, 2, 1))
    G1 = 2 * (y - label) * (h1 > 0) * W2[:, :, :4].transpose((0, 2, 1)) * x
    G1 = np.mean(G1, axis=0)

    return G1, G2

def mse(prediction, label):
    """Mean squared error loss."""

    return np.mean((prediction - label) ** 2)

def gradient_descent(W1, W2, G1, G2, learning_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Update and return weights."""

    W1 = W1 - learning_rate * G1
    W2 = W2 - learning_rate * G2

    return W1, W2

def main(epochs: int, learning_rate: float, batch_size: int, dataset_function: Callable):
    """Train the model."""

    # Initialize weights as small random values.
    W1 = (np.random.rand(1, 4, 3) - 0.5*0) * 1e-4
    W2 = (np.random.rand(1, 1, 5) - 0.5*0) * 1e-4

    # Create the dataset.
    dataset_size = 10000
    dataset = Dataset(function=dataset_function, dataset_size=dataset_size, batch_size=batch_size, input_size=2, input_range=[-5, 5])

    # Initialize lists of loss values.
    training_loss = []
    testing_loss = []

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}')

        dataset.shuffle()

        # Train the model.
        total_loss = 0
        for batch, (x, label) in enumerate(dataset.training(), 1):
            # Make predictions with the model.
            h1, y = forward(x, W1, W2)

            # Calculate loss.
            loss = mse(y, label)
            total_loss += loss
            # Calculate gradients for the weights.
            G1, G2 = backpropagate(y, label, x, h1, W1, W2)
            # Update weights using gradient descent.
            W1, W2 = gradient_descent(W1, W2, G1, G2, learning_rate)

            if batch % 100 == 0:
                print(f'Batch {batch}: {loss:,.2e}', end='\r')
        
        average_loss = total_loss / batch
        training_loss.append(average_loss)
    
        # Test the model.
        total_loss = 0
        for batch, (x, label) in enumerate(dataset.testing(), 1):
            # Make predictions with the model.
            h1, y = forward(x, W1, W2)

            # Calculate loss.
            loss = mse(y, label)
            total_loss += loss

            if batch % 100 == 0:
                print(f'Batch {batch}: {loss:,.2e}', end='\r')
        
        average_loss = total_loss / batch
        testing_loss.append(average_loss)

        print(f'Loss: {training_loss[-1]:,.2e} (training), {testing_loss[-1]:,.2e} (testing)')

        # Visualize results.
        pass

    
    # Plot the loss values over each iteration.
    plt.figure()
    plt.semilogy(range(1, epochs+1), training_loss, '.-', label='Training')
    plt.semilogy(range(1, epochs+1), testing_loss, '.-', label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sum_function = lambda x1, x2: x1 + x2
    square_function = lambda x1, x2: x1 ** 2 + x2 ** 2
    sin_function = lambda x1, x2: np.sin(x1 + x2)
    exp_function = lambda x1, x2: np.exp(x1 + x2)

    main(epochs=50, learning_rate=1e-4, batch_size=10, dataset_function=square_function)