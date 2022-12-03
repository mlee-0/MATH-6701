"""Generates datasets."""


import random
from typing import *

import numpy as np



class Dataset():
    def __init__(self, function, dataset_size: int, batch_size: int, input_size: int, input_range: Tuple[int, int]) -> None:
        """
        Inputs:
        `function`: The function used to generate the dataset.
        `dataset_size`: The number of data to generate.
        `batch_size`: Number of data to train on at once.
        `input_size`: The number of input arguments to the function.
        `input_range`: The minimum and maximum input values to input into the function.
        """

        assert dataset_size % batch_size == 0, f'The dataset size must divide evenly into the batch size.'

        self.x = np.random.rand(dataset_size, input_size, 1) * (input_range[1] - input_range[0]) + input_range[0]
        self.y = function(*[self.x[:, i:i+1, :] for i in range(input_size)])

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.training_ratio = 0.8

    def training(self) -> Generator:
        """Return a generator of training data."""
        count = round(self.training_ratio * self.dataset_size)
        return ((self.x[i:i+self.batch_size, ...], self.y[i:i+self.batch_size, ...]) for i in range(0, count, self.batch_size))
    
    def testing(self) -> Iterable:
        """Return a generator of testing data."""
        count = round((1 - self.training_ratio) * self.dataset_size)
        return ((self.x[i:i+self.batch_size, ...], self.y[i:i+self.batch_size, ...]) for i in range(self.dataset_size - count, self.dataset_size, self.batch_size))