"""Generates datasets."""


import random
from typing import *

import numpy as np



class Dataset():
    def __init__(self, function, input_size: int, dataset_size: int, input_range: Tuple[int, int]) -> None:
        """
        Inputs:
        `function`: The function used to generate the dataset.
        `input_size`: The number of input arguments to the function.
        `dataset_size`: The number of data to generate.
        `input_range`: The minimum and maximum input values to input into the function.
        """

        self.x = np.random.rand(dataset_size, input_size, 1) * (input_range[1] - input_range[0]) + input_range[0]
        self.y = function(self.x)
    
    # def get_batch(start: int):
    #     x = self.x[batch]
    #     return 