"""
K-Folds cross-validator
"""

import numpy as np

class KFold:
    """
    K-Folds cross-validator
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """
        Initializes the KFold class with the given parameters.

        Parameters:
        - n_splits: Number of folds. Must be at least 2.
        - shuffle: Whether to shuffle the data before splitting into batches.
        - random_state: If shuffle is True, seed for the random number generator.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x):
        """
        Generates indices to split data into training and test sets.

        Parameters:
        - X: Array-like, shape (n_samples,). Training data.

        Yields:
        - train_indices: The training set indices for that split.
        - test_indices: The testing set indices for that split.
        """
        n_samples = len(x)  # Number of samples in the dataset
        indices = np.arange(n_samples)  # Array of sample indices

        if self.shuffle:
            # If shuffle is True, shuffle the indices with the provided random state
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # Calculate the size of each fold
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        # Distribute the remainder of samples across the first few folds
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0  # Starting index for the first fold

        for fold_size in fold_sizes:
            # Define the test indices for the current fold
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]

            # Define the train indices for the current fold
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            # Yield sorted train and test indices
            yield np.sort(train_indices), np.sort(test_indices)

            # Update the starting index for the next fold
            current = stop

    def get_n_splits(self):
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
