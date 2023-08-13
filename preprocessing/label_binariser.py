"""Label binariser class."""

import numpy as np


class LabelBinariser:
    """Binarises labels in a one-vs-all fashion."""

    def __init__(self) -> None:
        self.classes: np.ndarray = np.empty(0)

    def fit(self, y: np.ndarray) -> 'LabelBinariser':
        """Fits the label binariser to a given array of classes.

        Parameters
        ----------
        y : np.ndarray
            Array of shape (n_samples,) containing the classes used
            to fit the label binariser.

        Returns
        -------
        LabelBinariser
            Fitted label binariser instance.
        """
        # Creates a sorted list of the unique classes in the input
        self.classes = np.array(sorted(set(y)))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transforms multi-class labels to binary labels.

        Parameters
        ----------
        y : np.ndarray
            Array of shape (n_samples,) containing classes to be
            transformed into binary labels.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) containing the
            result of converting the input classes to binary labels.
            Shape will be (n_samples, 1) for binary problems.
        """
        # Ensures `y` is a numpy array
        y = np.array(y)

        # Binary classes transform to a column vector
        if len(self.classes) == 2:
            binary_labels = (y == self.classes[1]).astype(int)
            return binary_labels.reshape(-1, 1)

        # Generates binary labels for generic multi-class case
        binary_labels = np.array([y == c for c in self.classes], dtype=int).T
        return binary_labels

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fits the binariser to and transforms a given list of classes.

        Parameters
        ----------
        y : np.ndarray
            Array of shape (n_samples,) containing classes to be
            transformed into binary labels.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) containing the
            result of converting the input classes to binary labels.
            Shape will be (n_samples, 1) for binary problems.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transforms binary labels back to multi-class labels.

        Parameters
        ----------
        y : np.ndarray
            Binary labels to be converted back to original form,
            with shape (n_samples, n_classes).

        Returns
        -------
        np.ndarray
            Array with shape (n_samples,) containing the class represented
            by each binary label.
        """
        # Ensures input is a numpy array
        y = np.array(y)

        # Inverse of column vector in case of binary classes
        if len(self.classes) == 2:
            return self.classes[y].reshape(-1)

        # Inverse of binary labels for generic multi-class case
        return np.array([self.classes[np.where(row == 1)][0] for row in y])
