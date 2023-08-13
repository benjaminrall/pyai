"""Gaussian Naive Bayes classifier class."""

import numpy as np

from pyai.preprocessing import LabelBinariser


class GaussianNB:
    """Naive Bayes classifier for classes that follow Gaussian distributions."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        """Constructs a new Gaussian Naive Bayes classifier.

        Parameters
        ----------
        epsilon : float, optional
            Small value to ensure numerical stability.
        """
        self.classes = None
        self.binariser = LabelBinariser()
        self.epsilon = epsilon

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the classifier according to given training data and labels.

        Parameters
        ----------
        x : np.ndarray
            Training vectors with the shape (n_samples, n_features).
        y : np.ndarray
            Target values with the shape (n_samples,).
        """
        # Ensures `y` is a numpy array and resets classes
        y = np.array(y)
        self.classes = None
        self.binariser_used = len(y.shape) == 1

        # Binarises the labels if necessary
        if self.binariser_used:
            y = self.binariser.fit_transform(y)
            self.classes = self.binariser.classes

        # Ensures labels are in one-hot encoded form
        if y.shape[1] == 1:
            y = np.hstack([1 - y, y])

        # Ensures classes attribute is not None
        if self.classes is None:
            self.classes = np.arange(y.shape[1])

        # Stores number of input features and classes
        self.n_features_in = x.shape[1]
        n_classes = len(self.classes)

        self.theta = np.zeros((n_classes, self.n_features_in))
        self.var = np.zeros((n_classes, self.n_features_in))
        self.class_count = np.zeros(n_classes)
        for i in range(n_classes):
            mask = y[:, i].astype(bool)
            x_feature = x[mask]
            self.theta[i] = np.mean(x_feature, axis=0)
            self.var[i] = np.var(x_feature, axis=0)
            self.class_count[i] = sum(mask)

        self.class_prior = self.class_count / np.sum(self.class_count)
        self.class_log_prior = np.log(self.class_prior)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generates output predictions for a set of inputs.

        Parameters
        ----------
        x : np.ndarray
            Test vectors with the shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted target values for `x`.
        """
        # Calculates the probabilities for each class
        probabilities = np.zeros((len(x), len(self.classes)))
        for i in range(probabilities.shape[0]):
            sample = x[i]
            for j in range(probabilities.shape[1]):
                mean, var = self.theta[j], self.var[j]

                v1 = 2 * np.pi * var
                v1[v1 == 0] += self.epsilon
                var[var == 0] += self.epsilon

                log_likelihood = -0.5 * (np.log(v1) + (np.square(sample - mean) / var))

                probabilities[i, j] = np.sum(log_likelihood) + self.class_log_prior[j]

        # Calculates the predictions for each test vector
        predictions = np.argmax(probabilities, axis=1)

        # If the binariser was used, return predictions in original form
        if self.binariser_used:
            return self.binariser.classes[predictions]

        # Otherwise, return prediction indices
        return predictions

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Returns the accuracy for the given test data and labels.

        Parameters
        ----------
        x : np.ndarray
            Test vectors with the shape (n_samples, n_features).
        y : np.ndarray
            True labels for `x` with the shape (n_samples,).

        Returns
        -------
        float
            The mean accuracy of the classifier over all input samples.
        """
        return np.sum(self.predict(x) == y) / len(y)
