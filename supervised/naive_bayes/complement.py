"""Complement Naive Bayes classifier class."""

import numpy as np

from pyai.preprocessing import LabelBinariser


class ComplementNB:
    """Complement Naive Bayes classifier for multinomial models.

    Particularly suited for imbalanced datasets.
    """

    def __init__(self, alpha: float = 1.0, norm: bool = False) -> None:
        """Constructs a new Complement Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter.
        norm : bool, optional
            Whether to perform a second normalisation of the weights.
        """
        self.alpha = alpha
        self.norm = norm
        self.classes = None
        self.binariser = LabelBinariser()

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

        # Stores number of input features
        self.n_features_in = x.shape[1]

        # Counts the classes and features
        self.feature_count = np.dot(y.T, x)
        self.feature_all = np.sum(self.feature_count, axis=0)
        self.class_count = np.sum(y, axis=0)

        # Calculates complement count and applies Laplace smoothing
        comp_count = self.feature_all + self.alpha - self.feature_count
        total_count = np.sum(comp_count, axis=1, keepdims=True)

        # Calculates log probabilities
        feature_log_prob = np.log(comp_count) - np.log(total_count)
        self.class_log_prior = np.log(self.class_count) - np.log(np.sum(self.class_count))

        # Normalises feature log probabilities if necessary
        if self.norm:
            summed_feature_prob = np.sum(feature_log_prob, axis=1, keepdims=True)
            self.feature_log_prob = feature_log_prob / summed_feature_prob
        else:
            self.feature_log_prob = -feature_log_prob

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
        # Calculates the predictions
        log_likelihoods = np.dot(x, self.feature_log_prob.T)
        if len(self.classes) == 1:
            log_likelihoods += self.class_log_prior
        predictions = np.argmax(log_likelihoods, axis=1)

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
