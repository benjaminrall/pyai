"""Bernoulli Naive Bayes classifier class."""

import numpy as np

from pyai.preprocessing import LabelBinariser


class BernoulliNB:
    """Naive Bayes classifier for multivariate Bernoulli models.

    Designed for discrete data with binary features.
    """

    def __init__(self, alpha: float = 1.0, binarise_threshold: float = 0.0) -> None:
        """Constructs a new Bernoulli Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter.
        binarise_threshold : float, optional
            Threshold for the binarising of sample features.
        """
        self.alpha = alpha
        self.binarise_threshold = binarise_threshold
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

        # Binarises input vectors
        x = (x > self.binarise_threshold).astype(int)

        # Counts the classes and features
        self.feature_count = np.dot(y.T, x)
        self.class_count = np.sum(y, axis=0)

        # Applies Laplace smoothing to both counts
        smoothed_fc = self.feature_count + self.alpha
        smoothed_cc = self.class_count + self.alpha * 2

        # Calculates log probabilities
        self.feature_log_prob = np.log(smoothed_fc) - np.log(smoothed_cc.reshape((-1, 1)))
        self.class_log_prior = np.log(self.class_count) - np.log(np.sum(self.class_count))

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
        # Binarises input vectors
        x = (x > self.binarise_threshold).astype(int)

        # Computes log likelihoods using negated probability
        neg_prob = np.log(1 - np.exp(self.feature_log_prob))
        log_likelihoods = np.dot(x, (self.feature_log_prob - neg_prob).T)

        # Calculates the predictions
        posteriors = log_likelihoods + self.class_log_prior + np.sum(neg_prob, axis=1)
        predictions = np.argmax(posteriors, axis=1)

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
