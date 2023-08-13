"""Categorical Naive Bayes classifier class."""

import numpy as np

from pyai.preprocessing import LabelBinariser


class CategoricalNB:
    """Naive Bayes classifier for categorical features."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Constructs a new Categorical Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter.
        """
        self.alpha = alpha
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

        # Stores number of input features and classes
        self.n_features_in = x.shape[1]
        n_classes = len(self.classes)

        # Loops through all features
        self.feature_count = []
        for i in range(self.n_features_in):
            # Extracts the individual feature column
            x_feature = x[:, i].astype(int)

            # Counts each possible feature category for each class
            count_feature = []
            for j in range(n_classes):
                # Creates a mask for the class
                mask = y[:, j].astype(bool)

                # Counts the number of each feature category for the class
                counts = np.bincount(x_feature[mask])
                count_feature.append(counts)

            # Ensures that all arrays are padded to the same length
            max_length = max(len(arr) for arr in count_feature)
            count_feature = np.vstack([
                np.pad(arr, (0, max_length - arr.shape[0])) for arr in count_feature
            ])
            self.feature_count.append(count_feature)

        # Counts the classes
        self.class_count = np.sum(y, axis=0)

        # Calculates log probabilities for each feature
        self.feature_log_prob = []
        for i in range(self.n_features_in):
            # Applies Laplace smoothing to the feature count
            smoothed_count = self.feature_count[i] + self.alpha
            total_count = np.sum(smoothed_count, axis=1, keepdims=True)

            # Calculates the log probability for the feature
            log_probability = np.log(smoothed_count) - np.log(total_count)
            self.feature_log_prob.append(log_probability)

        # Calculates the log class prior
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
        # Calculates the log likelihoods for each feature
        log_likelihoods = np.zeros((len(x), len(self.classes)))
        for i in range(self.n_features_in):
            category = x[:, i]
            log_likelihoods += self.feature_log_prob[i][:, category].T

        # Calculates the predictions for each test vector
        posteriors = log_likelihoods + self.class_log_prior
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
