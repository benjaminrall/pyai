"""Ordinal encoder class."""

import numpy as np


class OrdinalEncoder:
    """Encodes categorical features as an integer array."""

    def __init__(self, encoded_missing_value: int = np.nan) -> None:
        """Creates an OrdinalEncoder object.

        Parameters
        ----------
        encoded_missing_value : int, optional
            Value to fill missing values with when encoding data.
        """
        self.mapping: list[dict] = []
        self.categories: list[list] = []
        self.encoded_missing_value = encoded_missing_value

    def fit(self, x: np.ndarray) -> 'OrdinalEncoder':
        """Fits the ordinal encoder to a given array of categorical feature vectors.

        Parameters
        ----------
        x : np.ndarray
            Array used to determine the categories of each feature,
            with a shape of (n_samples, n_features).

        Returns
        -------
        OrdinalEncoder
            Fitted ordinal encoder instance.
        """
        self.n_features_in = x.shape[1]
        self.mapping = []
        self.categories = []

        for fi in range(self.n_features_in):
            # Extracts all possible feature elements from the input vectors
            feature = x[:, fi]
            categories = sorted(set(feature))

            # Stores mapping and categories for the feature
            self.mapping.append({f: i for i, f in enumerate(categories)})
            self.categories.append(np.array(categories))

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the given categorical feature vectors into integer arrays.

        Parameters
        ----------
        x : np.ndarray
            Array containing the categorical feature vectors to be encoded,
            with a shape of (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array containing the result of encoding the input samples,
            with a shape of (n_samples, n_features).
        """
        # Creates array to hold the encoded data
        encoded_data = np.empty((len(x), self.n_features_in), dtype=int)

        # Encodes each feature in the input vectors array
        for i in range(self.n_features_in):
            # Gets the mapping for the specific feature
            feature_mapping = self.mapping[i]

            # Encodes the feature column
            encode = np.vectorize(feature_mapping.get)
            encoded_data[:, i] = encode(x[:, i], self.encoded_missing_value)

        return encoded_data

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fits the encoder and transforms the given data.

        Parameters
        ----------
        x : np.ndarray
            Input samples to fit the encoder with and then transform,
            with a shape of (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array containing the result of encoding the input samples,
            with a shape of (n_samples, n_features).
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms encoded data back into its original representation.

        Parameters
        ----------
        x : np.ndarray
            Encoded data to be reverted to its original form,
            with a shape of (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array containing the result of decoding the input samples,
            with a shape of (n_samples, n_features).
        """
        # Creates arry to hold the decoded data
        decoded_data = np.empty((len(x), self.n_features_in), dtype=object)

        # Decodes each feature in the input vectors array
        for i in range(self.n_features_in):
            # Gets the categories for the specific feature
            feature_categories = self.categories[i]

            # Decodes the feature column
            decode = np.vectorize(feature_categories.__getitem__)
            decoded_data[:, i] = decode(x[:, i])

        return decoded_data
