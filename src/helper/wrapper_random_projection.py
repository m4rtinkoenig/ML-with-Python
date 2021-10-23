from typing import Optional
import numpy as np


class RandomProjectionWrapper:
    """
    Wrapper class that allows to apply a random projection in front of sklearn classifiers.
    The sklearn classifier should have methods score, fit, and predict_proba.
    """

    def __init__(
        self,
        classifier,
        proj_dim: Optional[int] = 64,
        n_subset: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.dim = proj_dim
        self.n_subset = n_subset
        self.random_state = random_state
        self.classifier = classifier
        self.oth_mat = None
        if random_state is None:
            self.random_state = np.random.randint(1, 100)

    def score(self, X, y):
        """
        Method that computes the accuracy of the classifier
        """
        if self.n_subset:
            X, y = self._subset(X, y)
        X_proj = self._project(X)
        return self.classifier.score(X_proj, y)

    def fit(self, X, y):
        """
        Method that "fits" data to the classifier.
        Mainly, this method reads the dimensions
        needed for the random projection and initializes
        the orthogonal project matrix
        """
        shape = X.shape
        self.n_features = shape[1]

        if self.n_subset:
            X, y = self._subset(X, y)
        X_proj = self._project(X)
        self.classifier.fit(X_proj, y)
        return self

    def predict_proba(self, X):
        """
        Method that predicts data X and
        returns probabilities for each class
        """
        X_proj = self._project(X)
        return self.classifier.predict_proba(X_proj)

    def _subset(self, X, y):
        # use only a subset of the training data
        subset = np.random.choice(len(X), self.n_subset)
        return X[subset], y[subset]

    def _project(self, X):
        """
        project data using a random orthogonal matrix
        """
        if self.random_state:
            np.random.seed(self.random_state)
        # generate a random orthogonal matrix
        if self.oth_mat is None:
            H = np.random.rand(self.n_features, self.dim)
            u, s, vh = np.linalg.svd(H, full_matrices=False)
            self.oth_mat = u @ vh
        return X @ self.oth_mat
