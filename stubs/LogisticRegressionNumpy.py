"""Basic implementation of Logistic Regression using numpy only."""

import numpy as np


class LogisticRegression:
    """Logistic Regression"""

    def __init__(self):
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = True,
        learning_rate_decay_factor: float = 1.0,
        num_epochs: int = 100,
        track_loss_num_epochs: int = 10,
    ):
        """Train.

        Args:
            X_train (np.ndarray): Feature matrix. (N,d)
            y_train (np.ndarray): Target labels (0,1). (N,)
            learning_rate (float, optional): The initial learning rate. Defaults to 1e-3.
            learning_rate_decay (bool, optional): If True enables learning rate decay. Defaults to True.
            learning_rate_decay_factor (float, optional): The learning rate decay factor (1/(1+decay_factor*epoch)). Defaults to 1.0.
            num_epochs (int, optional): The number of epochs to train. Defaults to 100.
            track_loss_num_epochs (int, optional): Compute loss on training set once in k epochs. Defaults to 10.
        """
        pass

    def predict_proba(self, X: np.ndarray):
        """Predict the probability of the positive class (Pr(y=1)).

        Args:
            X (np.ndarray): Feature matrix. (N,d)

        Returns:
            y_pred_proba (np.ndarray): Predicted probabilities. (N,)
        """
        pass

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ):
        """Predict the label(0,1).

        Args:
            X (np.ndarray): Feature matrix. (N,d)
            add_colum_vector_for_bias (bool, optional): Add a column vector of ones to model the bias term. Defaults to True.
            threshold (float, optional): The threshold on the probabilit. Defaults to 0.5.

        Returns:
            y (np.ndarray): Prediction vector. (N,)
        """
        pass


def sample_data(N: int = 1000, d: int = 2, train_size: float = 0.7):
    """Generate sample train and test data.

    Args:
        N (int, optional): Number of samples. Defaults to 1000.
        d (int, optional): Number of features. Defaults to 2.
        train_size (float, optional): Fraction of instances in training. Defaults to 0.7.

    Returns:
        X_train : Features matrix for the train split. (N_train,d)
        y_train : Target vector for the train split. (N_train,)
        X_test : Features matrix for the test split. (N_test,d)
        y_test : Target vector for the test split. (N_test,)
    """
    N_positive = int(N / 2)
    N_negative = N - N_positive

    X_positive = np.random.multivariate_normal(
        mean=np.ones(d), cov=np.eye(d), size=N_positive
    )
    y_positive = np.ones(N_positive)

    X_negative = np.random.multivariate_normal(
        mean=-np.ones(d), cov=np.eye(d), size=N_negative
    )
    y_negative = np.zeros(N_negative)

    X = np.concatenate([X_positive, X_negative])
    y = np.concatenate([y_positive, y_negative])

    # Split into train and test.
    indices = np.random.permutation(N)
    N_train = int(N * train_size)
    train_idx, test_idx = indices[:N_train], indices[N_train:]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    # Sample traing data.
    X_train, y_train, X_test, y_test = sample_data(N=1000, d=3)
    print(X_train.shape)
    print(y_train.shape)

    # Classifier
    classifier = LogisticRegression()

    # Training
    classifier.fit(
        X_train,
        y_train,
        learning_rate=1e-2,
        learning_rate_decay=True,
        learning_rate_decay_factor=0.1,
        num_epochs=1000,
        track_loss_num_epochs=100,
    )
    print(f"[Estimated] Weight vector {classifier.weights_}")

    # Prediction
    y_train_pred = classifier.predict(X_train, threshold=0.5)
    y_test_pred = classifier.predict(X_test, threshold=0.5)

    # Accuracy
    accuracy_train = sum(y_train_pred == y_train) / len(y_train)
    print(f"Accuracy (train) = {accuracy_train:.5f}")
    accuracy_test = sum(y_test_pred == y_test) / len(y_test)
    print(f"Accuracy (test) = {accuracy_test:.5f}")
