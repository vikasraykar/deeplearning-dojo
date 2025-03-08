"""Basic implementation of Linear Regression using only numpy.
"""

import numpy as np


class LinearRegression:
    """Linear Regression."""

    def __init__(self):
        pass

    def __add_colum_vector_for_bias(self, X: np.ndarray) -> np.ndarray:
        """Add a column vector of ones to model the bias term."""
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def __loss(self, X, y, add_colum_vector_for_bias):
        """Loss function to minimize."""
        y_pred = self.pred(X, add_colum_vector_for_bias)
        loss = 0.5 * ((y - y_pred) ** 2).sum()
        return loss

    def pred(self, X: np.ndarray, add_colum_vector_for_bias: bool = True) -> np.ndarray:
        """Predction.

        Args:
            X (np.ndarray): Features matrix. (N,d)
            add_colum_vector_for_bias (bool, optional): Add a column vector of ones to model
            the bias term. Defaults to True.

        Returns:
            y (np.ndarray): Prediction vector. (N,)
        """
        if add_colum_vector_for_bias:
            y_pred = np.dot(self.__add_colum_vector_for_bias(X), self.weights_)
        else:
            y_pred = np.dot(X, self.weights_)

        return y_pred

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = True,
        learning_rate_decay_factor: float = 1.0,
        num_epochs: int = 100,
        track_loss_num_epochs: int = 100,
    ):
        """Training.

        Args:
            X_train (np.ndarray): Features matrix. (N,d)
            y_train (np.ndarray): Target vector. (N,)
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            learning_rate_decay (bool, optional): If True does learning rate deacy. Defaults to True.
            learning_rate_decay_factor (float, optional): The deacay factor (lr=lr/(1+decay_factor*epoch)). Defaults to 1.0.
            num_epochs (int, optional): Number of epochs. Defaults to 100.
            track_loss_num_epochs (int, optional): Compute loss on training set once in k epochs. Defaults to 100.
        """

        # Add a column vector of ones to model the bias term.
        X_train = self.__add_colum_vector_for_bias(X_train)
        N, d = X_train.shape

        # Initialize the parameters.
        self.weights_ = np.random.rand(d)

        # Start training.
        self.learning_rate_ = learning_rate
        self.loss_train_ = []
        for epoch in range(0, num_epochs):
            # Compute the gradient.
            y_pred_train = self.pred(X_train, add_colum_vector_for_bias=False)
            gradient = np.dot(X_train.T, y_pred_train - y_train)

            # Learning rate decay.
            if learning_rate_decay:
                self.learning_rate_ = learning_rate / (
                    1.0 + learning_rate_decay_factor * epoch
                )

            # Gradient descent updates.
            self.weights_ -= self.learning_rate_ * gradient

            # Compute loss on training set once in K epochs.
            if epoch % track_loss_num_epochs == 0:
                loss = self.__loss(X_train, y_train, add_colum_vector_for_bias=False)
                self.loss_train_.append(loss)
                print(
                    f"epoch {epoch:3d}/{num_epochs:3d} loss = {loss:.3f} learning_rate {self.learning_rate_:.2e}"
                )


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
        w : Weight vector. (d,)
        b : Bias term. (1,)
    """

    # Draw random samples from a multivariate normal distribution as features.
    X = np.random.multivariate_normal(mean=np.ones(d), cov=np.eye(d), size=N)

    # Create a random weight vector and bias term.
    w = np.random.rand(d)
    b = 0.1

    # Generate the target.
    y = np.dot(X, w) + b

    # Add some Gaussian noise.
    y += np.random.normal(0, 0.1, N)

    # Split into train and test.
    indices = np.random.permutation(N)
    N_train = int(N * train_size)
    train_idx, test_idx = indices[:N_train], indices[N_train:]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, y_train, X_test, y_test, w, b


if __name__ == "__main__":
    """Linear Regression."""

    # Sample train and test dataset.
    X_train, y_train, X_test, y_test, w, b = sample_data(N=1000, d=3)
    print(X_train.shape)
    print(y_train.shape)
    print(f"[Actual] Weight vector {w}")
    print(f"[Actual] Bias term {b}")

    # Linear Regression
    regressor = LinearRegression()

    # Train
    regressor.fit(
        X_train,
        y_train,
        learning_rate=1e-3,
        learning_rate_decay=True,
        learning_rate_decay_factor=0.1,
        num_epochs=1000,
        track_loss_num_epochs=100,
    )
    print(f"[Estimated] Weight vector {regressor.weights_[:-1]}")
    print(f"[Estimated] Bias term {regressor.weights_[-1]}")

    # Predict
    y_train_pred = regressor.pred(X_train)
    y_test_pred = regressor.pred(X_test)

    # Compute the MSE
    mse_train = ((y_train - y_train_pred) ** 2).mean()
    mse_test = ((y_test - y_test_pred) ** 2).mean()
    print(f"MSE (train) = {mse_train:.5f}")
    print(f"MSE (test) = {mse_test:.5f}")
