import numpy as np

class SVM:
    """
    Linear Support Vector Machine (SVM) implemented from scratch.
    Trains using Stochastic Gradient Descent (SGD) to minimize hinge loss with L2 regularization.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size for parameter updates.
    lambda_param : float, default=0.01
        Regularization strength to prevent overfitting.
    n_iters : int, default=1000
        Number of epochs to run SGD.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM classifier using the hinge loss gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target class labels, expected as {0, 1} or {-1, 1}.
        """
        n_samples, n_features = X.shape
        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop (Stochastic Gradient Descent)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check margin condition
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # Correctly classified: only apply regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified or within margin: adjust w and b
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels {-1, 1}.
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Create a simple binary classification dataset
    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    # Train SVM
    clf = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    clf.fit(X, y)

    # Predict
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Model accuracy: {accuracy:.2f}")

    # Visualize decision boundary
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=25)
    x0_min, x0_max = np.min(X[:, 0]), np.max(X[:, 0])
    x1_min = get_hyperplane_value(x0_min, clf.w, clf.b, 0)
    x1_max = get_hyperplane_value(x0_max, clf.w, clf.b, 0)
    ax.plot([x0_min, x0_max], [x1_min, x1_max], "y--", label="Decision boundary")
    ax.plot([x0_min, x0_max], [get_hyperplane_value(x0_min, clf.w, clf.b, 1),
                               get_hyperplane_value(x0_max, clf.w, clf.b, 1)], "k", label="Support +1")
    ax.plot([x0_min, x0_max], [get_hyperplane_value(x0_min, clf.w, clf.b, -1),
                               get_hyperplane_value(x0_max, clf.w, clf.b, -1)], "k", label="Support -1")
    plt.legend()
    plt.title("Linear SVM Decision Boundary")
    plt.show()