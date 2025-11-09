import numpy as np

class Perceptron:
    """
    A simple implementation of the Perceptron algorithm — the fundamental unit of a neural network.
    Performs binary classification using a linear decision boundary.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_iters : int, default=1000
        Number of training iterations (epochs).
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _unit_step(self, x):
        """Activation function: Unit step"""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Train the perceptron using the Perceptron Learning Rule.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target class labels (0 or 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure binary targets (0,1)
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._unit_step(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """Predict class labels for given samples."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step(linear_output)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Generate 2D binary classification data
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train perceptron
    clf = Perceptron(learning_rate=0.01, n_iters=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Perceptron Classification Accuracy: {acc:.2f}")

    # Decision boundary visualization
    fig, ax = plt.subplots()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=25)
    x0_min, x0_max = np.min(X_train[:, 0]), np.max(X_train[:, 0])
    x1_min = (-clf.weights[0] * x0_min - clf.bias) / clf.weights[1]
    x1_max = (-clf.weights[0] * x0_max - clf.bias) / clf.weights[1]
    ax.plot([x0_min, x0_max], [x1_min, x1_max], 'k-', lw=2)
    plt.title("Perceptron Decision Boundary")
    plt.show()
