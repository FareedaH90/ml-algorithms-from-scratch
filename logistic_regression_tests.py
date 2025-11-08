import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Import your custom logistic regression class
from logistic_regression import LogisticRegression

# Load dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


# Accuracy function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Create model instance
regressor = LogisticRegression(lr=0.0001, n_iters=1000)

print("Training logistic regression model...")
regressor.fit(X_train, y_train)

# Predict on test data
predictions = regressor.predict(X_test)

# Evaluate
acc = accuracy(y_test, predictions)
print(f"LR classification accuracy: {acc:.4f}")
