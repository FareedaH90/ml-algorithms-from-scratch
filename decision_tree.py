"""
Decision Tree Classifier from Scratch
Author: Fareeda Hamid
Description:
    This implementation builds a Decision Tree Classifier using Entropy
    and Information Gain (ID3-like algorithm). It recursively splits the
    dataset into purer subsets based on the feature and threshold that 
    reduce uncertainty (entropy) the most.
"""

import numpy as np
from collections import Counter


# --------------------------------------------------------------
# 1️⃣ Utility: Calculate Entropy
# --------------------------------------------------------------
def entropy(y):
    """
    Compute the entropy of a label array y.

    Entropy measures how mixed (impure) a dataset is.
    - If all samples are from one class => Entropy = 0 (pure)
    - If classes are evenly mixed => Entropy = 1 (impure)
    """
    proportions = np.bincount(y) / len(y)       # class probabilities
    return -np.sum(p * np.log2(p) for p in proportions if p > 0)


# --------------------------------------------------------------
# 2️⃣ Node Class: Represents each node in the tree
# --------------------------------------------------------------
class Node:
    """
    Each Node stores:
        - feature: the index of the feature used for splitting
        - threshold: the feature value used to split
        - left / right: child nodes
        - value: assigned if this is a leaf node (no more splits)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """Return True if this node is a leaf node (no further splits)."""
        return self.value is not None

    def __repr__(self):
        """Readable string representation for debugging/printing."""
        if self.is_leaf():
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"


# --------------------------------------------------------------
# 3️⃣ Decision Tree Classifier
# --------------------------------------------------------------
class DecisionTree:
    """
    A simple Decision Tree Classifier built from scratch.
    Splits data recursively using the feature that gives
    the highest Information Gain (i.e., largest entropy reduction).
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ----------------------------------------------------------
    # Fit method: builds the tree recursively
    # ----------------------------------------------------------
    def fit(self, X, y):
        """
        Fit the decision tree model to the training data.
        """
        self.root = self._grow_tree(X, y)

    # ----------------------------------------------------------
    # Predict method: predicts labels for given samples
    # ----------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels for all samples in X.
        """
        return np.array([self._predict_row(x, self.root) for x in X])

    # ----------------------------------------------------------
    # Recursive tree construction
    # ----------------------------------------------------------
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.
        Stops when:
        - All samples have the same class (pure node)
        - Maximum depth is reached
        - Too few samples to continue splitting
        """
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        # Stop condition: pure node or constraints reached
        if (len(unique_labels) == 1 or
            depth >= self.max_depth or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split (feature + threshold)
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            # No valid split, return a leaf
            return Node(value=self._most_common_label(y))

        # Split data into left and right subsets
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        # Recursively grow left and right subtrees
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        # Return internal node
        return Node(feature=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child)

    # ----------------------------------------------------------
    # Find the best feature and threshold for splitting
    # ----------------------------------------------------------
    def _best_split(self, X, y):
        """
        Loop through each feature and threshold, compute
        Information Gain, and return the best split.
        """
        best_gain = -1
        split_feature, split_threshold = None, None
        parent_entropy = entropy(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx

                # Skip invalid splits (empty child)
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                # Weighted average child entropy
                n = len(y)
                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                e_left, e_right = entropy(y[left_idx]), entropy(y[right_idx])
                child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

                # Information Gain = Entropy before - Entropy after
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_idx
                    split_threshold = threshold

        return split_feature, split_threshold

    # ----------------------------------------------------------
    # Predict helper (traverse tree for one sample)
    # ----------------------------------------------------------
    def _predict_row(self, x, node):
        """
        Traverse the tree for a single sample until a leaf is reached.
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)

    # ----------------------------------------------------------
    # Utility: return majority class in a leaf node
    # ----------------------------------------------------------
    def _most_common_label(self, y):
        """
        Return the most common label in y (majority vote).
        """
        return Counter(y).most_common(1)[0][0]


# --------------------------------------------------------------
# 4 Example Usage
# --------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and train Decision Tree
    tree = DecisionTree(max_depth=4)
    tree.fit(X_train, y_train)

    # Predict on test data
    preds = tree.predict(X_test)

    # Evaluate accuracy
    acc = accuracy_score(y_test, preds)
    print("Decision Tree Accuracy:", round(acc, 3))