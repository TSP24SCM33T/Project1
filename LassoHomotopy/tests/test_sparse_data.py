import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import csv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt



# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)
from model.LassoHomotopy import LassoHomotopyModel


def r2_score(y_true, y_pred):
    """Compute R² Score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def test_synthetic_sparse():

    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[[2, 5, 15]] = [3.0, -2.0, 1.5]
    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    model = LassoHomotopyModel(regularization_param=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((y - preds) ** 2)
    r2 = r2_score(y, preds)
    non_zero = np.sum(results.coef_ != 0)

    print("\n=== Synthetic Sparse Test ===")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Expected Non-Zeros: 3 | Model Non-Zeros: {non_zero}")

    plt.bar(range(len(results.coef_)), results.coef_)
    plt.title("Recovered Coefficients - Sparse Test")
    plt.show()


test_synthetic_sparse()