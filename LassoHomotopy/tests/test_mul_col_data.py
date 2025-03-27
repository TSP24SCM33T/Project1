
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

def test_multicollinear():


    np.random.seed(0)
    n_samples = 100
    X_base = np.random.randn(n_samples, 1)
    X = np.hstack([X_base, X_base + 0.01 * np.random.randn(n_samples, 1), X_base + 0.02 * np.random.randn(n_samples, 1)])
    y = 5 * X[:, 0] + np.random.randn(n_samples)

    model = LassoHomotopyModel(regularization_param=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((y - preds) ** 2)
    r2 = r2_score(y, preds)

    print("\n=== Multicollinear Test ===")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    plt.bar(range(len(results.coef_)), results.coef_)
    plt.title("Recovered Coefficients - Multicollinear Test")
    plt.show()


test_multicollinear()