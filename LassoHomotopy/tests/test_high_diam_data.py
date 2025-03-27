import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import csv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)
from model.LassoHomotopy import LassoHomotopyModel


def r2_score(y_true, y_pred):
    """Compute R² Score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def test_small_sample_high_dim():

    np.random.seed(101)
    X = np.random.randn(10, 50)  # Small sample, high-dimensional
    true_coef = np.zeros(50)
    true_coef[[5, 10, 20]] = [4.0, -3.5, 2.0]
    y = X @ true_coef + np.random.randn(10) * 0.1

    model = LassoHomotopyModel(regularization_param=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((y - preds) ** 2)
    r2 = r2_score(y, preds)

    print("\n=== Small Sample High Dim Test ===")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Non-zero Coefficients: {np.sum(results.coef_ != 0)}")

    plt.bar(range(len(results.coef_)), results.coef_)
    plt.title("Recovered Coefficients - Small Sample High Dim Test")
    plt.show()


test_small_sample_high_dim()