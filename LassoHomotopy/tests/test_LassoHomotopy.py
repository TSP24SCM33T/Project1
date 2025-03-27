import csv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)

print("Relative path being appended:", relative_path)

from model.LassoHomotopy import LassoHomotopyModel

def test_predict_with_plots():
    model = LassoHomotopyModel(
        regularization_param=0.1,  # Adjust if needed for sparsity
        max_iter=500,
        tol=1e-6
    )

    data = []
    csv_path = os.path.join(os.path.dirname(__file__), 'small_test.csv')
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract X and y
    X = np.array([
        [float(v) for k, v in datum.items() if k.startswith('x')]
        for datum in data
    ], dtype=float)
    y = np.array([float(datum['y']) for datum in data], dtype=float)

    # Fit the model
    results = model.fit(X, y)
    preds = results.predict(X)

    # ---------------------- Plot 1: y vs Predictions ----------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(y, preds, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Fit')
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('LASSO Predictions vs Actual (small_test.csv)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------- Plot 2: Coefficient Magnitudes ----------------------
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(results.coef_) + 1), results.coef_)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Learned Coefficients')
    plt.grid(True)
    plt.show()

    # ---------------------- Optional: Correlation Heatmap ----------------------
    corr = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Heatmap (X)')
    plt.show()

    # ---------------------- MSE and Summary ----------------------
    residuals = y - preds
    mse = np.mean(residuals ** 2)
    print("Mean Squared Error (MSE):", mse)

    coef = results.coef_
    nonzero_count = np.sum(coef != 0)
    zero_count = len(coef) - nonzero_count
    print(f"Number of nonzero coefficients: {nonzero_count}")
    print(f"Number of zero coefficients:    {zero_count}")

    # ---------------------- Sanity Checks ----------------------
    assert preds.shape[0] == X.shape[0], "Predictions should match number of rows."
    assert np.all(np.isfinite(preds)), "Predictions must be finite."
    assert mse < 1e4, "MSE is too large; possible issue."

if __name__ == "__main__":
    test_predict_with_plots()
