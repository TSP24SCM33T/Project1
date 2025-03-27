import csv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)


from model.LassoHomotopy import LassoHomotopyModel

def test_collinear_data_with_plots():
    model = LassoHomotopyModel(
        regularization_param=0.01,
        max_iter=500,
        tol=1e-6
    )

    data = []
    csv_path = os.path.join(os.path.dirname(__file__), 'collinear_data.csv')

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Build X (Nx10 matrix) and y:
    X = np.array([
        [float(row[f"X_{i}"]) for i in range(1, 11)]
        for row in data
    ], dtype=float)
    y = np.array([float(row["target"]) for row in data], dtype=float)

    # Fit the model
    results = model.fit(X, y)
    preds = results.predict(X)

    # ========== Plot 1: y vs predictions ==========
    
    plt.figure(figsize=(8, 5))
    plt.scatter(y, preds, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Fit')
    plt.xlabel('Actual Target y')
    plt.ylabel('Predicted y')
    plt.title('LASSO Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ========== Plot 2: Coefficient Magnitude ==========
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(results.coef_) + 1), results.coef_)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Learned LASSO Coefficients (Sparsity Check)')
    plt.grid(True)
    plt.show()

    # ========== Plot 3: Correlation Matrix Heatmap ==========
    corr = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Heatmap')
    plt.xticks(range(10), [f'X_{i}' for i in range(1, 11)])
    plt.yticks(range(10), [f'X_{i}' for i in range(1, 11)])
    plt.show()

    # ========== MSE and Coefficients Summary ==========
    residuals = y - preds
    mse = np.mean(residuals ** 2)
    nonzero_count = np.sum(results.coef_ != 0)
    zero_count = len(results.coef_) - nonzero_count

    print("Mean Squared Error (MSE):", mse)
    print(f"Non-zero coefficients: {nonzero_count}")
    print(f"Zero coefficients:     {zero_count}")

if __name__ == "__main__":
    test_collinear_data_with_plots()
