
# Lasso Homotopy Regression Model
This project implements the LASSO Regression Model using a Homotopy-inspired method finalized with Coordinate Descent optimization from scratch in Python.
The model is capable of solving high-dimensional regression problems while promoting sparsity by penalizing less important features.

The implementation mimics scikit-learn’s API for ease of use (fit, predict, and access to coef_) and supports plotting for model interpretability.

## **Group Members**

| Name                   | A-Number     | Email                      |
|------------------------|--------------|----------------------------|
| Tarun Sai Tirumala     | A20544933    | ttirumala@hawk.iit.edu     |
| Lokesh manchikanti     | A20544931    | Lmanchikanti@hawk.iit.edu  |
| Jaswanth Reddy Sure    | A20552169    | jsure@hawk.iit.edu         |
| Vinay kumar moka       | A20552044    | Vmoka@hawk.iit.edu         |

---

## **Core Implementation Summary**

This project implements the **LASSO (Least Absolute Shrinkage and Selection Operator) Regression Model** using:
- **Homotopy-inspired initialization**
- **Coordinate Descent optimization**  
The model automatically selects features by forcing irrelevant coefficients to zero through L1 regularization.

**Key Features of the Implementation:**
- Data centering for numerical stability
- Manual gradient update of each coefficient
- Soft-thresholding for L1 penalty handling
- Stops iterating when updates are within a small tolerance (`tol`)
- Returns sparse coefficients `coef_` and supports `.predict()` API like sklearn.

---

## Installation & Environment
1. Python Version: 3.11
2. Setup Virtual Environment:
```bash
virtualenv venv
source venv/bin/activate
```
3. Install Requirements:
```bash
pip install -r requirements.txt
```
Required: `numpy`, `matplotlib`

---

## **Usage Example**
### Running Tests:
```bash
python tests/test_colliner_data.py
python tests/test_LassoHomotopy.py
python tests/test_high_diam_data.py
python tests/test_sparse_data.py
python tests/test_mul_col_data.py
```

Example Python snippet:
```python
model = LassoHomotopyModel(regularization_param=0.1)
results = model.fit(X, y)
predictions = results.predict(X)
```

---

## **Test Cases, Results, and Screenshots**

### **Test 1: Synthetic Sparse Test**

```
=== Synthetic Sparse Test ===
MSE: 0.1999
R² Score: 0.9872
Expected Non-Zeros: 3 | Model Non-Zeros: 20

```
 **Coefficient Bar Chart Screenshot:** `tests/screenshots/sparse_test.png`
![](/LassoHomotopy/tests/screenshots/sparse_test.png)
---

###  **Test 2: Multicollinear Test**
```
=== Multicollinear Test ===
MSE: 0.8570
R² Score: 0.9667
```
---

###  **Test 3: Small Sample High Dimensional Test**
```
=== Small Sample High Dim Test ===
MSE: 0.0002
R² Score: 1.0000
Non-zero Coefficients: 10

```
 **Coefficient Bar Chart Screenshot:** ![](/LassoHomotopy/tests/screenshots/hight.png)
---

###  **Test 4: Small data test**
```
Mean Squared Error (MSE): 1.2173393607079919
Number of nonzero coefficients: 3
Number of zero coefficients:    0

```
 **Predictions vs Actual:** ![](/LassoHomotopy/tests/screenshots/small_test.png)
---

###  **Test 5: Colliner data test**
```
Mean Squared Error (MSE): 4.055540049907282
Non-zero coefficients: 10
Zero coefficients:     0

```
 **LASSO Predictions vs Actual:** ![](/LassoHomotopy/tests/screenshots/coll_data.png)
**Feature Correlation Heatmap:** ![](/LassoHomotopy/tests/screenshots/coll_heat.png)


---

##  **Code Implementation Explanation**


The model implements **LASSO regression** using a **Coordinate Descent algorithm** inspired by the **Homotopy/LARS method**.

### **Key Steps:**
1. **Data Centering**: Removes mean from `X` and `y` for stability.
2. **Coordinate Descent**: Iteratively updates each coefficient:
   - Calculates the **partial residual**
   - Computes the correlation `rho`
   - Applies **soft-thresholding** to update the coefficient:
   ![](/LassoHomotopy/tests/screenshots/code.png)

3. **Convergence Check**: Stops when coefficient changes are below the defined tolerance.

---

### **Core Code Snippet: Coordinate Descent Update**
```python
for j in range(d):
    beta[j] = 0.0
    r_j = y - X @ beta
    rho = X[:, j].T @ r_j
    aj = (X[:, j] ** 2).sum()
    if aj < 1e-12:
        beta[j] = 0.0
    else:
        beta[j] = np.sign(rho) * max(abs(rho) - mu, 0.0) / aj
```

---


### **Hyperparameters Exposed:**
| Parameter                 | Description                                     |
|---------------------------|-------------------------------------------------|
| `regularization_param (λ)`| L1 penalty controlling sparsity                 |
| `max_iter`                | Maximum number of iterations                    |
| `tol`                     | Convergence threshold                          |


---

## 1. **What does the model you have implemented do and when should it be used?**

The model we implemented is a **LASSO (Least Absolute Shrinkage and Selection Operator) regression model** using a **homotopy-inspired approach combined with coordinate descent optimization**. 

It solves the following optimization problem:
![](/LassoHomotopy/tests/screenshots/code2.png)
The goal of the model is not only to minimize the prediction error but also to enforce **sparsity** in the learned coefficients by applying an **L1 penalty (λ)**. This sparsity forces the coefficients of irrelevant or weak features toward zero, effectively performing **automatic feature selection**.

---

### **When should it be used?**
- When you have **high-dimensional datasets** (more features than observations).
- If your data suffers from **multicollinearity** (highly correlated features).
- When you need a model that provides **interpretability** by selecting only the most significant features.
- In real-world problems like **genetics (gene selection)**, **finance**, **text data**, or **any domain with many predictors** but few of them expected to be truly important.

---

## 2. **How did you test your model to determine if it is working reasonably correctly?**

We tested the model extensively using **synthetic datasets** and **real-world inspired datasets** where the true relationship between features and target is known or controlled. The following test cases were created and executed:

1. **Synthetic Sparse Test:**
   - Only a few features were designed to impact the output.
   - Checked if the model recovers only those features (sparsity recovery).

2. **Multicollinear Test:**
   - Highly correlated features were generated.
   - Verified if the model selects one feature and suppresses others.

3. **Small Sample High Dimensional Test:**
   - Very few samples and many features.
   - Tested LASSO's capability to handle underdetermined systems.

4. **Noise Test (No Signal):**
   - Pure noise input without any real relation between `X` and `y`.
   - Checked if the model avoids overfitting and reports low R².

### **Validation Metrics Used:**
- **Mean Squared Error (MSE):** Measures prediction error.
- **R² Score:** Measures how well the model explains the variance.
- **Sparsity Check:** Counting non-zero coefficients.

### **Visual Validation:**
- Plotted:
  - Prediction vs Actual scatter plots
  - Coefficient bar plots to observe sparsity
  - Correlation heatmaps

All these tests confirmed that the model behaves as expected:
- Recovers correct features in sparse cases  
- Avoids overfitting noise  
- Handles multicollinearity well

---

## 3 **What parameters have you exposed to users of your implementation in order to tune performance?**


We exposed three main hyperparameters that allow users to control model performance, accuracy, and computational efficiency:

| Parameter                  | Description                                                                                                                                                                                                 |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `regularization_param (λ)` | **Controls the strength of the L1 penalty**. Higher λ forces more coefficients to zero, increasing sparsity. Lower λ allows more features but may reduce sparsity. Directly balances model complexity.         |
| `max_iter`                 | **Sets the maximum number of iterations** for the coordinate descent loop. Higher values improve precision but increase computation time. Useful for large or complex datasets.                              |
| `tol`                      | **Tolerance for convergence**. The algorithm stops when coefficient updates are smaller than this threshold. A smaller `tol` ensures better accuracy but increases runtime. Controls the precision of the model. |

These parameters allow users to:
-  Control the trade-off between accuracy and sparsity  
- Adjust computation time based on dataset size  
- Achieve desired model interpretability

---

## 4 **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**

Yes, like any LASSO implementation, our model can struggle with certain types of inputs or scenarios:

| Scenario                                | Challenge / Limitation                                                                                           | Potential Workaround / Solution                                        |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| **Perfect Multicollinearity**           | LASSO may randomly select one variable among highly correlated features and zero out others.                    | Use **Elastic Net** (L1 + L2 penalty) to balance feature selection.   |
| **Pure Noise Data**                     | The model might still pick up 1-2 random features due to random correlations in noise.                          | Increase λ, perform cross-validation, or run multiple random tests.    |
| **Extremely High-Dimensional Data (100k+ features)** | Coordinate descent becomes computationally slow due to sequential updates.                                       | Parallelize coordinate updates or switch to stochastic coordinate descent. |
| **Features with very small variance**   | Numerically unstable updates or unnecessary feature retention.                                                   | Feature scaling or variance thresholding before model fitting.         |

---

###  **Could it be improved?**
Yes. Given more time and resources, we could:
- Add **parallelization** to speed up computation for large datasets.
- Implement the full **LARS/Homotopy path-following algorithm**.
- Integrate **cross-validation** to automatically select the best `λ`.
- Extend it to **Elastic Net** to better handle correlated features.

