import numpy as np

class LassoHomotopyResults:
    """
    Container for fitted LASSO solution.
    """
    def __init__(self, coef, X_mean=None, y_mean=None):
        self.coef_ = coef
        self.X_mean_ = X_mean
        self.y_mean_ = y_mean

    def predict(self, X):
        """
        Predict using the fitted coefficients, adding intercept correction.
        """
        if self.X_mean_ is None or self.y_mean_ is None:
            return X @ self.coef_
        Xc = X - self.X_mean_
        yhat_centered = Xc @ self.coef_
        return yhat_centered + self.y_mean_


class LassoHomotopyModel:
    """
    LASSO regression using a Homotopy / LARS-Lasso inspired method,
    finalized with Coordinate Descent.

    Parameters
    ----------
    regularization_param : float
        The L1 penalty parameter (lambda). Higher means more sparsity.
    max_iter : int
        Maximum iterations for coordinate descent.
    tol : float
        Tolerance for convergence.
    """
    def __init__(self, regularization_param=1.0, max_iter=500, tol=1e-6):
        self.regularization_param = regularization_param
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y):
        y = y.ravel()
        n, d = X.shape

        # Centering data (optional but good practice for LASSO)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean

        # Compute initial max lambda
        correlation = Xc.T @ yc
        mu_max = np.max(np.abs(correlation))

        if self.regularization_param >= mu_max:
            # Fully regularized => zero vector
            self.coef_ = np.zeros(d)
            return LassoHomotopyResults(self.coef_, X_mean, y_mean)

        # === Fallback to full Coordinate Descent for simplicity ===
        # (You can replace this with full homotopy path implementation later)
        beta = self._coordinate_descent(Xc, yc, 
                                        self.regularization_param,
                                        max_iter=self.max_iter, 
                                        tol=self.tol)

        self.coef_ = beta
        return LassoHomotopyResults(beta, X_mean, y_mean)

    def _coordinate_descent(self, X, y, mu, max_iter=100, tol=1e-6):
        """
        Simple full coordinate descent solver for LASSO:
            min (1/2)||y - Xb||^2 + mu * ||b||_1
        """
        n, d = X.shape
        beta = np.zeros(d)

        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(d):
                tmp = beta[j]
                beta[j] = 0.0
                r_j = y - X @ beta
                rho = X[:, j].T @ r_j
                aj = (X[:, j] ** 2).sum()
                if aj < 1e-12:
                    beta[j] = 0.0
                else:
                    beta[j] = np.sign(rho) * max(abs(rho) - mu, 0.0) / aj
            # Convergence check
            if np.linalg.norm(beta - beta_old) < tol:
                break
        return beta
