from typing import Callable
import numpy as np
import scipy.optimize


def _minimize_objective(objective: Callable, n_securities: int,
                        non_negative_weights: bool = True) -> np.array:
    """Find portfolio weights that sum to 1, and that minimize some objective function 'fun'

    Args:
        fun (Callable): The objective function
        n_securities (int): The number of securities to choose from
        non_negative_weights (bool): If true (default), do not allow negative weights.

    Returns:
        np.array: Non-negative portfolio weights that sum to 1
    """
    x0 = np.random.random(size=n_securities)
    x0 /= x0
    return scipy.optimize.minimize(
        fun = objective,
        bounds = [[0. if non_negative_weights else -1., 1.] for _ in range(n_securities)],
        constraints = [{'type': 'eq', 'fun': lambda w: 1-sum(w)}],
        x0=x0
    ).x


def min_risk(expected_returns: np.array, covariance: np.array) -> np.array:
    """Returns the portfolio that minimizes risk

    Args:
        expected_returns (np.array): The expected returns of the securities
        covariance (np.array): The covariance matrix for the securities

    Returns:
        np.array: The optimal portfolio weight
    """
    objective = lambda x: np.matmul(np.matmul(x.T, covariance), x)
    return _minimize_objective(objective, covariance.shape[0])


def max_sharpe(expected_returns: np.array, covariance: np.array) -> np.array:
    """Returns the portfolio that maximizes the sharpe ratio

    Args:
        expected_returns (np.array): The expected returns of the securities
        covariance (np.array): The covariance matrix for the securities

    Returns:
        np.array: The optimal portfolio weight
    """
    objective = lambda x: -np.dot(x, expected_returns) / np.matmul(np.matmul(x.T, covariance), x)
    return _minimize_objective(objective, covariance.shape[0])
