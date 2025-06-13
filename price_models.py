import numpy as np
from typing import List, Optional, Sequence, Tuple


def ou_process(
    T: float,
    N: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate an Ornstein-Uhlenbeck (OU) process over [0, T].

    Args:
        T (float): Total time horizon.
        N (int): Number of time steps (including t=0).
        theta (float): Speed of mean reversion.
        mu (float): Long-term mean level.
        sigma (float): Volatility coefficient.
        x0 (float): Initial value of the process at time 0.
        seed (Optional[int]): Seed for random number generation. Defaults to None.

    Returns:
        t (np.ndarray): Array of time points of length N.
        X (np.ndarray): Simulated OU process values of length N.
    """
    # Set up time grid
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    # Initialize process array
    X = np.empty(N)
    X[0] = x0

    # Random number generator
    rng = np.random.default_rng(seed)

    # Generate increments and simulate
    for i in range(1, N):
        dW = rng.standard_normal() * np.sqrt(dt)
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

    return t, X


def merton_jump_diffusion(
    T: float,
    N: int,
    mu: float,
    sigma: float,
    mu_j: float,
    sigma_j: float,
    lam: float,
    x0: float,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Merton jump-diffusion (jump-extended GBM) over [0, T].

    Args:
        T (float): Total simulation time.
        N (int): Number of time points (including t=0).
        mu (float): Drift of the diffusion component.
        sigma (float): Volatility of the diffusion component.
        mu_j (float): Mean of jump-size distribution (in log-space).
        sigma_j (float): Std. dev. of jump-size distribution (in log-space).
        lam (float): Jump intensity (expected jumps per unit time).
        x0 (float): Initial asset level.
        seed (Optional[int]): Seed for RNG. Defaults to None.

    Returns:
        t (np.ndarray): Time grid, shape (N,).
        X (np.ndarray): Simulated process values, shape (N,).
    """
    rng = np.random.default_rng(seed)
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    # Pre-allocate output array
    X = np.empty(N)
    X[0] = x0

    # Pre-compute compensator kappa = E[e^{J}] - 1
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1

    # Simulate increments
    for i in range(1, N):
        # Brownian increment
        dW = rng.standard_normal() * np.sqrt(dt)
        # Number of jumps in this interval
        n_jumps = rng.poisson(lam * dt)
        # Aggregate log-jump (0 if none)
        Q = rng.normal(n_jumps * mu_j, np.sqrt(n_jumps) * sigma_j) if n_jumps > 0 else 0.0

        # Euler update in log-space + jumps
        drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt
        X[i] = X[i-1] * np.exp(drift + sigma * dW + Q)

    return t, X


def geometric_brownian_motion(
    T: float,
    N: int,
    mu: float,
    sigma: float,
    x0: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Geometric Brownian Motion (GBM) over [0, T].

    dX_t = mu * X_t * dt + sigma * X_t * dW_t

    Args:
        T (float): Total time horizon.
        N (int): Number of time points (including t=0).
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        x0 (float): Initial value X(0).
        seed (Optional[int]): Seed for random number generator.

    Returns:
        t (np.ndarray): Array of time points, shape (N,).
        X (np.ndarray): Simulated GBM values, shape (N,).
    """
    rng = np.random.default_rng(seed)
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    X = np.empty(N)
    X[0] = x0

    for i in range(1, N):
        dW = rng.standard_normal() * np.sqrt(dt)
        X[i] = X[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return t, X