import numpy as np
from typing import Optional, Tuple


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
        seed (Optional[int]): Seed for random number generation.

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
    M: int,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised simulation of M Merton jump-diffusion paths over [0, T].

    dX_t/X_t = mu * dt + sigma * dW_t + dJ_t

    Args:
        T (float): Total simulation time.
        N (int): Number of time points (including t=0).
        mu (float): Drift of diffusion component.
        sigma (float): Volatility of diffusion component.
        mu_j (float): Mean jump size (log-space).
        sigma_j (float): Jump size volatility (log-space).
        lam (float): Jump intensity (expected jumps per unit time).
        x0 (float): Initial asset level.
        M (int): Number of paths to simulate.
        seed (Optional[int]): Seed for random number generation.

    Returns:
        t (np.ndarray): Time grid of shape (N,).
        X (np.ndarray): Simulated paths of shape (M, N).
    """
    rng = np.random.default_rng(seed)
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    # Pre-compute compensator for drift adjustment
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift = (mu - 0.5 * sigma**2 - lam * kappa) * dt

    # Diffusion increments: shape (M, N-1)
    Z = rng.standard_normal((M, N - 1))
    diffusion = sigma * np.sqrt(dt) * Z

    # Jump increments
    N_j = rng.poisson(lam * dt, size=(M, N - 1))
    Q = rng.standard_normal((M, N - 1)) * (sigma_j * np.sqrt(N_j)) + mu_j * N_j

    # Combine log increments and integrate
    log_increments = drift + diffusion + Q
    log_paths = np.concatenate([np.zeros((M, 1)), log_increments], axis=1).cumsum(axis=1)
    X = x0 * np.exp(log_paths)

    return t, X


def geometric_brownian_motion(
    T: float,
    N: int,
    mu: float,
    sigma: float,
    x0: float,
    M: int,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Monte Carlo simulation of M Geometric Brownian Motion paths.

    Args:
        T (float): Total time horizon.
        N (int): Number of time points (including t=0).
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        x0 (float): Initial value X(0).
        M (int): Number of simulated paths.
        seed (Optional[int]): Seed for random number generation.

    Returns:
        t (np.ndarray): Time grid of shape (N,).
        X (np.ndarray): Simulated paths array of shape (M, N).
    """
    rng = np.random.default_rng(seed)
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    # Generate random normals: shape (M, N-1)
    Z = rng.standard_normal((M, N - 1))
    # Compute log increments matrix of same shape, prepend zeros column
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    # prepend zeros for t=0 and compute cumulative sum along axis=1
    log_increments = np.concatenate(
        (np.zeros((M, 1)), drift + diffusion), axis=1
    )
    log_paths = np.cumsum(log_increments, axis=1)
    # exponentiate and scale by x0
    X = x0 * np.exp(log_paths)

    return t, X


def heston_model(
    T: float,
    N: int,
    mu: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    x0: float,
    v0: float,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Heston model for stochastic volatility.

    Args:
        T (float): Total time horizon.
        N (int): Number of time points (including t=0).
        mu (float): Drift of the asset price.
        kappa (float): Speed of mean reversion for volatility.
        theta (float): Long-term mean of volatility.
        sigma (float): Volatility of volatility.
        rho (float): Correlation between asset and volatility processes.
        x0 (float): Initial asset price.
        v0 (float): Initial variance.
        seed (Optional[int]): Seed for random number generation.

    Returns:
        t (np.ndarray): Time grid, shape (N,).
        X (np.ndarray): Simulated asset prices, shape (N,).
    """
    rng = np.random.default_rng(seed)
    dt = T / (N - 1)
    t = np.linspace(0.0, T, N)

    X = np.empty(N)
    V = np.empty(N)
    X[0] = x0
    V[0] = v0

    for i in range(1, N):
        dW1 = rng.standard_normal() * np.sqrt(dt)  # Asset process
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.standard_normal() * np.sqrt(dt)  # Volatility process

        # Update variance using Euler-Maruyama
        V[i] = max(V[i-1] + kappa * (theta - V[i-1]) * dt + sigma * np.sqrt(V[i-1]) * dW2, 0)

        # Update asset price using the Heston model formula
        X[i] = X[i-1] * np.exp((mu - 0.5 * V[i-1]) * dt + np.sqrt(V[i-1]) * dW1)

    return t, X, V