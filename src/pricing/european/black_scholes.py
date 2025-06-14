import numpy as np
from src.processes.price_models import geometric_brownian_motion
from scipy.stats import norm


class BlackScholes:
    """
    Class for calculating the Black-Scholes option pricing model.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, method: str = "analytical"):
        """
        Initialise the Black-Scholes model with parameters.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            sigma (float): Volatility of the underlying asset (annualised).
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.method = method

    
    def analytical_price(self, option_type: str = "call") -> float:
        """
        Calculate the Black-Scholes option price using the analytical formula.

        Args:
            option_type (str): 'call' for call option, 'put' for put option.

        Returns:
            float: The calculated option price.
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if option_type == "call":
            price = (self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        elif option_type == "put":
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        return price


    def monte_carlo_price(self, option_type: str = "call", N: int = 10000) -> float:
        """
        Calculate the option price using Monte Carlo simulation.

        Args:
            option_type (str): 'call' for call option, 'put' for put option.
            N (int): Number of Monte Carlo simulations.

        Returns:
            float: The calculated option price.
        """
        end_prices = geometric_brownian_motion(T=self.T, N=N, mu=self.r, sigma=self.sigma, x0=self.S)[1][-1]

        if option_type == "call":
            payoffs = np.maximum(0, end_prices - self.K)
        elif option_type == "put":
            payoffs = np.maximum(0, self.K - end_prices)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        # Discount back to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)

        return price
    

    def finite_difference_price(
        self,
        option_type: str = "call",
        M: int = 100,
        N: int = 100,
        S_max_factor: float = 2.0
    ) -> float:
        """
        Price option via Crank-Nicolson finite difference on Black-Scholes PDE.

        Args:
            option_type: 'call' or 'put'
            M: number of asset price steps
            N: number of time steps
            S_max_factor: multiple of K defining max asset price
        Returns:
            option price at S=self.S, t=0
        """
        # Grid setup
        S_max = S_max_factor * self.K
        dS = S_max / M
        dt = self.T / N
        grid = np.zeros((M+1, N+1))
        S_vals = np.linspace(0, S_max, M+1)

        # Terminal payoff
        if option_type == "call":
            grid[:, -1] = np.maximum(S_vals - self.K, 0)
        else:
            grid[:, -1] = np.maximum(self.K - S_vals, 0)

        # Boundary conditions
        if option_type == "call":
            grid[0, :] = 0
            grid[-1, :] = S_max - self.K * np.exp(-self.r * (self.T - dt * np.arange(N+1)))
        else:
            grid[0, :] = self.K * np.exp(-self.r * (self.T - dt * np.arange(N+1)))
            grid[-1, :] = 0

        # Coefficients
        alpha = 0.25 * dt * (self.sigma**2 * (np.arange(M+1)**2) - self.r * np.arange(M+1))
        beta  = -dt * 0.5 * (self.sigma**2 * (np.arange(M+1)**2) + self.r)
        gamma = 0.25 * dt * (self.sigma**2 * (np.arange(M+1)**2) + self.r * np.arange(M+1))

        # Prepare tridiagonal matrices A and B
        A = np.zeros((M-1, M-1))
        B = np.zeros((M-1, M-1))
        for i in range(1, M):
            if i-2 >= 0:
                A[i-1, i-2] = -alpha[i]
                B[i-1, i-2] = alpha[i]
            A[i-1, i-1] = 1 - beta[i]
            B[i-1, i-1] = 1 + beta[i]
            if i < M-1:
                A[i-1, i] = -gamma[i]
                B[i-1, i] = gamma[i]

        # Time-stepping backwards
        P, Q = A, B
        for j in reversed(range(N)):
            # Right-hand side
            rhs = Q.dot(grid[1:-1, j+1])
            # Add boundary effects
            rhs[0]  += alpha[1]  * grid[0, j]    + alpha[1]  * grid[0, j+1]
            rhs[-1] += gamma[M-1] * grid[-1, j]   + gamma[M-1] * grid[-1, j+1]
            # Solve P x = rhs via Thomas algorithm
            x = self._thomas(P, rhs)
            grid[1:-1, j] = x

        # Interpolate to find price at S
        price = np.interp(self.S, S_vals, grid[:, 0])
        return price

    @staticmethod
    def _thomas(A: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system A x = d using Thomas algorithm.
        A assumed to be tridiagonal stored as full matrix.
        """
        n = len(d)
        # Extract diagonals
        a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
        for i in range(n):
            b[i] = A[i, i]
            if i > 0:
                a[i] = A[i, i-1]
            if i < n-1:
                c[i] = A[i, i+1]
        # Forward sweep
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom if i < n-1 else 0
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in reversed(range(n-1)):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        return x
    

    def price(
        self,
        option_type: str = "call",
        mc_N: int = 100000,
        fd_M: int = 100,
        fd_N: int = 100,
        S_max_factor: float = 2.0
    ) -> float:
        """
        Dispatch to the selected pricing method.

        Args:
            option_type: 'call' or 'put'
            mc_N: simulations for Monte Carlo
            fd_M: asset price steps for finite difference
            fd_N: time steps for finite difference
            S_max_factor: max asset price factor for grid

        Returns:
            float: option price
        """
        if self.method == "analytical":
            return self.analytical_price(option_type)
        elif self.method in ("monte_carlo", "mc"):
            return self.monte_carlo_price(option_type, N=mc_N)
        elif self.method in ("finite_difference", "fd"):
            return self.finite_difference_price(option_type, M=fd_M, N=fd_N, S_max_factor=S_max_factor)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'analytical', 'monte_carlo', or 'finite_difference'.")