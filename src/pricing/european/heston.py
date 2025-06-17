import cmath
import math
import numpy as np
from src.processes.price_models import heston_process
from scipy.integrate import quad

class HestonModel:
    """
    Class for calculating the Heston model option pricing.
    """
    def __init__(self, S: float, K: float, T: float, r: float, v0: float, kappa: float, theta: float, sigma: float, rho: float, method: str = "analytical"):
        """
        Initialise the Heston model with parameters.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            v0 (float): Initial variance of the underlying asset.
            kappa (float): Rate of mean reversion of the variance.
            theta (float): Long-term variance level.
            sigma (float): Volatility of volatility.
            rho (float): Correlation between the asset and its volatility.
            method (str): Pricing method to use ('analytical', 'monte_carlo').
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.method = method
    

    def monte_carlo_price(self, option_type: str = "call", N: int = 2, M: int = 10000) -> float:
        """
        Calculate the option price using Monte Carlo simulation.

        Args:
            option_type (str): 'call' for call option, 'put' for put option.
            N (int): Number of Monte Carlo simulations.
            M (int): Number of paths to simulate.

        Returns:
            float: The calculated option price.
        """
        # Simulate M paths of the Heston process
        t, X, V = heston_process(
            T=self.T,
            N=N,
            mu=self.r,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            x0=self.S,
            v0=self.v0,
            M=M
        )
        
        # Calculate the terminal asset prices
        end_prices = X[:, -1].flatten()

        # Calculate the option payoffs
        if option_type == "call":
            payoffs = np.maximum(end_prices - self.K, 0)
        elif option_type == "put":
            payoffs = np.maximum(self.K - end_prices, 0)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        # Discount the expected payoff back to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)

        return price
    

    def price(self, option_type: str = "call", **kwargs) -> float:
        """
        Dispatch to chosen pricing method: 'analytical' or 'monte_carlo'.
        Additional kwargs passed to the respective method.
        """
        if self.method in ("monte_carlo", "mc"):
            return self.monte_carlo_price(option_type, **kwargs)
        else:
            raise ValueError(f"Unknown method '{self.method}'")