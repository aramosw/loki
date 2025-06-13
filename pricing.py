import numpy as np
from price_models import geometric_brownian_motion
from scipy.stats import norm
from typing import Optional, Tuple


class BlackScholes:
    @staticmethod
    def calculate_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> float:
        """
        Calculate the Black-Scholes option price.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            sigma (float): Volatility of the underlying asset (annualised).
            option_type (str): 'call' for call option, 'put' for put option.

        Returns:
            float: The calculated option price.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == "put":
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        return price
    

class BinomialTree:
    """
    Binomial Tree model for option pricing.
    """
    def __init__(self, S: float, K: float, T: float, r: float, steps: int, option_type: str = "call", american: bool = False):
        """
        Initialise the Binomial Tree model.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            steps (int): Number of time steps in the binomial tree.
            option_type (str): 'call' for call option, 'put' for put option.
            american (bool): True for American option, False for European option.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.steps = steps
        self.option_type = option_type
        self.american = american
        self.dt = T / steps
        self.discount = np.exp(-r * self.dt)  
    

    def calculate_price(self, u: float, d: float, q: float) -> float:
        """
        Calculate the option price using the Binomial Tree method.

        Args:
            u (float): Up factor per step.
            d (float): Down factor per step.
            q (float): Risk-neutral probability of an up move.

        Returns:
            float: The calculated option price.
        """
        # Initialise asset prices at maturity
        asset_prices = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            asset_prices[i] = self.S * (u ** (self.steps - i)) * (d ** i)

        # Initialise option values at maturity
        option_values = np.zeros(self.steps + 1)
        if self.option_type == "call":
            option_values = np.maximum(0, asset_prices - self.K)
        elif self.option_type == "put":
            option_values = np.maximum(0, self.K - asset_prices)

        # Backward induction for option pricing
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[i] = (q * option_values[i] + (1 - q) * option_values[i + 1]) * self.discount

                # Check for early exercise if American option
                if self.american:
                    if self.option_type == "call":
                        option_values[i] = max(option_values[i], asset_prices[i] - self.K)
                    elif self.option_type == "put":
                        option_values[i] = max(option_values[i], self.K - asset_prices[i])

        return option_values[0]


class MonteCarlo:
    """
    Monte Carlo simulation for option pricing.
    """
    @staticmethod
    def calculate_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        num_simulations: int = 10000,
        option_type: str = "call",
        process: Optional[str] = "gbm"
    ) -> float:
        """
        Calculate the option price using Monte Carlo simulation.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            sigma (float): Volatility of the underlying asset (annualised).
            num_simulations (int): Number of Monte Carlo simulations.
            option_type (str): 'call' for call option, 'put' for put option.

        Returns:
            float: The calculated option price.
        """
        dt = T

        if process != "gbm":
            raise ValueError("Currently, only 'gbm' process is supported for Monte Carlo simulation.")
        
        # Simulate end prices using GBM
        end_prices = geometric_brownian_motion(T=dt, N=num_simulations, mu=r, sigma=sigma, x0=S)[1][-1]

        if option_type == "call":
            payoffs = np.maximum(0, end_prices - K)
        elif option_type == "put":
            payoffs = np.maximum(0, K - end_prices)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        # Discount back to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price