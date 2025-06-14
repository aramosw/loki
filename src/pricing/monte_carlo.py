import numpy as np
from src.processes.price_models import geometric_brownian_motion
from typing import Optional


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