import numpy as np


class BinomialTree:
    """
    Binomial Tree model for European option pricing.
    """
    def __init__(self, S: float, K: float, T: float, r: float, steps: int, option_type: str = "call"):
        """
        Initialise the Binomial Tree model.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            steps (int): Number of time steps in the binomial tree.
            option_type (str): 'call' for call option, 'put' for put option.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.steps = steps
        self.option_type = option_type
        self.dt = T / steps
        self.discount = np.exp(-r * self.dt)
    
    
    def price(self, u: float, d: float, q: float) -> float:
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

        return option_values[0]