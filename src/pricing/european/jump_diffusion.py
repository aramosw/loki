import numpy as np
from src.processes.price_models import merton_jump_diffusion


class JumpDiffusion:
    """
    Class for calculating the Merton jump diffusion option pricing model.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, lambd: float, mu_j: float, sigma_j: float, method: str = "analytical"):
        """
        Initialise the Merton jump diffusion model with parameters.

        Args:
            S (float): Current stock price.
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualised).
            sigma (float): Volatility of the underlying asset (annualised).
            lambd (float): Jump intensity (average number of jumps per year).
            mu_j (float): Mean of the jump size distribution.
            sigma_j (float): Standard deviation of the jump size distribution.
            method (str): Pricing method to use ('analytical', 'monte_carlo').
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lambd = lambd
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.method = method
    

    def analytical_price(self, option_type: str = "call", k_max: int = None) -> float:
        """
        Calculate the Merton jump diffusion option price using the analytical (series) formula.

        Args:
            option_type (str): 'call' or 'put'.
            k_max (int): Maximum number of jump terms to sum (optional).
                        If None, uses lambda*T + 10*sqrt(lambda*T).

        Returns:
            float: The calculated option price.
        """
        lambda_T = self.lambd * self.T
        # determine truncation level if not provided
        if k_max is None:
            k_max = int(math.ceil(lambda_T + 10 * math.sqrt(lambda_T)))

        price = 0.0
        for k in range(k_max + 1):
            # Poisson probability of k jumps
            log_p_k = -lambda_T + k * math.log(lambda_T) - math.lgamma(k + 1)
            p_k = math.exp(log_p_k)

            # Adjusted vol for k jumps
            sigma_k = math.sqrt(self.sigma**2 + (k * self.sigma_j**2) / self.T)
            # d1 and d2 for Black-Scholes with adjusted initial asset multiplier
            d1 = (
                math.log(self.S / self.K)
                + k * self.mu_j
                + (self.r + 0.5 * sigma_k**2) * self.T
            ) / (sigma_k * math.sqrt(self.T))
            d2 = d1 - sigma_k * math.sqrt(self.T)

            # Black-Scholes formula for this term
            if option_type == "call":
                bs_term = (
                    self.S * math.exp(k * self.mu_j) * norm.cdf(d1)
                    - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
                )
            elif option_type == "put":
                bs_term = (
                    self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)
                    - self.S * math.exp(k * self.mu_j) * norm.cdf(-d1)
                )
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            price += p_k * bs_term

        return price

    
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
        # Simulate M paths of the Merton jump diffusion process
        t, X = merton_jump_diffusion(
            T=self.T,
            N=N,
            mu=self.r - 0.5 * self.sigma ** 2,
            sigma=self.sigma,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
            lam=self.lambd,
            x0=self.S,
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
        if self.method == "analytical":
            return self.analytical_price(option_type, **kwargs)
        elif self.method in ("monte_carlo", "mc"):
            return self.monte_carlo_price(option_type, **kwargs)
        else:
            raise ValueError(f"Unknown method '{self.method}'")