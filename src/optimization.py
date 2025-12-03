import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple
import scipy.optimize as sco


# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

@dataclass
class Config:
    ASSETS: List[str]
    START_DATE: str = "2021-01-01"
    END_DATE: str = "2024-01-01"
    FREQUENCY: str = "1d"

    TRADING_DAYS_YEAR: int = 365
    RISK_FREE_RATE: float = 0.0

    NUM_SIMULATIONS: int = 15000
    BOUNDS: Tuple[float, float] = (0.0, 1.0)   # weight bounds (no shorting)


default_config = Config(
    ASSETS=["ADA-USD", "AVAX-USD", "BNB-USD", "BTC-USD", "ETH-USD", "SOL-USD"]
)


# ============================================================
# PRICE DATA LOADING
# ============================================================

def load_price_data(config: Config = default_config) -> pd.DataFrame:
    """
    Download historical prices for the asset universe.
    """
    raw = yf.download(
        tickers=config.ASSETS,
        start=config.START_DATE,
        end=config.END_DATE,
        interval=config.FREQUENCY,
        auto_adjust=True
    )

    # yfinance may return OHLC with MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw

    data = data.dropna()
    return data


# ============================================================
# RETURNS & STATISTICS
# ============================================================

def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns r_t = ln(P_t / P_{t-1}).
    """
    return np.log(price_df / price_df.shift(1)).dropna()


def compute_statistics(returns_df: pd.DataFrame, config: Config = default_config):
    """
    Compute annualized mean returns and covariance matrix.
    """
    mean_returns = returns_df.mean() * config.TRADING_DAYS_YEAR
    covariance = returns_df.cov() * config.TRADING_DAYS_YEAR
    return mean_returns, covariance


# ============================================================
# MONTE CARLO PORTFOLIO GENERATION
# ============================================================

def generate_random_portfolios(mean_returns, covariance, config: Config = default_config):
    """
    Generate random portfolios to approximate the frontier.
    Returns:
        results: np.array shape (3, NUM_SIMULATIONS)
                 [0] = volatilities, [1] = returns, [2] = Sharpe ratios
        weight_records: list of weight vectors
    """
    n = len(mean_returns)
    results = np.zeros((3, config.NUM_SIMULATIONS))
    weight_records = []

    for i in range(config.NUM_SIMULATIONS):
        weights = np.random.random(n)
        weights /= np.sum(weights)

        portfolio_return = float(np.dot(weights, mean_returns))
        portfolio_volatility = float(np.sqrt(weights @ covariance @ weights))
        sharpe = (portfolio_return - config.RISK_FREE_RATE) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe

        weight_records.append(weights)

    return results, weight_records


# ============================================================
# MARKOWITZ OPTIMIZATION (EXACT FRONTIER)
# ============================================================

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0
):
    """
    Compute portfolio expected return, volatility and Sharpe ratio for given weights.
    """
    portfolio_return = float(np.dot(weights, mean_returns))
    portfolio_volatility = float(np.sqrt(weights @ covariance @ weights))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe


def _minimize_volatility(
    target_return: float,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    config: Config = default_config
):
    """
    Solve: minimize w'Σw  subject to:
        w'μ = target_return
        sum(w) = 1
        bounds: BOUNDS
    """
    n = len(mean_returns)
    args = (mean_returns, covariance, config.RISK_FREE_RATE)

    # initial guess: equal weights
    x0 = np.array(n * [1.0 / n])

    bounds = tuple((config.BOUNDS[0], config.BOUNDS[1]) for _ in range(n))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},                 # fully invested
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},  # target μ
    )

    def objective(w):
        _, vol, _ = portfolio_performance(w, *args)
        return vol

    result = sco.minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False}
    )
    return result


def efficient_frontier(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    config: Config = default_config,
    points: int = 50
):
    """
    Compute the exact efficient frontier by solving a sequence of
    minimum-variance problems for different target returns.

    Returns:
        frontier_vols: list of volatilities
        frontier_rets: list of expected returns
        frontier_weights: list of optimal weight vectors
    """
    min_return = float(mean_returns.min())
    max_return = float(mean_returns.max())
    target_returns = np.linspace(min_return, max_return, points)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        res = _minimize_volatility(target, mean_returns, covariance, config)
        if not res.success:
            continue

        w_opt = res.x
        ret, vol, _ = portfolio_performance(w_opt, mean_returns, covariance, config.RISK_FREE_RATE)

        frontier_vols.append(vol)
        frontier_rets.append(ret)
        frontier_weights.append(w_opt)

    return np.array(frontier_vols), np.array(frontier_rets), frontier_weights


def max_sharpe_ratio(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    config: Config = default_config
):
    """
    Solve for the portfolio that maximizes the Sharpe ratio:
        max_w  (w'μ - r_f) / sqrt(w'Σw)
        s.t.   sum(w) = 1,  bounds
    """
    n = len(mean_returns)
    args = (mean_returns, covariance, config.RISK_FREE_RATE)

    x0 = np.array(n * [1.0 / n])
    bounds = tuple((config.BOUNDS[0], config.BOUNDS[1]) for _ in range(n))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    def negative_sharpe(w):
        _, _, sharpe = portfolio_performance(w, *args)
        return -sharpe

    result = sco.minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False}
    )

    if not result.success:
        raise RuntimeError("Max Sharpe optimization failed:", result.message)

    w_opt = result.x
    ret, vol, sharpe = portfolio_performance(w_opt, mean_returns, covariance, config.RISK_FREE_RATE)
    return w_opt, ret, vol, sharpe
