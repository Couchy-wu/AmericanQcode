"""Custom exceptions for the quant trading system."""


class QuantError(Exception):
    """Base exception for the application."""
    pass


class DataProviderError(QuantError):
    """Error fetching data from a provider."""
    pass


class RateLimitError(DataProviderError):
    """API rate limit exceeded."""

    def __init__(self, provider: str, retry_after: float = 60.0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {provider}. Retry after {retry_after:.0f}s")


class DataGapError(DataProviderError):
    """Missing data in the expected range."""
    pass


class InvalidTickerError(DataProviderError):
    """Ticker symbol not found by provider."""
    pass


class IndicatorError(QuantError):
    """Error computing a technical indicator."""
    pass


class StrategyError(QuantError):
    """Error in strategy analysis."""
    pass


class BacktestError(QuantError):
    """Error during backtesting."""
    pass


class ConfigurationError(QuantError):
    """Invalid configuration."""
    pass


class MarketClosedError(QuantError):
    """Market is currently closed."""
    pass
