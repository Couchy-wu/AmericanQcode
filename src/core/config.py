"""Configuration loader using OmegaConf with YAML + environment variable interpolation."""

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_env() -> None:
    """Load .env file if present."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


@lru_cache(maxsize=1)
def load_config() -> DictConfig:
    """Load and merge all YAML configs, resolving env variable interpolations."""
    _load_env()
    cfg = OmegaConf.create()
    for yaml_file in sorted(CONFIG_DIR.glob("*.yaml")):
        loaded = OmegaConf.load(yaml_file)
        cfg = OmegaConf.merge(cfg, loaded)
    return cfg


def get_provider_config(cfg: DictConfig | None = None) -> dict:
    """Extract provider configuration."""
    if cfg is None:
        cfg = load_config()
    return {
        "primary": cfg.providers.primary,
        "fallback": cfg.providers.fallback,
        "polygon_api_key": os.getenv("POLYGON_API_KEY", cfg.providers.get("polygon_api_key", "")),
        "alpha_vantage_key": os.getenv("ALPHA_VANTAGE_KEY", cfg.providers.get("alpha_vantage_key", "")),
    }


def get_scanner_config(cfg: DictConfig | None = None) -> dict:
    """Extract scanner configuration."""
    if cfg is None:
        cfg = load_config()
    return OmegaConf.to_container(cfg.scanner, resolve=True)


def get_backtest_config(cfg: DictConfig | None = None) -> dict:
    """Extract backtest configuration."""
    if cfg is None:
        cfg = load_config()
    return OmegaConf.to_container(cfg.backtest, resolve=True)


def get_web_config(cfg: DictConfig | None = None) -> dict:
    """Extract web server configuration."""
    if cfg is None:
        cfg = load_config()
    return OmegaConf.to_container(cfg.web, resolve=True)


def get_database_url(cfg: DictConfig | None = None) -> str:
    """Get the SQLAlchemy database URL."""
    if cfg is None:
        cfg = load_config()
    return cfg.database.url


def load_watchlists(cfg: DictConfig | None = None) -> dict[str, list[str]]:
    """Load all watchlists from config, returning {name: [tickers]}."""
    if cfg is None:
        cfg = load_config()
    watchlists = {}
    if hasattr(cfg, "watchlists"):
        wl_cfg = cfg.watchlists
    else:
        # Load from separate file
        wl_path = CONFIG_DIR / "watchlists.yaml"
        if wl_path.exists():
            wl_cfg = OmegaConf.load(wl_path)
        else:
            return watchlists

    for name in wl_cfg:
        watchlists[name] = list(wl_cfg[name])
    return watchlists


def get_strategy_configs(cfg: DictConfig | None = None) -> dict:
    """Get enabled strategy configurations."""
    if cfg is None:
        cfg = load_config()
    strategies = {}
    if hasattr(cfg, "strategies") and hasattr(cfg.strategies, "strategies"):
        for name, sc in cfg.strategies.strategies.items():
            if sc.get("enabled", True):
                strategies[name] = OmegaConf.to_container(sc, resolve=True)
    return strategies
