"""Residual adapters and language/corpus routing."""

from .residual import ResidualAdapter
from .router import AdapterRouter

__all__ = ["AdapterRouter", "ResidualAdapter"]
