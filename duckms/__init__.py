"""
DuckMS: Convert CASA Measurement Sets to hybrid DuckDB/Parquet format.

A robust Python tool for converting CASA Measurement Sets into a hybrid
DuckDB/Parquet format optimized for modern data analysis workflows.
"""

__version__ = "0.1.0"
__author__ = "Claude"
__email__ = "noreply@anthropic.com"

from .ms_to_duckdb import MSToDuckDBConverter

__all__ = ["MSToDuckDBConverter"]
