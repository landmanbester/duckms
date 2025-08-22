# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuckMS is a Python tool for converting CASA Measurement Sets (radio astronomy data format) into a hybrid DuckDB/Parquet format optimized for modern data analysis workflows. The tool handles complex visibility data by splitting complex numbers into real/imaginary components and stores main data in partitioned Parquet files while subtables go into DuckDB.

## Core Architecture

The main architecture revolves around the `MSToDuckDBConverter` class in `duckms/ms_to_duckdb.py:27`:

- **Main Data Processing**: Converts the primary MS table to partitioned Parquet files (partitioned by `OBSERVATION_ID`)
- **Subtable Handling**: Converts 13 standard MS subtables (ANTENNA, FIELD, etc.) to DuckDB tables
- **Complex Data Conversion**: Splits complex numpy arrays into separate real/imaginary columns with shape metadata preservation
- **Unified View Creation**: Creates a comprehensive DuckDB view that joins main data with relevant subtables

### Key Processing Flow

1. `_convert_main_table()` - Processes main table in configurable chunks, handles complex data types
2. `_convert_subtables()` - Iterates through standard subtables, converts to DuckDB format
3. `_create_unified_view()` - Creates `main_view` joining main data with antenna, field, spectral_window, and observation tables

### Data Type Handling

- **Complex arrays**: Split into `_REAL` and `_IMAG` columns with `_SHAPE` metadata
- **Multi-dimensional arrays**: Flattened and stored as lists with shape information
- **Boolean arrays**: Preserved as boolean type
- **Simple types**: Passed through unchanged

## Common Development Tasks

### Build and Test Commands

```bash
# Install dependencies with uv (recommended)
uv sync --dev

# Linting and formatting
uv run ruff check .          # Check linting
uv run ruff format .         # Auto-format code
uv run ruff check --fix .    # Auto-fix linting issues

# Type checking
uv run mypy duckms/

# Build package
uv build

# Run basic installation test
uv pip install -e .
```

### Testing the Converter

```bash
# Command-line usage
python duckms/ms_to_duckdb.py /path/to/measurement_set.ms
python duckms/ms_to_duckdb.py /path/to/measurement_set.ms --chunk-size 50000 --verbose

# Or via entry point
ms-to-duckdb /path/to/measurement_set.ms
```

### Python API Usage

```python
from duckms import MSToDuckDBConverter

converter = MSToDuckDBConverter(chunk_size=100000)
converter.convert('/path/to/measurement_set.ms', '/path/to/output.duckdb')
```

## Code Configuration

- **Linting**: Uses Ruff with configuration in `pyproject.toml:63-103`
- **Type Checking**: MyPy configured with strict settings in `pyproject.toml:105-127`
- **Pre-commit**: Hooks configured for ruff, mypy, and standard checks
- **CI/CD**: Comprehensive GitHub Actions workflow testing across Python 3.8-3.12 and Ubuntu/macOS

## Dependencies

Core dependencies (defined in `pyproject.toml:29-36`):
- `duckdb>=0.9.0` - Database engine
- `python-casacore>=3.5.0` - CASA table reading (requires system libraries)
- `pyarrow>=14.0.0` - Parquet file handling
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data manipulation
- `fsspec>=2.1.0` - File system abstraction

## System Dependencies

The project requires system libraries for python-casacore:

**Ubuntu/Debian:**
```bash
sudo apt-get install casacore-dev libcfitsio-dev wcslib-dev libfftw3-dev libhdf5-dev libboost-dev
```

**macOS:**
```bash
brew install casacore cfitsio wcslib fftw hdf5 boost
```

## Output Structure

The converter creates a directory structure:
```
output.duckdb/
├── data.duckdb              # DuckDB database with subtables and unified view
└── main_data/               # Partitioned Parquet files
    ├── OBSERVATION_ID=0/
    └── OBSERVATION_ID=1/
```

Access via: `conn = duckdb.connect('output.duckdb/data.duckdb')`

## Error Handling Patterns

The codebase uses comprehensive error handling with:
- Graceful subtable skipping when missing
- Column-level error handling with warnings
- Chunk-based processing for memory efficiency
- Detailed logging throughout conversion process

## CLI Entry Point

The package provides a `ms-to-duckdb` command via `project.scripts` in pyproject.toml that maps to `duckms.ms_to_duckdb:main`.
