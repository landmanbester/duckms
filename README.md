# CASA Measurement Set to DuckDB/Parquet Converter

A robust Python tool for converting CASA Measurement Sets into a hybrid DuckDB/Parquet format optimized for modern data analysis workflows.

## Features

- **Efficient Storage**: Converts main data table to partitioned Parquet files for optimal query performance
- **Complete Metadata**: Preserves all subtables in DuckDB format
- **Complex Data Handling**: Properly handles complex visibility data by splitting into real/imaginary components
- **Memory Efficient**: Processes large datasets in configurable chunks
- **Unified Access**: Creates a comprehensive view joining all data together

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install the package and all dependencies
uv pip install .

# Or install directly from PyPI (when published)
uv pip install duckms
```

### Using pip

```bash
pip install .

# Or from PyPI (when published)
pip install duckms
```

## Usage

### Command Line

```bash
# Basic usage
python ms_to_duckdb.py /path/to/measurement_set.ms

# Specify output location
python ms_to_duckdb.py /path/to/measurement_set.ms /path/to/output.duckdb

# Process in smaller chunks for memory efficiency
python ms_to_duckdb.py /path/to/measurement_set.ms --chunk-size 50000

# Enable verbose logging
python ms_to_duckdb.py /path/to/measurement_set.ms --verbose
```

### Python API

```python
from ms_to_duckdb import MSToDuckDBConverter

converter = MSToDuckDBConverter(chunk_size=100000)
converter.convert('/path/to/measurement_set.ms', '/path/to/output.duckdb')
```

## Output Structure

The converter creates a directory containing:

```
output.duckdb/
├── data.duckdb              # DuckDB database with subtables and unified view
└── main_data/               # Partitioned Parquet files
    ├── OBSERVATION_ID=0/
    │   └── *.parquet
    └── OBSERVATION_ID=1/
        └── *.parquet
```

## Data Processing Details

### Complex Data Types

Complex visibility data (DATA, MODEL_DATA, CORRECTED_DATA) is split into separate real and imaginary columns:
- `DATA_REAL`, `DATA_IMAG`
- `MODEL_DATA_REAL`, `MODEL_DATA_IMAG`
- etc.

Multi-dimensional arrays preserve their structure using shape metadata.

### Partitioning Strategy

The main data table is partitioned by `OBSERVATION_ID` to optimize queries that filter by observation, which is common in radio astronomy workflows.

### Subtables

All standard MS subtables are converted to DuckDB tables:
- ANTENNA, FIELD, OBSERVATION
- SPECTRAL_WINDOW, POLARIZATION
- SOURCE, POINTING, etc.

## Querying the Converted Data

```python
import duckdb

# Connect to the converted database
conn = duckdb.connect('/path/to/output.duckdb/data.duckdb')

# Query the unified view
result = conn.execute("""
    SELECT
        TIME,
        ANTENNA1_NAME,
        ANTENNA2_NAME,
        FIELD_NAME,
        DATA_REAL[1] as first_channel_real
    FROM main_view
    WHERE OBSERVATION_ID = 0
    LIMIT 10
""").fetchdf()

print(result)
```

## Requirements

- Python 3.8+
- DuckDB 0.9.0+
- python-casacore 3.5.0+
- PyArrow 14.0.0+
- NumPy 1.24.0+
- Pandas 2.0.0+

## Error Handling

The converter includes comprehensive error handling:
- Graceful handling of missing subtables
- Robust complex data type conversion
- Memory-efficient processing for large datasets
- Detailed logging for troubleshooting

## Performance Tips

1. **Chunk Size**: Adjust `--chunk-size` based on available memory
2. **Storage**: Use fast storage (SSD) for both input and output
3. **Partitioning**: The OBSERVATION_ID partitioning optimizes many common queries
4. **Compression**: Snappy compression provides good balance of speed and size
