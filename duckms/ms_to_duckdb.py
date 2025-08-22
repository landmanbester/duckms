#!/usr/bin/env python3
"""
CASA Measurement Set to DuckDB/Parquet Converter

This module converts CASA Measurement Sets into a hybrid DuckDB/Parquet format:
- Main data table -> Partitioned Parquet files (partitioned by OBSERVATION_ID)
- Subtables -> DuckDB tables
- Unified view joining all data together

Author: Claude
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from casacore.tables import table


class MSToDuckDBConverter:
    """
    Converts CASA Measurement Sets to hybrid DuckDB/Parquet format.
    
    The converter creates:
    1. Partitioned Parquet files for the main data table
    2. DuckDB tables for all subtables
    3. A unified view that joins everything together
    """
    
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the converter.
        
        Args:
            chunk_size: Number of rows to process at a time for large tables
        """
        self.chunk_size = chunk_size
        self.logger = self._setup_logging()
        
        # Standard MS subtables to convert
        self.subtables = [
            'ANTENNA', 'FIELD', 'OBSERVATION', 'SPECTRAL_WINDOW', 
            'POLARIZATION', 'FEED', 'SOURCE', 'POINTING', 'DATA_DESCRIPTION',
            'HISTORY', 'DOPPLER', 'FREQ_OFFSET', 'SYSCAL', 'WEATHER'
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def convert(self, ms_path: str, output_path: str) -> None:
        """
        Convert a Measurement Set to DuckDB/Parquet format.
        
        Args:
            ms_path: Path to the source Measurement Set directory
            output_path: Path for the output directory (will be created)
        """
        ms_path = Path(ms_path)
        output_path = Path(output_path)
        
        if not ms_path.exists():
            raise FileNotFoundError(f"Measurement Set not found: {ms_path}")
        
        self.logger.info(f"Converting {ms_path} to {output_path}")
        
        try:
            # Create output directory structure
            self._create_output_structure(output_path)
            
            # Convert main table to Parquet
            self._convert_main_table(ms_path, output_path)
            
            # Convert subtables to DuckDB
            self._convert_subtables(ms_path, output_path)
            
            # Create unified view
            self._create_unified_view(output_path)
            
            self.logger.info("Conversion completed successfully")
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise
    
    def _create_output_structure(self, output_path: Path) -> None:
        """Create the output directory structure."""
        output_path.mkdir(parents=True, exist_ok=True)
        main_data_dir = output_path / "main_data"
        main_data_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created output structure at {output_path}")
    
    def _convert_main_table(self, ms_path: Path, output_path: Path) -> None:
        """
        Convert the main MS table to partitioned Parquet files.
        
        Args:
            ms_path: Path to the MS
            output_path: Output directory path
        """
        self.logger.info("Converting main table to Parquet...")
        
        try:
            with table(str(ms_path), readonly=True) as ms_table:
                total_rows = len(ms_table)
                self.logger.info(f"Main table has {total_rows} rows")
                
                # Get column information
                colnames = ms_table.colnames()
                self.logger.info(f"Columns: {colnames}")
                
                # Process in chunks to handle large datasets
                main_data_dir = output_path / "main_data"
                
                for start_row in range(0, total_rows, self.chunk_size):
                    end_row = min(start_row + self.chunk_size, total_rows)
                    self.logger.info(f"Processing rows {start_row} to {end_row}")
                    
                    # Read chunk
                    chunk_data = self._read_main_table_chunk(
                        ms_table, start_row, end_row - start_row, colnames
                    )
                    
                    # Write to Parquet partitioned by OBSERVATION_ID
                    self._write_parquet_chunk(chunk_data, main_data_dir)
                
        except Exception as e:
            self.logger.error(f"Failed to convert main table: {e}")
            raise
    
    def _read_main_table_chunk(self, ms_table, start_row: int, nrows: int, colnames: List[str]) -> pd.DataFrame:
        """
        Read a chunk of the main table and handle complex data types.
        
        Args:
            ms_table: Opened casacore table
            start_row: Starting row index
            nrows: Number of rows to read
            colnames: List of column names
        
        Returns:
            DataFrame with processed data
        """
        data = {}
        
        for col in colnames:
            try:
                col_data = ms_table.getcol(col, startrow=start_row, nrow=nrows)
                
                # Handle complex data types by splitting into real/imaginary parts
                if col_data.dtype in [np.complex64, np.complex128]:
                    self.logger.debug(f"Splitting complex column {col}")
                    
                    if col_data.ndim == 1:
                        # Simple 1D complex array
                        data[f"{col}_REAL"] = col_data.real.astype(np.float32)
                        data[f"{col}_IMAG"] = col_data.imag.astype(np.float32)
                    else:
                        # Multi-dimensional complex array (e.g., DATA with shape [nrow, nchan, npol])
                        # Flatten the array but preserve the structure information
                        shape = col_data.shape
                        flattened_real = col_data.real.flatten().astype(np.float32)
                        flattened_imag = col_data.imag.flatten().astype(np.float32)
                        
                        # Store as lists to preserve structure in Parquet
                        chunk_size = shape[1] * shape[2] if len(shape) == 3 else shape[1]
                        data[f"{col}_REAL"] = [flattened_real[i:i+chunk_size].tolist() 
                                             for i in range(0, len(flattened_real), chunk_size)]
                        data[f"{col}_IMAG"] = [flattened_imag[i:i+chunk_size].tolist() 
                                             for i in range(0, len(flattened_imag), chunk_size)]
                        data[f"{col}_SHAPE"] = [list(shape[1:])] * shape[0]
                
                elif col_data.dtype == bool:
                    # Handle boolean arrays
                    data[col] = col_data.astype(bool)
                
                elif col_data.ndim > 1:
                    # Handle multi-dimensional non-complex arrays
                    shape = col_data.shape
                    if len(shape) == 2:
                        # Convert to list of lists for Parquet
                        data[col] = [row.tolist() for row in col_data]
                    else:
                        # Flatten higher dimensions
                        flattened = col_data.flatten()
                        chunk_size = np.prod(shape[1:])
                        data[col] = [flattened[i:i+chunk_size].tolist() 
                                   for i in range(0, len(flattened), chunk_size)]
                    data[f"{col}_SHAPE"] = [list(shape[1:])] * shape[0]
                
                else:
                    # Handle simple data types
                    data[col] = col_data
                    
            except Exception as e:
                self.logger.warning(f"Skipping column {col} due to error: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def _write_parquet_chunk(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Write DataFrame chunk to Parquet, partitioned by OBSERVATION_ID.
        
        Args:
            df: DataFrame to write
            output_dir: Output directory for Parquet files
        """
        if 'OBSERVATION_ID' not in df.columns:
            self.logger.warning("No OBSERVATION_ID column found, using default partition")
            df['OBSERVATION_ID'] = 0
        
        # Convert to PyArrow table for efficient writing
        table = pa.Table.from_pandas(df)
        
        # Write partitioned by OBSERVATION_ID
        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=['OBSERVATION_ID'],
            compression='snappy',
            use_legacy_dataset=False
        )
    
    def _convert_subtables(self, ms_path: Path, output_path: Path) -> None:
        """
        Convert all subtables to DuckDB tables.
        
        Args:
            ms_path: Path to the MS
            output_path: Output directory path
        """
        self.logger.info("Converting subtables to DuckDB...")
        
        db_path = output_path / "data.duckdb"
        
        try:
            with duckdb.connect(str(db_path)) as conn:
                for subtable_name in self.subtables:
                    subtable_path = ms_path / subtable_name
                    
                    if not subtable_path.exists():
                        self.logger.debug(f"Subtable {subtable_name} not found, skipping")
                        continue
                    
                    self.logger.info(f"Converting subtable: {subtable_name}")
                    
                    try:
                        # Read subtable using casacore
                        with table(str(subtable_path), readonly=True) as sub_table:
                            df = self._read_subtable_to_dataframe(sub_table)
                            
                            # Register as DuckDB table
                            conn.register(f"temp_{subtable_name.lower()}", df)
                            conn.execute(f"""
                                CREATE TABLE {subtable_name.lower()} AS 
                                SELECT * FROM temp_{subtable_name.lower()}
                            """)
                            conn.unregister(f"temp_{subtable_name.lower()}")
                            
                            self.logger.info(f"Created DuckDB table: {subtable_name.lower()}")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to convert subtable {subtable_name}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Failed to create DuckDB tables: {e}")
            raise
    
    def _read_subtable_to_dataframe(self, sub_table) -> pd.DataFrame:
        """
        Read a subtable into a pandas DataFrame.
        
        Args:
            sub_table: Opened casacore subtable
        
        Returns:
            DataFrame with subtable data
        """
        data = {}
        colnames = sub_table.colnames()
        
        for col in colnames:
            try:
                col_data = sub_table.getcol(col)
                
                # Handle different data types
                if col_data.dtype in [np.complex64, np.complex128]:
                    # For subtables, we'll store complex as strings for simplicity
                    data[col] = [str(x) for x in col_data]
                elif col_data.ndim > 1:
                    # Convert arrays to strings for subtables
                    data[col] = [str(x.tolist()) for x in col_data]
                else:
                    data[col] = col_data
                    
            except Exception as e:
                self.logger.warning(f"Skipping column {col} in subtable: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def _create_unified_view(self, output_path: Path) -> None:
        """
        Create a unified DuckDB view that joins main data with subtables.
        
        Args:
            output_path: Output directory path
        """
        self.logger.info("Creating unified view...")
        
        db_path = output_path / "data.duckdb"
        main_data_dir = output_path / "main_data"
        
        try:
            with duckdb.connect(str(db_path)) as conn:
                # Register the Parquet dataset
                conn.execute(f"""
                    CREATE TABLE main_data AS 
                    SELECT * FROM read_parquet('{main_data_dir}/*/*.parquet')
                """)
                
                # Create a comprehensive view with common joins
                view_sql = """
                CREATE VIEW main_view AS
                SELECT 
                    m.*,
                    a1.NAME as ANTENNA1_NAME,
                    a2.NAME as ANTENNA2_NAME,
                    f.NAME as FIELD_NAME,
                    f.PHASE_DIR as FIELD_PHASE_DIR,
                    spw.NAME as SPW_NAME,
                    spw.NUM_CHAN,
                    spw.CHAN_FREQ,
                    obs.OBSERVER,
                    obs.PROJECT
                FROM main_data m
                LEFT JOIN antenna a1 ON m.ANTENNA1 = a1.ROWID
                LEFT JOIN antenna a2 ON m.ANTENNA2 = a2.ROWID
                LEFT JOIN field f ON m.FIELD_ID = f.ROWID
                LEFT JOIN spectral_window spw ON m.DATA_DESC_ID = spw.ROWID
                LEFT JOIN observation obs ON m.OBSERVATION_ID = obs.ROWID
                """
                
                # Execute only if the required tables exist
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]
                
                if all(t in table_names for t in ['main_data', 'antenna', 'field']):
                    conn.execute(view_sql)
                    self.logger.info("Created unified view: main_view")
                else:
                    # Create a simpler view if not all tables are available
                    conn.execute("CREATE VIEW main_view AS SELECT * FROM main_data")
                    self.logger.info("Created simple view (some subtables missing)")
        
        except Exception as e:
            self.logger.error(f"Failed to create unified view: {e}")
            raise


def main():
    """Command-line interface for the MS to DuckDB converter."""
    parser = argparse.ArgumentParser(
        description="Convert CASA Measurement Set to DuckDB/Parquet format"
    )
    parser.add_argument(
        "ms_path", 
        help="Path to the source Measurement Set directory"
    )
    parser.add_argument(
        "output_path", 
        nargs='?',
        help="Path for the output directory (default: <ms_name>.duckdb)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=100000,
        help="Number of rows to process at a time (default: 100000)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output path
    if args.output_path is None:
        ms_path = Path(args.ms_path)
        output_path = ms_path.parent / f"{ms_path.name}.duckdb"
    else:
        output_path = args.output_path
    
    # Create converter and run conversion
    converter = MSToDuckDBConverter(chunk_size=args.chunk_size)
    
    try:
        converter.convert(args.ms_path, output_path)
        print(f"Conversion completed successfully. Output: {output_path}")
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()