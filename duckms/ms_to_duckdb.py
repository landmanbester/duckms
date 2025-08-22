#!/usr/bin/env python3
"""
CASA Measurement Set to DuckDB/Parquet Converter

This module converts CASA Measurement Sets into a hybrid DuckDB/Parquet format:
- Main data table -> Partitioned Parquet files (partitioned by OBSERVATION_ID)
- Subtables -> DuckDB tables
- Unified view joining all data together

Author: Claude
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
            "ANTENNA",
            "FIELD",
            "OBSERVATION",
            "SPECTRAL_WINDOW",
            "POLARIZATION",
            "FEED",
            "SOURCE",
            "POINTING",
            "DATA_DESCRIPTION",
            "HISTORY",
            "DOPPLER",
            "FREQ_OFFSET",
            "SYSCAL",
            "WEATHER",
        ]

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def convert(self, ms_path: str, output_path: str) -> None:
        """
        Convert a Measurement Set to DuckDB/Parquet format.

        Args:
            ms_path: Path to the source Measurement Set directory
            output_path: Path for the output directory (will be created)
        """
        ms_path_obj = Path(ms_path)
        output_path_obj = Path(output_path)

        if not ms_path_obj.exists():
            raise FileNotFoundError(f"Measurement Set not found: {ms_path}")

        self.logger.info(f"Converting {ms_path} to {output_path}")

        try:
            # Create output directory structure
            self._create_output_structure(output_path_obj)

            # Convert main table to Parquet
            self._convert_main_table(ms_path_obj, output_path_obj)

            # Convert subtables to DuckDB
            self._convert_subtables(ms_path_obj, output_path_obj)

            # Create unified view
            self._create_unified_view(output_path_obj)

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

    def _read_main_table_chunk(
        self, ms_table: "table", start_row: int, nrows: int, colnames: List[str]
    ) -> pd.DataFrame:
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
                col_data = self._get_column_data(ms_table, col, start_row, nrows)
                if col_data is None:
                    continue

                processed_data = self._process_column_data(col, col_data)
                data.update(processed_data)

            except Exception as e:
                self.logger.warning(f"Skipping column {col} due to error: {e}")
                continue

        return pd.DataFrame(data)

    def _get_column_data(
        self, ms_table: "table", col: str, start_row: int, nrows: int
    ) -> Optional[Any]:
        """Get column data with error handling."""
        try:
            col_data = ms_table.getcol(col, startrow=start_row, nrow=nrows)
        except Exception as col_error:
            # Try to get column info to understand the issue
            try:
                col_desc = ms_table.getcoldesc(col)
                if "ndim" in col_desc and col_desc.get("ndim", 0) == -1:
                    # Variable shape array that may be empty
                    self.logger.warning(
                        f"Skipping variable shape column {col} (likely empty): {col_error}"
                    )
                    return None
            except Exception:
                pass
            self.logger.warning(
                f"Skipping column {col} due to access error: {col_error}"
            )
            return None

        # Skip if no data
        if col_data is None or (hasattr(col_data, "size") and col_data.size == 0):
            self.logger.warning(f"Skipping empty column {col}")
            return None

        return col_data

    def _process_column_data(self, col: str, col_data: Any) -> Dict[str, Any]:
        """Process column data based on its type."""
        if col_data.dtype in [np.complex64, np.complex128]:
            return self._process_complex_data(col, col_data)
        elif col_data.dtype == bool:
            return self._process_boolean_data(col, col_data)
        elif col_data.ndim > 1:
            return self._process_multidimensional_data(col, col_data)
        else:
            return {col: col_data.tolist() if hasattr(col_data, "tolist") else col_data}

    def _process_complex_data(self, col: str, col_data: Any) -> Dict[str, Any]:
        """Process complex data by splitting into real/imaginary parts."""
        self.logger.debug(f"Splitting complex column {col}")

        if col_data.ndim == 1:
            return {
                f"{col}_REAL": col_data.real.astype(np.float32),
                f"{col}_IMAG": col_data.imag.astype(np.float32),
            }

        shape = col_data.shape
        flattened_real = col_data.real.flatten().astype(np.float32)
        flattened_imag = col_data.imag.flatten().astype(np.float32)
        chunk_size = np.prod(shape[1:]) if len(shape) > 1 else 1

        real_lists = []
        imag_lists = []
        for i in range(shape[0]):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            real_lists.append(flattened_real[start_idx:end_idx].tolist())
            imag_lists.append(flattened_imag[start_idx:end_idx].tolist())

        return {
            f"{col}_REAL": real_lists,
            f"{col}_IMAG": imag_lists,
            f"{col}_SHAPE": [list(shape[1:])] * shape[0],
        }

    def _process_boolean_data(self, col: str, col_data: Any) -> Dict[str, Any]:
        """Process boolean data."""
        if col_data.ndim == 1:
            return {col: col_data.astype(bool)}

        shape = col_data.shape
        flattened = col_data.flatten()
        chunk_size = np.prod(shape[1:]) if len(shape) > 1 else 1
        bool_lists = []
        for i in range(shape[0]):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            bool_lists.append(flattened[start_idx:end_idx].tolist())

        return {col: bool_lists, f"{col}_SHAPE": [list(shape[1:])] * shape[0]}

    def _process_multidimensional_data(self, col: str, col_data: Any) -> Dict[str, Any]:
        """Process multi-dimensional non-complex arrays."""
        shape = col_data.shape
        flattened = col_data.flatten()
        chunk_size = np.prod(shape[1:]) if len(shape) > 1 else 1

        array_lists = []
        for i in range(shape[0]):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            array_lists.append(flattened[start_idx:end_idx].tolist())

        return {col: array_lists, f"{col}_SHAPE": [list(shape[1:])] * shape[0]}

    def _write_parquet_chunk(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Write DataFrame chunk to Parquet, partitioned by OBSERVATION_ID.

        Args:
            df: DataFrame to write
            output_dir: Output directory for Parquet files
        """
        if "OBSERVATION_ID" not in df.columns:
            self.logger.warning(
                "No OBSERVATION_ID column found, using default partition"
            )
            df["OBSERVATION_ID"] = 0

        # Convert to PyArrow table for efficient writing
        table = pa.Table.from_pandas(df)

        # Write partitioned by OBSERVATION_ID
        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=["OBSERVATION_ID"],
            compression="snappy",
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
                        self.logger.debug(
                            f"Subtable {subtable_name} not found, skipping"
                        )
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

                            self.logger.info(
                                f"Created DuckDB table: {subtable_name.lower()}"
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to convert subtable {subtable_name}: {e}"
                        )
                        continue

        except Exception as e:
            self.logger.error(f"Failed to create DuckDB tables: {e}")
            raise

    def _read_subtable_to_dataframe(self, sub_table: "table") -> pd.DataFrame:
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
                col_data = self._get_subtable_column_data(sub_table, col)
                if col_data is None:
                    continue

                processed_data = self._process_subtable_column_data(col_data)
                data[col] = processed_data

            except Exception as e:
                self.logger.warning(f"Skipping column {col} in subtable: {e}")
                continue

        return pd.DataFrame(data)

    def _get_subtable_column_data(self, sub_table: "table", col: str) -> Optional[Any]:
        """Get subtable column data with error handling."""
        try:
            col_data = sub_table.getcol(col)
        except Exception as col_error:
            self.logger.warning(f"Skipping column {col} in subtable: {col_error}")
            return None

        if col_data is None or (hasattr(col_data, "size") and col_data.size == 0):
            self.logger.warning(f"Skipping empty column {col} in subtable")
            return None

        return col_data

    def _process_subtable_column_data(
        self, col_data: Any
    ) -> Union[List[str], List[Any]]:
        """Process subtable column data, converting to appropriate format."""
        if not hasattr(col_data, "dtype"):
            if isinstance(col_data, (list, tuple)):
                return [str(x) for x in col_data]
            return [str(col_data)]

        if col_data.dtype in [np.complex64, np.complex128]:
            return [str(x) for x in col_data]

        if col_data.dtype.kind in ["U", "S"]:
            return (
                [str(col_data)]
                if col_data.ndim == 0
                else [str(x) for x in col_data.flat]
            )

        if col_data.ndim > 1:
            return [str(x.tolist()) for x in col_data]

        return col_data.tolist() if hasattr(col_data, "tolist") else [col_data]

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

                # Get available tables
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]

                # Build dynamic view
                select_parts, join_parts = self._build_view_components(
                    conn, table_names
                )
                self._execute_view_creation(conn, select_parts, join_parts, table_names)

        except Exception as e:
            self.logger.error(f"Failed to create unified view: {e}")
            raise

    def _build_view_components(
        self, conn: "duckdb.DuckDBPyConnection", table_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Build SELECT and JOIN components for the unified view."""
        select_parts = ["m.*"]
        join_parts = []

        # Process each subtable type
        antenna_parts = self._process_antenna_table(conn, table_names)
        if antenna_parts:
            select_parts.extend(antenna_parts["select"])
            join_parts.extend(antenna_parts["join"])

        field_parts = self._process_field_table(conn, table_names)
        if field_parts:
            select_parts.extend(field_parts["select"])
            join_parts.extend(field_parts["join"])

        spw_parts = self._process_spectral_window_table(conn, table_names)
        if spw_parts:
            select_parts.extend(spw_parts["select"])
            join_parts.extend(spw_parts["join"])

        obs_parts = self._process_observation_table(conn, table_names)
        if obs_parts:
            select_parts.extend(obs_parts["select"])
            join_parts.extend(obs_parts["join"])

        return select_parts, join_parts

    def _process_antenna_table(
        self, conn: "duckdb.DuckDBPyConnection", table_names: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """Process antenna table for view creation."""
        if "antenna" not in table_names:
            return None

        try:
            antenna_cols = [
                col[0] for col in conn.execute("DESCRIBE antenna").fetchall()
            ]
            name_col = next(
                (col for col in antenna_cols if "NAME" in col.upper()), None
            )

            if name_col:
                return {
                    "select": [
                        f"a1.{name_col} as ANTENNA1_NAME",
                        f"a2.{name_col} as ANTENNA2_NAME",
                    ],
                    "join": [
                        "LEFT JOIN antenna a1 ON m.ANTENNA1 = a1.ROWID",
                        "LEFT JOIN antenna a2 ON m.ANTENNA2 = a2.ROWID",
                    ],
                }
        except Exception as e:
            self.logger.debug(f"Could not analyze antenna table: {e}")
        return None

    def _process_field_table(
        self, conn: "duckdb.DuckDBPyConnection", table_names: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """Process field table for view creation."""
        if "field" not in table_names:
            return None

        try:
            field_cols = [col[0] for col in conn.execute("DESCRIBE field").fetchall()]
            available_cols = []

            for col_name, sql_name in [
                ("NAME", "FIELD_NAME"),
                ("PHASE_DIR", "FIELD_PHASE_DIR"),
            ]:
                actual_col = next(
                    (col for col in field_cols if col_name in col.upper()), None
                )
                if actual_col:
                    available_cols.append(f"f.{actual_col} as {sql_name}")

            if available_cols:
                return {
                    "select": available_cols,
                    "join": ["LEFT JOIN field f ON m.FIELD_ID = f.ROWID"],
                }
        except Exception as e:
            self.logger.debug(f"Could not analyze field table: {e}")
        return None

    def _process_spectral_window_table(
        self, conn: "duckdb.DuckDBPyConnection", table_names: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """Process spectral_window table for view creation."""
        if "spectral_window" not in table_names:
            return None

        try:
            spw_cols = [
                col[0] for col in conn.execute("DESCRIBE spectral_window").fetchall()
            ]
            available_cols = []

            for col_name, sql_name in [
                ("NAME", "SPW_NAME"),
                ("NUM_CHAN", "NUM_CHAN"),
                ("CHAN_FREQ", "CHAN_FREQ"),
            ]:
                actual_col = next(
                    (col for col in spw_cols if col_name in col.upper()), None
                )
                if actual_col:
                    alias = f" as {sql_name}" if sql_name != col_name else ""
                    available_cols.append(f"spw.{actual_col}{alias}")

            if available_cols:
                return {
                    "select": available_cols,
                    "join": [
                        "LEFT JOIN spectral_window spw ON m.DATA_DESC_ID = spw.ROWID"
                    ],
                }
        except Exception as e:
            self.logger.debug(f"Could not analyze spectral_window table: {e}")
        return None

    def _process_observation_table(
        self, conn: "duckdb.DuckDBPyConnection", table_names: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """Process observation table for view creation."""
        if "observation" not in table_names:
            return None

        try:
            obs_cols = [
                col[0] for col in conn.execute("DESCRIBE observation").fetchall()
            ]
            available_cols = []

            for col_name in ["OBSERVER", "PROJECT"]:
                actual_col = next(
                    (col for col in obs_cols if col_name in col.upper()), None
                )
                if actual_col:
                    available_cols.append(f"obs.{actual_col}")

            if available_cols:
                return {
                    "select": available_cols,
                    "join": [
                        "LEFT JOIN observation obs ON m.OBSERVATION_ID = obs.ROWID"
                    ],
                }
        except Exception as e:
            self.logger.debug(f"Could not analyze observation table: {e}")
        return None

    def _execute_view_creation(
        self,
        conn: "duckdb.DuckDBPyConnection",
        select_parts: List[str],
        join_parts: List[str],
        table_names: List[str],
    ) -> None:
        """Execute the view creation SQL."""
        if "main_data" not in table_names:
            self.logger.warning("No main_data table found, cannot create view")
            return

        select_clause = ",\n                    ".join(select_parts)
        join_clause = " ".join(join_parts)
        view_sql = f"""
        CREATE VIEW main_view AS
        SELECT
            {select_clause}
        FROM main_data m
        {join_clause}
        """

        conn.execute(view_sql)
        self.logger.info("Created unified view: main_view")


def main() -> None:
    """Command-line interface for the MS to DuckDB converter."""
    parser = argparse.ArgumentParser(
        description="Convert CASA Measurement Set to DuckDB/Parquet format"
    )
    parser.add_argument("ms_path", help="Path to the source Measurement Set directory")
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path for the output directory (default: <ms_name>.duckdb)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows to process at a time (default: 100000)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
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
        converter.convert(args.ms_path, str(output_path))
        print(f"Conversion completed successfully. Output: {output_path}")
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
