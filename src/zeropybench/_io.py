"""I/O utilities for reading and writing benchmark data."""

from pathlib import Path

import polars as pl


class BenchmarkWriter:
    """Writes benchmark data to various file formats."""

    def __init__(self, df: pl.DataFrame, repeat: int, min_duration_per_repeat: float) -> None:
        self.df = df
        self.repeat = repeat
        self.min_duration_per_repeat = min_duration_per_repeat

    def write_csv(self, path: Path | str) -> None:
        """Writes the benchmark report as CSV.

        The metadata (``repeat`` and ``min_duration_per_repeat``) is stored in
        the name of an extra empty column, so that the file remains valid CSV
        readable by any CSV parser while still displaying the parameters as
        comment-like lines when viewed in a text editor.

        Args:
            path: The path of the CSV file.
        """
        if not isinstance(path, Path):
            path = Path(path)
        metadata_col = (
            f'\n# repeat = {self.repeat}'
            f'\n# min_duration_per_repeat = {self.min_duration_per_repeat}'
        )
        self.df.with_columns(
            execution_times='['
            + pl.col('execution_times').cast(pl.List(pl.String)).list.join(', ')
            + ']',
            **{metadata_col: pl.lit(None)},
        ).write_csv(path)

    def write_parquet(self, path: Path | str) -> None:
        """Writes the benchmark report as Parquet.

        The file includes metadata:
        - ``repeat``: The number of measurement repetitions
        - ``min_duration_per_repeat``: The minimum duration per repeat in seconds

        Args:
            path: The path of the Parquet file.
        """
        metadata = {
            'repeat': str(self.repeat),
            'min_duration_per_repeat': str(self.min_duration_per_repeat),
        }
        self.df.write_parquet(path, metadata=metadata)

    def write_markdown(self, path: Path | str) -> None:
        """Writes the benchmark report as MarkDown table.

        Args:
            path: The path of the MarkDown file.
        """
        if not isinstance(path, Path):
            path = Path(path)
        with pl.Config(
            tbl_formatting='ASCII_MARKDOWN',
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):
            path.write_text(str(self.df))


class BenchmarkReader:
    """Reads benchmark data from various file formats."""

    def __init__(self, default_repeat: int, default_min_duration_per_repeat: float) -> None:
        self.default_repeat = default_repeat
        self.default_min_duration_per_repeat = default_min_duration_per_repeat

    def read(self, path: Path | str) -> tuple[pl.DataFrame, int, float]:
        """Reads a benchmark from a CSV or Parquet file.

        Args:
            path: The path to the CSV or Parquet file.

        Returns:
            A tuple of (DataFrame, repeat, min_duration_per_repeat).

        Raises:
            ValueError: If the file extension is not .csv or .parquet.
        """
        if not isinstance(path, Path):
            path = Path(path)

        suffix = path.suffix.lower()
        if suffix == '.csv':
            return self._read_csv(path)
        elif suffix == '.parquet':
            return self._read_parquet(path)
        else:
            raise ValueError(f'Unsupported file extension: {suffix}. Use .csv or .parquet.')

    def _read_csv(self, path: Path) -> tuple[pl.DataFrame, int, float]:
        """Read a benchmark from a CSV file with metadata.

        Metadata is stored in the name of an extra column whose name contains
        ``# key = value`` lines. Leading lines starting with ``#`` are skipped.
        If no metadata column is found, default values are used.
        """
        content = path.read_text()
        lines = content.split('\n')

        # Skip leading comment lines
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                data_start = i + 1
            else:
                break

        df = pl.read_csv('\n'.join(lines[data_start:]).encode())

        # Extract metadata from column names containing "# key = value" lines
        repeat = self.default_repeat
        min_duration_per_repeat = self.default_min_duration_per_repeat
        metadata_cols: list[str] = []
        for col in df.columns:
            if not col.startswith('\n# '):
                continue
            metadata_cols.append(col)
            for line in col.strip().splitlines():
                if line.startswith('# ') and '=' in line:
                    key, value = line[2:].split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'repeat':
                        repeat = int(value)
                    elif key == 'min_duration_per_repeat':
                        min_duration_per_repeat = float(value)
        if metadata_cols:
            df = df.drop(metadata_cols)

        # Parse execution_times from string back to list
        if 'execution_times' in df.columns:
            df = df.with_columns(
                pl.col('execution_times')
                .str.strip_chars('[]')
                .str.split(', ')
                .list.eval(pl.element().cast(pl.Float64))
            )

        return df, repeat, min_duration_per_repeat

    def _read_parquet(self, path: Path) -> tuple[pl.DataFrame, int, float]:
        """Read a benchmark from a Parquet file with metadata."""
        metadata = pl.read_parquet_metadata(path)

        repeat = int(metadata.get('repeat', str(self.default_repeat)))
        min_duration_per_repeat = float(
            metadata.get('min_duration_per_repeat', str(self.default_min_duration_per_repeat))
        )

        df = pl.read_parquet(path)

        return df, repeat, min_duration_per_repeat
