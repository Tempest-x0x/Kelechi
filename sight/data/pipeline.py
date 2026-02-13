"""SIGHT Data Pipeline - High-performance data ingestion and transformation."""
import os
import zipfile
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.base import BaseLogger
from ..core.constants import (
    PARQUET_COMPRESSION, SUPPORTED_PAIRS, PARQUET_DIR,
    HISTORICAL_DATA_DIR, RESAMPLE_TIMEFRAMES
)
from ..core.types import Timeframe


class DataPipeline(BaseLogger):
    """
    High-performance data pipeline for SIGHT trading engine.
    
    Responsibilities:
    1. Extract ZIP files from historical_data directory
    2. Standardize CSV schema to Timestamp, OHLCV format
    3. Convert to Parquet with compression
    4. Resample 1m data to 15m and 1H
    5. Support vectorized backtesting via memory-mapped access
    """
    
    STANDARD_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(
        self,
        historical_data_dir: str = HISTORICAL_DATA_DIR,
        output_dir: str = PARQUET_DIR,
        n_workers: int = 4
    ):
        super().__init__("DataPipeline")
        self.historical_data_dir = Path(historical_data_dir)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for tf in ['1min', '15min', '1H']:
            (self.output_dir / tf).mkdir(exist_ok=True)
    
    def discover_zip_files(self) -> Dict[str, List[Path]]:
        """
        Discover all ZIP files and organize by currency pair.
        
        Returns:
            Dict mapping pair names to list of ZIP file paths
        """
        zip_files: Dict[str, List[Path]] = {}
        
        pattern = str(self.historical_data_dir / "*.zip")
        for zip_path in glob.glob(pattern):
            path = Path(zip_path)
            # Extract pair from filename: HISTDATA_COM_MT_EURUSD_M12024.zip
            filename = path.stem
            parts = filename.split('_')
            
            # Find the pair in the filename
            pair = None
            for p in SUPPORTED_PAIRS:
                if p in filename.upper():
                    pair = p
                    break
            
            if pair:
                if pair not in zip_files:
                    zip_files[pair] = []
                zip_files[pair].append(path)
                self.log_debug(f"Discovered {pair}: {path.name}")
        
        self.log_info(f"Discovered {len(zip_files)} pairs with {sum(len(v) for v in zip_files.values())} ZIP files")
        return zip_files
    
    def extract_and_parse_csv(
        self, 
        zip_path: Path,
        pair: str
    ) -> Optional[pd.DataFrame]:
        """
        Extract CSV from ZIP and parse to standardized DataFrame.
        
        HistData format: Date,Time,Open,High,Low,Close,Volume
        Example: 2024.01.01,17:00,1.104270,1.104290,1.104250,1.104290,0
        
        Args:
            zip_path: Path to ZIP file
            pair: Currency pair name
            
        Returns:
            Standardized DataFrame or None on error
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find CSV file in archive
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    self.log_warning(f"No CSV found in {zip_path}")
                    return None
                
                csv_name = csv_files[0]
                
                with zf.open(csv_name) as csv_file:
                    # HistData has no header, columns are:
                    # Date, Time, Open, High, Low, Close, Volume
                    df = pd.read_csv(
                        csv_file,
                        header=None,
                        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
                        dtype={
                            'open': np.float64,
                            'high': np.float64,
                            'low': np.float64,
                            'close': np.float64,
                            'volume': np.float64
                        }
                    )
                    
                    # Combine date and time columns
                    df['timestamp'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'],
                        format='%Y.%m.%d %H:%M'
                    )
                    
                    # Select and reorder columns
                    df = df[self.STANDARD_COLUMNS].copy()
                    df = df.set_index('timestamp').sort_index()
                    
                    # Remove duplicates
                    df = df[~df.index.duplicated(keep='first')]
                    
                    self.log_debug(f"Parsed {len(df)} rows from {zip_path.name}")
                    return df
                    
        except Exception as e:
            self.log_error(f"Error processing {zip_path}: {e}")
            return None
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames, handling overlaps and gaps.
        
        Args:
            dataframes: List of DataFrames to merge
            
        Returns:
            Merged and sorted DataFrame
        """
        if not dataframes:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS[1:])
        
        # Concatenate all dataframes
        merged = pd.concat(dataframes, axis=0)
        
        # Remove duplicate timestamps, keeping first occurrence
        merged = merged[~merged.index.duplicated(keep='first')]
        
        # Sort by timestamp
        merged = merged.sort_index()
        
        return merged
    
    def resample_ohlcv(
        self, 
        df: pd.DataFrame, 
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe using proper aggregation.
        
        Aggregation rules:
        - Open: First value in period
        - High: Maximum value in period
        - Low: Minimum value in period
        - Close: Last value in period
        - Volume: Sum of volumes in period
        
        Args:
            df: Source 1-minute DataFrame
            target_timeframe: Target timeframe string (e.g., '15min', '1H')
            
        Returns:
            Resampled DataFrame
        """
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample using pandas
        resampled = df.resample(target_timeframe).agg(agg_rules)
        
        # Drop rows with NaN (incomplete periods)
        resampled = resampled.dropna()
        
        return resampled
    
    def save_to_parquet(
        self, 
        df: pd.DataFrame, 
        pair: str, 
        timeframe: str
    ) -> Path:
        """
        Save DataFrame to Parquet with compression.
        
        Args:
            df: DataFrame to save
            pair: Currency pair name
            timeframe: Timeframe string
            
        Returns:
            Path to saved Parquet file
        """
        output_path = self.output_dir / timeframe / f"{pair}.parquet"
        
        # Reset index to include timestamp as column
        df_to_save = df.reset_index()
        
        df_to_save.to_parquet(
            output_path,
            compression=PARQUET_COMPRESSION,
            index=False
        )
        
        self.log_info(f"Saved {pair} {timeframe}: {len(df)} rows to {output_path}")
        return output_path
    
    def process_pair(self, pair: str, zip_files: List[Path]) -> Dict[str, Path]:
        """
        Process all data for a single currency pair.
        
        Pipeline steps:
        1. Extract and parse all ZIP files
        2. Merge into single DataFrame
        3. Save 1m Parquet
        4. Resample to 15m and 1H
        5. Save resampled Parquet files
        
        Args:
            pair: Currency pair name
            zip_files: List of ZIP file paths
            
        Returns:
            Dict mapping timeframe to output Parquet path
        """
        self.log_info(f"Processing {pair}: {len(zip_files)} files")
        
        # Step 1: Extract and parse all files
        dataframes = []
        for zip_path in sorted(zip_files):
            df = self.extract_and_parse_csv(zip_path, pair)
            if df is not None and len(df) > 0:
                dataframes.append(df)
        
        if not dataframes:
            self.log_error(f"No valid data found for {pair}")
            return {}
        
        # Step 2: Merge dataframes
        merged_df = self.merge_dataframes(dataframes)
        self.log_info(f"{pair}: Merged {len(merged_df)} total rows, "
                     f"range: {merged_df.index.min()} to {merged_df.index.max()}")
        
        # Step 3: Save 1-minute data
        output_paths = {}
        output_paths['1min'] = self.save_to_parquet(merged_df, pair, '1min')
        
        # Step 4 & 5: Resample and save
        for tf_name, tf_rule in RESAMPLE_TIMEFRAMES.items():
            resampled = self.resample_ohlcv(merged_df, tf_rule)
            output_paths[tf_name] = self.save_to_parquet(resampled, pair, tf_name)
        
        return output_paths
    
    def run_pipeline(
        self, 
        pairs: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Path]]:
        """
        Execute full data pipeline for all or specified pairs.
        
        Args:
            pairs: Optional list of pairs to process (default: all discovered)
            
        Returns:
            Nested dict: {pair: {timeframe: path}}
        """
        self.log_info("Starting SIGHT Data Pipeline")
        
        # Discover available data
        discovered = self.discover_zip_files()
        
        # Filter to specified pairs if provided
        if pairs:
            discovered = {p: f for p, f in discovered.items() if p in pairs}
        
        if not discovered:
            self.log_error("No data files found to process")
            return {}
        
        results: Dict[str, Dict[str, Path]] = {}
        
        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.process_pair, pair, files): pair
                for pair, files in discovered.items()
            }
            
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[pair] = result
                except Exception as e:
                    self.log_error(f"Failed to process {pair}: {e}")
        
        self.log_info(f"Pipeline complete: {len(results)} pairs processed")
        return results
    
    def validate_output(self) -> Dict[str, Dict[str, bool]]:
        """
        Validate pipeline output by checking Parquet files.
        
        Returns:
            Validation results: {pair: {timeframe: valid}}
        """
        validation = {}
        
        for pair in SUPPORTED_PAIRS:
            validation[pair] = {}
            for tf in ['1min', '15min', '1H']:
                path = self.output_dir / tf / f"{pair}.parquet"
                if path.exists():
                    try:
                        df = pd.read_parquet(path)
                        validation[pair][tf] = len(df) > 0
                    except Exception:
                        validation[pair][tf] = False
                else:
                    validation[pair][tf] = False
        
        return validation
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Generate summary of all processed data.
        
        Returns:
            Summary DataFrame with pair stats
        """
        summary_data = []
        
        for pair in SUPPORTED_PAIRS:
            for tf in ['1min', '15min', '1H']:
                path = self.output_dir / tf / f"{pair}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    summary_data.append({
                        'pair': pair,
                        'timeframe': tf,
                        'rows': len(df),
                        'start': df['timestamp'].min() if 'timestamp' in df.columns else df.index.min(),
                        'end': df['timestamp'].max() if 'timestamp' in df.columns else df.index.max(),
                        'file_size_mb': path.stat().st_size / (1024 * 1024)
                    })
        
        return pd.DataFrame(summary_data)


class DataValidator(BaseLogger):
    """Validates data quality for backtesting suitability."""
    
    def __init__(self):
        super().__init__("DataValidator")
    
    def check_gaps(
        self, 
        df: pd.DataFrame,
        expected_freq: str = '1min',
        max_gap_minutes: int = 60
    ) -> List[Tuple[datetime, datetime, int]]:
        """
        Detect data gaps exceeding threshold.
        
        Args:
            df: DataFrame with datetime index
            expected_freq: Expected data frequency
            max_gap_minutes: Maximum allowed gap in minutes
            
        Returns:
            List of (start, end, gap_minutes) tuples
        """
        gaps = []
        
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = df.index
        
        time_diff = timestamps.diff()
        
        for i, diff in enumerate(time_diff):
            if pd.notna(diff):
                gap_minutes = diff.total_seconds() / 60
                if gap_minutes > max_gap_minutes:
                    gaps.append((
                        timestamps.iloc[i-1],
                        timestamps.iloc[i],
                        int(gap_minutes)
                    ))
        
        return gaps
    
    def check_price_anomalies(
        self, 
        df: pd.DataFrame,
        max_change_percent: float = 5.0
    ) -> pd.DataFrame:
        """
        Detect suspicious price movements.
        
        Args:
            df: OHLCV DataFrame
            max_change_percent: Maximum expected single-candle change
            
        Returns:
            DataFrame of anomalous rows
        """
        df = df.copy()
        
        # Calculate percentage changes
        df['change_pct'] = df['close'].pct_change().abs() * 100
        
        anomalies = df[df['change_pct'] > max_change_percent]
        
        return anomalies
    
    def validate_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check OHLC consistency rules:
        - High >= Open, Close
        - Low <= Open, Close
        - High >= Low
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame of inconsistent rows
        """
        inconsistent = df[
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        ]
        
        return inconsistent
