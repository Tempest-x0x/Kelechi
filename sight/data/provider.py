"""SIGHT Data Provider - Parquet-based data access layer."""
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import lru_cache

from ..core.base import DataProvider, BaseLogger
from ..core.types import Timeframe, OHLCV
from ..core.constants import PARQUET_DIR, SUPPORTED_PAIRS


class ParquetDataProvider(DataProvider):
    """
    High-performance data provider using Parquet files.
    
    Features:
    - Memory-mapped access for large datasets
    - LRU caching for frequently accessed data
    - Vectorized operations support
    - Multi-timeframe data retrieval
    """
    
    def __init__(
        self,
        data_dir: str = PARQUET_DIR,
        cache_size: int = 32
    ):
        super().__init__("ParquetDataProvider")
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size
        self._cache: Dict[str, pd.DataFrame] = {}
        self._spreads: Dict[str, float] = self._initialize_spreads()
    
    def _initialize_spreads(self) -> Dict[str, float]:
        """Initialize typical spreads for each pair (in pips)."""
        return {
            "EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 0.9,
            "USDCHF": 1.5, "AUDUSD": 1.0, "USDCAD": 1.8,
            "EURJPY": 1.5, "GBPJPY": 2.5, "AUDJPY": 1.8,
            "EURCHF": 2.0, "XAUUSD": 2.5, "NZDUSD": 1.5
        }
    
    def _get_cache_key(
        self, 
        pair: str, 
        timeframe: Timeframe,
        start: datetime,
        end: datetime
    ) -> str:
        """Generate cache key for data request."""
        return f"{pair}_{timeframe.value}_{start.isoformat()}_{end.isoformat()}"
    
    def _timeframe_to_dir(self, timeframe: Timeframe) -> str:
        """Convert Timeframe enum to directory name."""
        mapping = {
            Timeframe.M1: "1min",
            Timeframe.M5: "5min",
            Timeframe.M15: "15min",
            Timeframe.M30: "30min",
            Timeframe.H1: "1H",
            Timeframe.H4: "4H",
            Timeframe.D1: "1D",
            Timeframe.W1: "1W"
        }
        return mapping.get(timeframe, "1min")
    
    def _load_parquet(self, pair: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """
        Load full Parquet file for pair/timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Data timeframe
            
        Returns:
            DataFrame or None if not found
        """
        tf_dir = self._timeframe_to_dir(timeframe)
        file_path = self.data_dir / tf_dir / f"{pair}.parquet"
        
        if not file_path.exists():
            self.log_warning(f"Parquet file not found: {file_path}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.log_error(f"Error loading {file_path}: {e}")
            return None
    
    def get_ohlcv(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for specified pair, timeframe, and date range.
        
        Args:
            pair: Currency pair (e.g., "EURUSD")
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            
        Returns:
            OHLCV DataFrame with datetime index
        """
        cache_key = f"{pair}_{timeframe.value}"
        
        # Check if full dataset is cached
        if cache_key not in self._cache:
            df = self._load_parquet(pair, timeframe)
            if df is None:
                return pd.DataFrame()
            
            # Cache management - remove oldest if at capacity
            if len(self._cache) >= self.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = df
        
        # Slice cached data to requested range
        df = self._cache[cache_key]
        
        # Filter to date range
        mask = (df.index >= start) & (df.index <= end)
        result = df.loc[mask].copy()
        
        return result
    
    def get_latest_candle(self, pair: str, timeframe: Timeframe) -> Optional[OHLCV]:
        """
        Get the most recent completed candle.
        
        Args:
            pair: Currency pair
            timeframe: Data timeframe
            
        Returns:
            OHLCV object or None
        """
        cache_key = f"{pair}_{timeframe.value}"
        
        if cache_key not in self._cache:
            df = self._load_parquet(pair, timeframe)
            if df is None:
                return None
            self._cache[cache_key] = df
        
        df = self._cache[cache_key]
        
        if len(df) == 0:
            return None
        
        last_row = df.iloc[-1]
        
        return OHLCV(
            timestamp=df.index[-1],
            open=float(last_row['open']),
            high=float(last_row['high']),
            low=float(last_row['low']),
            close=float(last_row['close']),
            volume=float(last_row['volume'])
        )
    
    def get_spread(self, pair: str) -> float:
        """
        Get current bid-ask spread in pips.
        
        Args:
            pair: Currency pair
            
        Returns:
            Spread in pips
        """
        return self._spreads.get(pair, 2.0)
    
    def set_spread(self, pair: str, spread: float) -> None:
        """Update spread for a pair."""
        self._spreads[pair] = spread
    
    def get_multi_timeframe(
        self,
        pair: str,
        timeframes: List[Timeframe],
        start: datetime,
        end: datetime
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Retrieve data for multiple timeframes at once.
        
        Args:
            pair: Currency pair
            timeframes: List of timeframes
            start: Start datetime
            end: End datetime
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        result = {}
        
        for tf in timeframes:
            result[tf] = self.get_ohlcv(pair, tf, start, end)
        
        return result
    
    def get_aligned_data(
        self,
        pair: str,
        htf: Timeframe,
        ltf: Timeframe,
        start: datetime,
        end: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get HTF and LTF data aligned for analysis.
        
        Args:
            pair: Currency pair
            htf: Higher timeframe
            ltf: Lower timeframe
            start: Start datetime
            end: End datetime
            
        Returns:
            Tuple of (htf_df, ltf_df) aligned DataFrames
        """
        htf_df = self.get_ohlcv(pair, htf, start, end)
        ltf_df = self.get_ohlcv(pair, ltf, start, end)
        
        return htf_df, ltf_df
    
    def preload_pairs(self, pairs: List[str], timeframes: List[Timeframe]) -> None:
        """
        Preload data into cache for faster backtesting.
        
        Args:
            pairs: List of pairs to preload
            timeframes: List of timeframes to preload
        """
        self.log_info(f"Preloading {len(pairs)} pairs x {len(timeframes)} timeframes")
        
        for pair in pairs:
            for tf in timeframes:
                cache_key = f"{pair}_{tf.value}"
                if cache_key not in self._cache:
                    df = self._load_parquet(pair, tf)
                    if df is not None:
                        self._cache[cache_key] = df
        
        self.log_info(f"Preload complete. Cache size: {len(self._cache)}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.log_info("Cache cleared")
    
    def get_available_pairs(self) -> List[str]:
        """Get list of pairs with available data."""
        available = []
        
        for pair in SUPPORTED_PAIRS:
            # Check if 1-minute data exists
            path = self.data_dir / "1min" / f"{pair}.parquet"
            if path.exists():
                available.append(pair)
        
        return available
    
    def get_data_range(self, pair: str, timeframe: Timeframe) -> tuple[datetime, datetime]:
        """
        Get available date range for pair/timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Data timeframe
            
        Returns:
            Tuple of (start_date, end_date)
        """
        cache_key = f"{pair}_{timeframe.value}"
        
        if cache_key not in self._cache:
            df = self._load_parquet(pair, timeframe)
            if df is None:
                return (datetime.min, datetime.min)
            self._cache[cache_key] = df
        
        df = self._cache[cache_key]
        
        return (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())


class VectorizedDataView(BaseLogger):
    """
    Provides vectorized views of data for high-speed backtesting.
    
    Converts DataFrame operations to numpy array operations
    for maximum performance.
    """
    
    def __init__(self, df: pd.DataFrame):
        super().__init__("VectorizedDataView")
        self.df = df
        self._prepare_arrays()
    
    def _prepare_arrays(self) -> None:
        """Convert DataFrame columns to numpy arrays."""
        self.timestamps = self.df.index.values
        self.open = self.df['open'].values.astype(np.float64)
        self.high = self.df['high'].values.astype(np.float64)
        self.low = self.df['low'].values.astype(np.float64)
        self.close = self.df['close'].values.astype(np.float64)
        self.volume = self.df['volume'].values.astype(np.float64)
        self.n_candles = len(self.df)
    
    @property
    def range_size(self) -> np.ndarray:
        """Vectorized candle range calculation."""
        return self.high - self.low
    
    @property
    def body_size(self) -> np.ndarray:
        """Vectorized body size calculation."""
        return np.abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> np.ndarray:
        """Vectorized bullish candle detection."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> np.ndarray:
        """Vectorized bearish candle detection."""
        return self.close < self.open
    
    def get_slice(self, start_idx: int, end_idx: int) -> 'VectorizedDataView':
        """Get a slice of the data view."""
        sliced_df = self.df.iloc[start_idx:end_idx]
        return VectorizedDataView(sliced_df)
    
    def rolling_high(self, window: int) -> np.ndarray:
        """Vectorized rolling high calculation."""
        return pd.Series(self.high).rolling(window).max().values
    
    def rolling_low(self, window: int) -> np.ndarray:
        """Vectorized rolling low calculation."""
        return pd.Series(self.low).rolling(window).min().values
    
    def atr(self, period: int = 14) -> np.ndarray:
        """Vectorized ATR calculation."""
        high_low = self.high - self.low
        high_close_prev = np.abs(self.high - np.roll(self.close, 1))
        low_close_prev = np.abs(self.low - np.roll(self.close, 1))
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        tr[0] = high_low[0]  # First value has no previous close
        
        # EMA-based ATR
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(tr)):
            atr[i] = (tr[i] - atr[i-1]) * multiplier + atr[i-1]
        
        return atr
    
    def ema(self, period: int, source: str = 'close') -> np.ndarray:
        """Vectorized EMA calculation."""
        data = getattr(self, source)
        ema = np.zeros(len(data))
        ema[period-1] = np.mean(data[:period])
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def sma(self, period: int, source: str = 'close') -> np.ndarray:
        """Vectorized SMA calculation."""
        data = getattr(self, source)
        return pd.Series(data).rolling(period).mean().values
    
    def std(self, period: int, source: str = 'close') -> np.ndarray:
        """Vectorized standard deviation calculation."""
        data = getattr(self, source)
        return pd.Series(data).rolling(period).std().values
