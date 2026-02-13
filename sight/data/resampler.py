"""SIGHT OHLCV Resampler - Timeframe conversion utilities."""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..core.base import BaseLogger
from ..core.types import Timeframe


class OHLCVResampler(BaseLogger):
    """
    OHLCV data resampling utility with proper aggregation rules.
    
    Supports both downsampling (1m -> 15m) and upsampling with fill-forward.
    """
    
    AGGREGATION_RULES = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    TIMEFRAME_MAP: Dict[Timeframe, str] = {
        Timeframe.M1: '1min',
        Timeframe.M5: '5min',
        Timeframe.M15: '15min',
        Timeframe.M30: '30min',
        Timeframe.H1: '1H',
        Timeframe.H4: '4H',
        Timeframe.D1: '1D',
        Timeframe.W1: '1W'
    }
    
    def __init__(self):
        super().__init__("OHLCVResampler")
    
    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: Timeframe,
        source_timeframe: Optional[Timeframe] = None
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.
        
        Args:
            df: Source DataFrame with datetime index
            target_timeframe: Target timeframe
            source_timeframe: Optional source timeframe for validation
            
        Returns:
            Resampled DataFrame
        """
        if len(df) == 0:
            return df
        
        # Validate source vs target
        if source_timeframe:
            if target_timeframe.minutes < source_timeframe.minutes:
                self.log_error("Cannot upsample OHLCV data")
                return df
        
        target_rule = self.TIMEFRAME_MAP[target_timeframe]
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                self.log_error("DataFrame must have datetime index or timestamp column")
                return df
        
        # Perform resampling
        resampled = df.resample(target_rule).agg(self.AGGREGATION_RULES)
        
        # Drop incomplete periods
        resampled = resampled.dropna()
        
        self.log_debug(f"Resampled {len(df)} -> {len(resampled)} rows to {target_timeframe.value}")
        
        return resampled
    
    def resample_with_session_alignment(
        self,
        df: pd.DataFrame,
        target_timeframe: Timeframe,
        session_start_hour: int = 0
    ) -> pd.DataFrame:
        """
        Resample with session boundary alignment.
        
        Useful for daily candles that need to align with specific
        session opens (e.g., New York midnight vs Sydney midnight).
        
        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe
            session_start_hour: Hour of session start (UTC)
            
        Returns:
            Session-aligned resampled DataFrame
        """
        if target_timeframe not in [Timeframe.D1, Timeframe.W1]:
            return self.resample(df, target_timeframe)
        
        # Shift timestamps to align with session
        df_shifted = df.copy()
        df_shifted.index = df_shifted.index - timedelta(hours=session_start_hour)
        
        # Resample
        resampled = self.resample(df_shifted, target_timeframe)
        
        # Shift back
        resampled.index = resampled.index + timedelta(hours=session_start_hour)
        
        return resampled
    
    def create_multi_timeframe_dataset(
        self,
        df_1m: pd.DataFrame,
        timeframes: list[Timeframe]
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Create aligned multi-timeframe dataset from 1-minute data.
        
        Args:
            df_1m: 1-minute source data
            timeframes: List of target timeframes
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        result = {Timeframe.M1: df_1m}
        
        for tf in timeframes:
            if tf == Timeframe.M1:
                continue
            result[tf] = self.resample(df_1m, tf, Timeframe.M1)
        
        return result
    
    def align_htf_ltf(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align HTF and LTF DataFrames to common date range.
        
        Args:
            htf_df: Higher timeframe DataFrame
            ltf_df: Lower timeframe DataFrame
            
        Returns:
            Tuple of aligned (htf_df, ltf_df)
        """
        # Find common range
        start = max(htf_df.index.min(), ltf_df.index.min())
        end = min(htf_df.index.max(), ltf_df.index.max())
        
        htf_aligned = htf_df[(htf_df.index >= start) & (htf_df.index <= end)]
        ltf_aligned = ltf_df[(ltf_df.index >= start) & (ltf_df.index <= end)]
        
        return htf_aligned, ltf_aligned
    
    def forward_fill_htf(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Forward-fill HTF data to match LTF timestamps.
        
        Useful for creating a single DataFrame where each LTF row
        has the corresponding HTF values.
        
        Args:
            htf_df: Higher timeframe DataFrame
            ltf_df: Lower timeframe DataFrame
            
        Returns:
            HTF data forward-filled to LTF timestamps
        """
        # Reindex HTF to LTF timestamps and forward-fill
        htf_reindexed = htf_df.reindex(ltf_df.index, method='ffill')
        
        # Rename columns to avoid conflicts
        htf_reindexed.columns = [f'htf_{col}' for col in htf_reindexed.columns]
        
        return htf_reindexed


class SessionSplitter(BaseLogger):
    """Split data by trading sessions."""
    
    SESSIONS = {
        'ASIA': (0, 8),      # 00:00 - 08:00 UTC
        'LONDON': (7, 16),   # 07:00 - 16:00 UTC
        'NEW_YORK': (12, 21) # 12:00 - 21:00 UTC
    }
    
    def __init__(self):
        super().__init__("SessionSplitter")
    
    def get_session_data(
        self,
        df: pd.DataFrame,
        session: str
    ) -> pd.DataFrame:
        """
        Extract data for specific trading session.
        
        Args:
            df: Source DataFrame with datetime index
            session: Session name ('ASIA', 'LONDON', 'NEW_YORK')
            
        Returns:
            Filtered DataFrame
        """
        if session not in self.SESSIONS:
            self.log_error(f"Unknown session: {session}")
            return df
        
        start_hour, end_hour = self.SESSIONS[session]
        
        # Extract hour from index
        hours = df.index.hour
        
        if start_hour < end_hour:
            mask = (hours >= start_hour) & (hours < end_hour)
        else:
            # Handle overnight sessions
            mask = (hours >= start_hour) | (hours < end_hour)
        
        return df[mask]
    
    def get_session_high_low(
        self,
        df: pd.DataFrame,
        session: str,
        date: datetime
    ) -> tuple[float, float]:
        """
        Get previous session's high and low.
        
        Args:
            df: Source DataFrame
            session: Session name
            date: Current date
            
        Returns:
            Tuple of (session_high, session_low)
        """
        # Filter to previous day's session
        prev_date = date - timedelta(days=1)
        
        session_df = self.get_session_data(df, session)
        prev_session = session_df[
            (session_df.index.date == prev_date.date())
        ]
        
        if len(prev_session) == 0:
            return (np.nan, np.nan)
        
        return (prev_session['high'].max(), prev_session['low'].min())
    
    def label_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session labels to DataFrame.
        
        Args:
            df: Source DataFrame
            
        Returns:
            DataFrame with 'session' column
        """
        df = df.copy()
        df['session'] = 'OTHER'
        
        hours = df.index.hour
        
        for session_name, (start, end) in self.SESSIONS.items():
            if start < end:
                mask = (hours >= start) & (hours < end)
            else:
                mask = (hours >= start) | (hours < end)
            
            df.loc[mask, 'session'] = session_name
        
        return df
