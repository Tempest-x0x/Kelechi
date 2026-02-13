"""SIGHT Technical Indicators - Vectorized indicator calculations."""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..core.base import BaseLogger


@dataclass
class BollingerBands:
    """Bollinger Bands values container."""
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray
    bandwidth: np.ndarray
    percent_b: np.ndarray


@dataclass
class KeltnerChannel:
    """Keltner Channel values container."""
    upper: np.ndarray
    middle: np.ndarray
    lower: np.ndarray
    atr: np.ndarray


class TechnicalIndicators(BaseLogger):
    """
    Vectorized technical indicator calculations.
    
    All methods are designed for high-performance backtesting
    using numpy arrays and pandas vectorization.
    """
    
    def __init__(self):
        super().__init__("TechnicalIndicators")
    
    def ema(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Formula:
        EMA = Price(t) * k + EMA(t-1) * (1-k)
        where k = 2 / (period + 1)
        
        Args:
            data: Price array
            period: EMA period
            
        Returns:
            EMA array
        """
        ema = np.zeros(len(data))
        ema[:period] = np.nan
        
        # Initial SMA
        ema[period - 1] = np.mean(data[:period])
        
        # EMA calculation
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def sma(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price array
            period: SMA period
            
        Returns:
            SMA array
        """
        return pd.Series(data).rolling(period).mean().values
    
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Average True Range.
        
        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        ATR = EMA(True Range, period)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR array
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Use EMA for ATR
        atr = self.ema(tr, period)
        
        return atr
    
    def bollinger_bands(
        self,
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> BollingerBands:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(close, period)
        Upper Band = Middle + (std_dev * STD(close, period))
        Lower Band = Middle - (std_dev * STD(close, period))
        
        %B = (Price - Lower) / (Upper - Lower)
        Bandwidth = (Upper - Lower) / Middle
        
        Args:
            close: Close prices
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            BollingerBands dataclass
        """
        close_series = pd.Series(close)
        
        middle = close_series.rolling(period).mean().values
        std = close_series.rolling(period).std().values
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # Calculate %B
        bandwidth = (upper - lower) / middle
        percent_b = (close - lower) / (upper - lower)
        
        return BollingerBands(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )
    
    def keltner_channel(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20,
        atr_multiplier: float = 1.5,
        atr_period: int = 10
    ) -> KeltnerChannel:
        """
        Calculate Keltner Channel.
        
        Middle = EMA(close, period)
        Upper = Middle + (atr_multiplier * ATR)
        Lower = Middle - (atr_multiplier * ATR)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: EMA period
            atr_multiplier: ATR multiplier for band width
            atr_period: ATR calculation period
            
        Returns:
            KeltnerChannel dataclass
        """
        middle = self.ema(close, period)
        atr = self.atr(high, low, close, atr_period)
        
        upper = middle + (atr_multiplier * atr)
        lower = middle - (atr_multiplier * atr)
        
        return KeltnerChannel(
            upper=upper,
            middle=middle,
            lower=lower,
            atr=atr
        )
    
    def rsi(
        self,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Note: RSI is included for completeness but NOT used in
        the SIGHT confluence filter per requirements.
        
        Args:
            close: Close prices
            period: RSI period
            
        Returns:
            RSI array (0-100)
        """
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = self.ema(gains, period)
        avg_loss = self.ema(losses, period)
        
        # Avoid division by zero
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self,
        close: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            close: Close prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = self.ema(close, fast_period)
        slow_ema = self.ema(close, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return (macd_line, signal_line, histogram)
    
    def pivot_points(
        self,
        high: float,
        low: float,
        close: float
    ) -> dict:
        """
        Calculate pivot points from previous period OHLC.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dict with P, R1, R2, R3, S1, S2, S3
        """
        pivot = (high + low + close) / 3
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'P': pivot,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
    
    def vwap(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price.
        
        VWAP = Cumsum(Typical Price * Volume) / Cumsum(Volume)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            
        Returns:
            VWAP array
        """
        typical_price = (high + low + close) / 3
        vp = typical_price * volume
        
        cumulative_vp = np.cumsum(vp)
        cumulative_volume = np.cumsum(volume)
        
        # Avoid division by zero
        vwap = np.where(
            cumulative_volume != 0,
            cumulative_vp / cumulative_volume,
            typical_price
        )
        
        return vwap
    
    def check_boundary_touch(
        self,
        price: float,
        boundary: float,
        tolerance_pct: float = 0.001
    ) -> bool:
        """
        Check if price touched a boundary level.
        
        Args:
            price: Current price
            boundary: Boundary level to check
            tolerance_pct: Tolerance percentage
            
        Returns:
            True if price is within tolerance of boundary
        """
        tolerance = boundary * tolerance_pct
        return abs(price - boundary) <= tolerance
