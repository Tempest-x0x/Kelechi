"""SIGHT Confluence Filter - Statistical exhaustion confirmation (20% weight)."""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from ..core.base import ConfluenceChecker
from ..core.types import SignalType, TradeSetup, PairConfig
from ..core.constants import (
    BOLLINGER_PERIOD, BOLLINGER_STD, KELTNER_PERIOD, KELTNER_ATR_MULTIPLE
)
from .indicators import TechnicalIndicators, BollingerBands, KeltnerChannel


class ConfluenceFilter(ConfluenceChecker):
    """
    Statistical Confluence Filter - 20% decision weight.
    
    Purpose:
    - Validate exhaustion at reversal zones
    - Confirm price reaching boundary conditions
    - Filter out weak setups
    
    Valid Setup Criteria:
    - Price touches Bollinger Band boundary, OR
    - Price touches Keltner Channel boundary
    
    Note: Stochastic RSI is explicitly excluded per requirements.
    
    Usage:
    The ICT Engine provides the primary setup (80% weight).
    The Confluence Filter provides exhaustion confirmation (20% weight).
    A trade is only valid when ICT criteria are met AND at least
    one confluence filter triggers.
    """
    
    def __init__(
        self,
        bb_period: int = BOLLINGER_PERIOD,
        bb_std: float = BOLLINGER_STD,
        kc_period: int = KELTNER_PERIOD,
        kc_atr_multiple: float = KELTNER_ATR_MULTIPLE,
        touch_tolerance: float = 0.001
    ):
        super().__init__("ConfluenceFilter")
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_multiple = kc_atr_multiple
        self.touch_tolerance = touch_tolerance
        
        self.indicators = TechnicalIndicators()
        
        # Cached calculations
        self._bb_cache: Optional[BollingerBands] = None
        self._kc_cache: Optional[KeltnerChannel] = None
    
    def update_config(self, config: PairConfig) -> None:
        """Update filter configuration from PairConfig."""
        self.bb_period = config.bb_period
        self.bb_std = config.bb_std
        self.kc_period = config.kc_period
        self.kc_atr_multiple = config.kc_atr_multiple
        
        # Clear cache on config change
        self._bb_cache = None
        self._kc_cache = None
    
    def _calculate_bollinger(self, data: pd.DataFrame) -> BollingerBands:
        """Calculate Bollinger Bands for the data."""
        close = data['close'].values
        return self.indicators.bollinger_bands(close, self.bb_period, self.bb_std)
    
    def _calculate_keltner(self, data: pd.DataFrame) -> KeltnerChannel:
        """Calculate Keltner Channel for the data."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        return self.indicators.keltner_channel(
            high, low, close, 
            self.kc_period, 
            self.kc_atr_multiple
        )
    
    def check_bollinger_touch(
        self,
        data: pd.DataFrame,
        direction: SignalType,
        lookback: int = 3
    ) -> bool:
        """
        Check if price touched Bollinger Band boundary.
        
        Bollinger Touch Logic:
        - For LONG signals: Price touched lower band (exhaustion to downside)
        - For SHORT signals: Price touched upper band (exhaustion to upside)
        
        Touch is confirmed if within the last N candles:
        - Low <= Lower Band (for LONG)
        - High >= Upper Band (for SHORT)
        
        Args:
            data: OHLCV DataFrame
            direction: Trade signal direction
            lookback: Number of candles to check
            
        Returns:
            True if boundary touch confirmed
        """
        if len(data) < self.bb_period + lookback:
            return False
        
        bb = self._calculate_bollinger(data)
        self._bb_cache = bb
        
        # Get recent data
        n = len(data)
        recent_range = slice(n - lookback, n)
        
        if direction == SignalType.LONG:
            # Check if low touched lower band
            lows = data['low'].values[recent_range]
            lower_band = bb.lower[recent_range]
            
            # Touch with tolerance
            for i in range(len(lows)):
                if not np.isnan(lower_band[i]):
                    tolerance = lower_band[i] * self.touch_tolerance
                    if lows[i] <= lower_band[i] + tolerance:
                        self.log_debug(f"BB lower touch detected: {lows[i]:.5f} <= {lower_band[i]:.5f}")
                        return True
        
        elif direction == SignalType.SHORT:
            # Check if high touched upper band
            highs = data['high'].values[recent_range]
            upper_band = bb.upper[recent_range]
            
            for i in range(len(highs)):
                if not np.isnan(upper_band[i]):
                    tolerance = upper_band[i] * self.touch_tolerance
                    if highs[i] >= upper_band[i] - tolerance:
                        self.log_debug(f"BB upper touch detected: {highs[i]:.5f} >= {upper_band[i]:.5f}")
                        return True
        
        return False
    
    def check_keltner_touch(
        self,
        data: pd.DataFrame,
        direction: SignalType,
        lookback: int = 3
    ) -> bool:
        """
        Check if price touched Keltner Channel boundary.
        
        Keltner Touch Logic:
        - For LONG signals: Price touched lower channel (exhaustion)
        - For SHORT signals: Price touched upper channel (exhaustion)
        
        Args:
            data: OHLCV DataFrame
            direction: Trade signal direction
            lookback: Number of candles to check
            
        Returns:
            True if boundary touch confirmed
        """
        if len(data) < self.kc_period + lookback:
            return False
        
        kc = self._calculate_keltner(data)
        self._kc_cache = kc
        
        n = len(data)
        recent_range = slice(n - lookback, n)
        
        if direction == SignalType.LONG:
            lows = data['low'].values[recent_range]
            lower_channel = kc.lower[recent_range]
            
            for i in range(len(lows)):
                if not np.isnan(lower_channel[i]):
                    tolerance = lower_channel[i] * self.touch_tolerance
                    if lows[i] <= lower_channel[i] + tolerance:
                        self.log_debug(f"KC lower touch detected: {lows[i]:.5f} <= {lower_channel[i]:.5f}")
                        return True
        
        elif direction == SignalType.SHORT:
            highs = data['high'].values[recent_range]
            upper_channel = kc.upper[recent_range]
            
            for i in range(len(highs)):
                if not np.isnan(upper_channel[i]):
                    tolerance = upper_channel[i] * self.touch_tolerance
                    if highs[i] >= upper_channel[i] - tolerance:
                        self.log_debug(f"KC upper touch detected: {highs[i]:.5f} >= {upper_channel[i]:.5f}")
                        return True
        
        return False
    
    def calculate_confluence_score(self, setup: TradeSetup) -> float:
        """
        Calculate overall confluence score for the setup.
        
        Scoring:
        - Base ICT setup: 0.8 (80% weight)
        - BB touch: +0.1 (10% weight)
        - KC touch: +0.1 (10% weight)
        
        Maximum score: 1.0
        
        Args:
            setup: TradeSetup with confluence flags
            
        Returns:
            Confluence score (0.0 to 1.0)
        """
        score = 0.8  # ICT base weight
        
        if setup.bollinger_touch:
            score += 0.1
        
        if setup.keltner_touch:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_band_values(
        self,
        data: pd.DataFrame
    ) -> dict:
        """
        Get current Bollinger and Keltner band values.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dict with band values
        """
        bb = self._calculate_bollinger(data)
        kc = self._calculate_keltner(data)
        
        last_idx = len(data) - 1
        
        return {
            'bb_upper': bb.upper[last_idx],
            'bb_middle': bb.middle[last_idx],
            'bb_lower': bb.lower[last_idx],
            'bb_bandwidth': bb.bandwidth[last_idx],
            'bb_percent_b': bb.percent_b[last_idx],
            'kc_upper': kc.upper[last_idx],
            'kc_middle': kc.middle[last_idx],
            'kc_lower': kc.lower[last_idx],
            'atr': kc.atr[last_idx]
        }
    
    def check_squeeze(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check for Bollinger/Keltner squeeze condition.
        
        Squeeze occurs when Bollinger Bands are inside Keltner Channels.
        This indicates low volatility that often precedes significant moves.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (is_squeeze, squeeze_intensity)
        """
        bb = self._calculate_bollinger(data)
        kc = self._calculate_keltner(data)
        
        last_idx = len(data) - 1
        
        bb_width = bb.upper[last_idx] - bb.lower[last_idx]
        kc_width = kc.upper[last_idx] - kc.lower[last_idx]
        
        is_squeeze = (
            bb.lower[last_idx] > kc.lower[last_idx] and
            bb.upper[last_idx] < kc.upper[last_idx]
        )
        
        # Intensity: how tight is the squeeze
        if kc_width > 0:
            squeeze_intensity = 1 - (bb_width / kc_width)
        else:
            squeeze_intensity = 0.0
        
        return (is_squeeze, squeeze_intensity)
    
    def validate_exhaustion(
        self,
        data: pd.DataFrame,
        direction: SignalType
    ) -> Tuple[bool, str]:
        """
        Comprehensive exhaustion validation.
        
        Validates that price has reached an exhaustion point
        suitable for reversal entry.
        
        Args:
            data: OHLCV DataFrame
            direction: Expected trade direction
            
        Returns:
            Tuple of (is_exhausted, description)
        """
        bb_touch = self.check_bollinger_touch(data, direction)
        kc_touch = self.check_keltner_touch(data, direction)
        is_squeeze, squeeze_intensity = self.check_squeeze(data)
        
        reasons = []
        
        if bb_touch:
            reasons.append("BB boundary touch")
        if kc_touch:
            reasons.append("KC boundary touch")
        if is_squeeze and squeeze_intensity > 0.5:
            reasons.append(f"Squeeze ({squeeze_intensity:.2f})")
        
        is_exhausted = bb_touch or kc_touch
        
        description = ", ".join(reasons) if reasons else "No exhaustion signals"
        
        return (is_exhausted, description)


class ConfluenceIntegration:
    """
    Integration helper for Confluence Filter with ICT Engine.
    
    Example Usage:
    ```python
    ict_engine = ICTEngine(config)
    confluence_filter = ConfluenceFilter()
    integration = ConfluenceIntegration(ict_engine, confluence_filter)
    
    # Generate and validate setup
    setup = integration.generate_validated_setup(
        pair, h1_data, m15_data, m1_data
    )
    ```
    """
    
    def __init__(self, ict_engine, confluence_filter: ConfluenceFilter):
        self.ict_engine = ict_engine
        self.confluence_filter = confluence_filter
    
    def generate_validated_setup(
        self,
        pair: str,
        h1_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        m1_data: pd.DataFrame
    ) -> Optional[TradeSetup]:
        """
        Generate ICT setup with confluence validation.
        
        Process:
        1. Generate ICT setup (80% weight)
        2. Validate with confluence (20% weight)
        3. Return only if both criteria met
        
        Args:
            pair: Currency pair
            h1_data: 1-Hour data
            m15_data: 15-Minute data
            m1_data: 1-Minute data
            
        Returns:
            Validated TradeSetup or None
        """
        # Step 1: ICT Setup
        setup = self.ict_engine.generate_setup(pair, h1_data, m15_data, m1_data)
        
        if setup is None:
            return None
        
        # Step 2: Confluence Validation
        bb_touch = self.confluence_filter.check_bollinger_touch(m15_data, setup.signal)
        kc_touch = self.confluence_filter.check_keltner_touch(m15_data, setup.signal)
        
        setup.bollinger_touch = bb_touch
        setup.keltner_touch = kc_touch
        
        # Require at least one confluence
        if not bb_touch and not kc_touch:
            setup.is_valid = False
            setup.invalidation_reasons.append("No confluence filter triggered")
            return setup
        
        # Calculate confluence score
        setup.confluence_score = self.confluence_filter.calculate_confluence_score(setup)
        
        return setup
