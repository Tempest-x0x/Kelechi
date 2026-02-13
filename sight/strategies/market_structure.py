"""SIGHT Market Structure Analyzer - HH/HL/LH/LL detection and bias determination."""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import StructureAnalyzer
from ..core.types import (
    Timeframe, MarketBias, MarketStructure, SwingPoint,
    DisplacementEvent, SignalType
)
from ..core.constants import DEFAULT_SWING_STRENGTH, DEFAULT_DISPLACEMENT_ATR


class MarketStructureAnalyzer(StructureAnalyzer):
    """
    Market structure analysis for ICT-style trading.
    
    Core responsibilities:
    1. Detect swing highs and swing lows
    2. Classify HH/HL/LH/LL sequences
    3. Determine bullish/bearish bias
    4. Detect Market Structure Shifts (MSS)
    
    Algorithm for swing detection:
    - A swing high is confirmed when N candles on each side have lower highs
    - A swing low is confirmed when N candles on each side have higher lows
    - N is the 'strength' parameter (default: 3)
    """
    
    def __init__(self, default_strength: int = DEFAULT_SWING_STRENGTH):
        super().__init__("MarketStructureAnalyzer")
        self.default_strength = default_strength
    
    def detect_swings(
        self,
        data: pd.DataFrame,
        strength: int = None
    ) -> List[SwingPoint]:
        """
        Detect swing highs and swing lows in price data.
        
        Algorithm (Fractal-based):
        For each candle i:
          - Swing High: high[i] > high[i-n:i] AND high[i] > high[i+1:i+n+1]
          - Swing Low: low[i] < low[i-n:i] AND low[i] < low[i+1:i+n+1]
        
        Args:
            data: OHLCV DataFrame with datetime index
            strength: Number of candles on each side (default: 3)
            
        Returns:
            List of SwingPoint objects sorted by timestamp
        """
        strength = strength or self.default_strength
        swings = []
        
        highs = data['high'].values
        lows = data['low'].values
        n = len(data)
        
        # Vectorized swing detection
        for i in range(strength, n - strength):
            # Check swing high
            left_highs = highs[i - strength:i]
            right_highs = highs[i + 1:i + strength + 1]
            
            if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
                swings.append(SwingPoint(
                    timestamp=data.index[i],
                    price=highs[i],
                    index=i,
                    is_high=True,
                    strength=strength
                ))
            
            # Check swing low
            left_lows = lows[i - strength:i]
            right_lows = lows[i + 1:i + strength + 1]
            
            if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
                swings.append(SwingPoint(
                    timestamp=data.index[i],
                    price=lows[i],
                    index=i,
                    is_high=False,
                    strength=strength
                ))
        
        # Sort by index (chronological order)
        swings.sort(key=lambda x: x.index)
        
        return swings
    
    def classify_swing_sequence(
        self,
        swings: List[SwingPoint]
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Classify swings into HH/HL and LH/LL sequences.
        
        Rules:
        - Higher High (HH): Swing high > previous swing high
        - Higher Low (HL): Swing low > previous swing low
        - Lower High (LH): Swing high < previous swing high
        - Lower Low (LL): Swing low < previous swing low
        
        Args:
            swings: List of SwingPoints in chronological order
            
        Returns:
            Tuple of (bullish_sequence, bearish_sequence)
        """
        if len(swings) < 2:
            return ([], [])
        
        # Separate highs and lows
        swing_highs = [s for s in swings if s.is_high]
        swing_lows = [s for s in swings if not s.is_high]
        
        # Track HH/HL for bullish
        bullish_sequence = []
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price > swing_highs[i-1].price:
                bullish_sequence.append(swing_highs[i])  # HH
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price > swing_lows[i-1].price:
                bullish_sequence.append(swing_lows[i])  # HL
        
        # Track LH/LL for bearish
        bearish_sequence = []
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price < swing_highs[i-1].price:
                bearish_sequence.append(swing_highs[i])  # LH
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price < swing_lows[i-1].price:
                bearish_sequence.append(swing_lows[i])  # LL
        
        return (
            sorted(bullish_sequence, key=lambda x: x.index),
            sorted(bearish_sequence, key=lambda x: x.index)
        )
    
    def determine_bias(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe
    ) -> MarketStructure:
        """
        Determine current market bias from structure analysis.
        
        Algorithm:
        1. Detect all swing points
        2. Analyze the last N swings
        3. Count HH/HL vs LH/LL patterns
        4. Determine bias based on dominant pattern
        5. Calculate confidence score
        
        Bias Rules:
        - BULLISH: Sequence of HH + HL (protected lows, expanding highs)
        - BEARISH: Sequence of LH + LL (protected highs, expanding lows)
        - NEUTRAL: Mixed structure or ranging
        
        Args:
            data: OHLCV DataFrame
            timeframe: Timeframe for context
            
        Returns:
            MarketStructure object with bias and swing points
        """
        swings = self.detect_swings(data)
        
        if len(swings) < 4:
            return MarketStructure(
                bias=MarketBias.UNDEFINED,
                swing_points=swings,
                confidence_score=0.0
            )
        
        # Get last N swing points for analysis
        lookback = min(10, len(swings))
        recent_swings = swings[-lookback:]
        
        # Separate into highs and lows
        recent_highs = [s for s in recent_swings if s.is_high]
        recent_lows = [s for s in recent_swings if not s.is_high]
        
        # Count patterns
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0
        
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i-1].price:
                hh_count += 1
            else:
                lh_count += 1
        
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i-1].price:
                hl_count += 1
            else:
                ll_count += 1
        
        # Determine bias
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count
        total_patterns = bullish_score + bearish_score
        
        if total_patterns == 0:
            bias = MarketBias.NEUTRAL
            confidence = 0.0
        elif bullish_score > bearish_score * 1.5:
            bias = MarketBias.BULLISH
            confidence = bullish_score / total_patterns
        elif bearish_score > bullish_score * 1.5:
            bias = MarketBias.BEARISH
            confidence = bearish_score / total_patterns
        else:
            bias = MarketBias.NEUTRAL
            confidence = 0.5
        
        # Build structure object
        structure = MarketStructure(
            bias=bias,
            swing_points=swings,
            confidence_score=confidence
        )
        
        # Assign latest swing points
        if recent_highs:
            if bias == MarketBias.BULLISH and len(recent_highs) >= 1:
                structure.last_higher_high = recent_highs[-1]
            elif bias == MarketBias.BEARISH and len(recent_highs) >= 1:
                structure.last_lower_high = recent_highs[-1]
        
        if recent_lows:
            if bias == MarketBias.BULLISH and len(recent_lows) >= 1:
                structure.last_higher_low = recent_lows[-1]
            elif bias == MarketBias.BEARISH and len(recent_lows) >= 1:
                structure.last_lower_low = recent_lows[-1]
        
        # Set structure break price (key level to watch)
        if bias == MarketBias.BULLISH and recent_lows:
            structure.structure_break_price = recent_lows[-1].price
        elif bias == MarketBias.BEARISH and recent_highs:
            structure.structure_break_price = recent_highs[-1].price
        
        return structure
    
    def detect_structure_shift(
        self,
        data: pd.DataFrame,
        current_structure: MarketStructure,
        atr: Optional[np.ndarray] = None,
        displacement_threshold: float = DEFAULT_DISPLACEMENT_ATR
    ) -> Optional[DisplacementEvent]:
        """
        Detect Market Structure Shift (MSS) - aggressive displacement through structure.
        
        MSS Criteria:
        1. Price breaks through the structure break level
        2. Displacement is aggressive (body > threshold * ATR)
        3. Candle closes through the level (not just wick)
        
        For BULLISH MSS (in bearish structure):
        - Price breaks above last lower high with strong bullish candle
        
        For BEARISH MSS (in bullish structure):
        - Price breaks below last higher low with strong bearish candle
        
        Args:
            data: Recent OHLCV data (last 10-20 candles)
            current_structure: Current market structure
            atr: ATR values array (optional)
            displacement_threshold: Minimum displacement in ATR multiples
            
        Returns:
            DisplacementEvent if MSS detected, None otherwise
        """
        if len(data) < 3:
            return None
        
        # Calculate ATR if not provided
        if atr is None:
            atr = self._calculate_atr(data, period=14)
        
        current_atr = atr[-1] if len(atr) > 0 else data['high'].iloc[-1] - data['low'].iloc[-1]
        min_displacement = displacement_threshold * current_atr
        
        # Get the last few candles for MSS detection
        for i in range(-3, 0):
            if abs(i) > len(data):
                continue
                
            candle_idx = len(data) + i
            row = data.iloc[i]
            
            body_size = abs(row['close'] - row['open'])
            is_bullish = row['close'] > row['open']
            is_bearish = row['close'] < row['open']
            
            # Check for BULLISH MSS
            if (current_structure.bias == MarketBias.BEARISH and 
                current_structure.last_lower_high is not None):
                
                break_level = current_structure.last_lower_high.price
                
                if (is_bullish and 
                    row['close'] > break_level and
                    body_size >= min_displacement):
                    
                    return DisplacementEvent(
                        timestamp=data.index[i],
                        index=candle_idx,
                        direction=SignalType.LONG,
                        displacement_size=body_size / current_atr,
                        candle_count=1,
                        closes_through_level=True,
                        level_broken=break_level
                    )
            
            # Check for BEARISH MSS
            if (current_structure.bias == MarketBias.BULLISH and
                current_structure.last_higher_low is not None):
                
                break_level = current_structure.last_higher_low.price
                
                if (is_bearish and
                    row['close'] < break_level and
                    body_size >= min_displacement):
                    
                    return DisplacementEvent(
                        timestamp=data.index[i],
                        index=candle_idx,
                        direction=SignalType.SHORT,
                        displacement_size=body_size / current_atr,
                        candle_count=1,
                        closes_through_level=True,
                        level_broken=break_level
                    )
        
        return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Simple moving average for ATR
        atr = np.zeros(len(tr))
        atr[:period] = np.nan
        
        for i in range(period, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        return atr
    
    def get_key_levels(
        self,
        structure: MarketStructure,
        n_levels: int = 4
    ) -> List[Tuple[float, str]]:
        """
        Extract key structural levels for trading.
        
        Args:
            structure: MarketStructure object
            n_levels: Number of levels to return
            
        Returns:
            List of (price, description) tuples
        """
        levels = []
        
        if structure.last_higher_high:
            levels.append((structure.last_higher_high.price, "HH"))
        if structure.last_higher_low:
            levels.append((structure.last_higher_low.price, "HL"))
        if structure.last_lower_high:
            levels.append((structure.last_lower_high.price, "LH"))
        if structure.last_lower_low:
            levels.append((structure.last_lower_low.price, "LL"))
        
        # Add recent swing points
        for swing in structure.swing_points[-n_levels:]:
            label = "SH" if swing.is_high else "SL"
            if (swing.price, label) not in levels:
                levels.append((swing.price, label))
        
        return levels[:n_levels]
    
    def is_bias_aligned(
        self,
        htf_structure: MarketStructure,
        ltf_structure: MarketStructure
    ) -> Tuple[bool, str]:
        """
        Check if HTF and LTF biases are aligned for trade direction.
        
        Trading Rule:
        - Only trade in direction of both HTF (1H) and LTF (15m) bias
        
        Args:
            htf_structure: Higher timeframe structure (1H)
            ltf_structure: Lower timeframe structure (15m)
            
        Returns:
            Tuple of (is_aligned, allowed_direction)
        """
        htf_bias = htf_structure.bias
        ltf_bias = ltf_structure.bias
        
        if htf_bias == MarketBias.BULLISH and ltf_bias == MarketBias.BULLISH:
            return (True, "LONG")
        elif htf_bias == MarketBias.BEARISH and ltf_bias == MarketBias.BEARISH:
            return (True, "SHORT")
        else:
            return (False, "NONE")
