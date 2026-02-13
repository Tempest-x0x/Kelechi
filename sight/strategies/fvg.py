"""SIGHT Fair Value Gap Detector - FVG identification and entry logic."""
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import GapAnalyzer
from ..core.types import FairValueGap, FVGType, SignalType
from ..core.constants import DEFAULT_FVG_MIN_SIZE_ATR, FVG_ENTRY_OFFSETS


class FVGDetector(GapAnalyzer):
    """
    Fair Value Gap (FVG) detection and entry management.
    
    FVG Definition:
    - A price imbalance created by aggressive momentum
    - Gap between candle 1's high/low and candle 3's low/high
    - Candle 2 creates the displacement
    
    BULLISH FVG:
    - Gap between candle 1's high and candle 3's low
    - Created by strong bullish momentum
    - Entry: Expect price to retrace into gap
    
    BEARISH FVG:
    - Gap between candle 1's low and candle 3's high  
    - Created by strong bearish momentum
    - Entry: Expect price to retrace into gap
    """
    
    def __init__(
        self,
        min_size_atr: float = DEFAULT_FVG_MIN_SIZE_ATR,
        entry_offset: float = 0.5
    ):
        super().__init__("FVGDetector")
        self.min_size_atr = min_size_atr
        self.entry_offset = entry_offset
    
    def detect_fvg(
        self,
        data: pd.DataFrame,
        min_size_atr: float = None,
        atr: Optional[np.ndarray] = None
    ) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price data.
        
        FVG Detection Algorithm:
        
        For BULLISH FVG (3-candle pattern):
        ```
        Candle 1: Reference candle
        Candle 2: Displacement candle (strong bullish)
        Candle 3: Continuation candle
        
        Gap exists if: candle_3.low > candle_1.high
        FVG zone: candle_1.high to candle_3.low
        ```
        
        For BEARISH FVG:
        ```
        Gap exists if: candle_3.high < candle_1.low
        FVG zone: candle_3.high to candle_1.low
        ```
        
        Args:
            data: OHLCV DataFrame
            min_size_atr: Minimum FVG size in ATR multiples
            atr: Pre-calculated ATR array
            
        Returns:
            List of FairValueGap objects
        """
        min_size_atr = min_size_atr or self.min_size_atr
        fvgs = []
        
        # Calculate ATR if not provided
        if atr is None:
            atr = self._calculate_atr(data)
        
        n = len(data)
        
        for i in range(2, n):
            candle_1 = data.iloc[i - 2]
            candle_2 = data.iloc[i - 1]
            candle_3 = data.iloc[i]
            
            current_atr = atr[i] if i < len(atr) and not np.isnan(atr[i]) else 0.001
            min_gap_size = min_size_atr * current_atr
            
            # Check for BULLISH FVG
            # Condition: Candle 3's low is higher than Candle 1's high
            if candle_3['low'] > candle_1['high']:
                gap_low = candle_1['high']
                gap_high = candle_3['low']
                gap_size = gap_high - gap_low
                
                if gap_size >= min_gap_size:
                    # Validate displacement (candle 2 should be strong bullish)
                    candle_2_body = candle_2['close'] - candle_2['open']
                    candle_2_range = candle_2['high'] - candle_2['low']
                    
                    if candle_2_body > 0 and candle_2_range > 0:
                        body_ratio = candle_2_body / candle_2_range
                        
                        if body_ratio >= 0.5:  # At least 50% body
                            fvg = FairValueGap(
                                type=FVGType.BULLISH,
                                high=gap_high,
                                low=gap_low,
                                midpoint=(gap_high + gap_low) / 2,
                                timestamp=data.index[i - 1],  # Displacement candle time
                                index=i - 1,
                                filled=False,
                                fill_percent=0.0,
                                creation_candle_size=candle_2_range
                            )
                            fvgs.append(fvg)
            
            # Check for BEARISH FVG
            # Condition: Candle 3's high is lower than Candle 1's low
            if candle_3['high'] < candle_1['low']:
                gap_high = candle_1['low']
                gap_low = candle_3['high']
                gap_size = gap_high - gap_low
                
                if gap_size >= min_gap_size:
                    # Validate displacement (candle 2 should be strong bearish)
                    candle_2_body = candle_2['open'] - candle_2['close']
                    candle_2_range = candle_2['high'] - candle_2['low']
                    
                    if candle_2_body > 0 and candle_2_range > 0:
                        body_ratio = candle_2_body / candle_2_range
                        
                        if body_ratio >= 0.5:
                            fvg = FairValueGap(
                                type=FVGType.BEARISH,
                                high=gap_high,
                                low=gap_low,
                                midpoint=(gap_high + gap_low) / 2,
                                timestamp=data.index[i - 1],
                                index=i - 1,
                                filled=False,
                                fill_percent=0.0,
                                creation_candle_size=candle_2_range
                            )
                            fvgs.append(fvg)
        
        return fvgs
    
    def check_fvg_fill(
        self,
        fvg: FairValueGap,
        current_price: float
    ) -> float:
        """
        Calculate FVG fill percentage based on current price.
        
        Fill Calculation:
        - 0%: Price hasn't entered the gap
        - 50%: Price at midpoint
        - 100%: Price has completely filled the gap
        
        Args:
            fvg: FairValueGap to check
            current_price: Current market price
            
        Returns:
            Fill percentage (0.0 to 1.0+)
        """
        if fvg.type == FVGType.BULLISH:
            # For bullish FVG, price retracing down fills it
            if current_price >= fvg.high:
                return 0.0  # Not touched yet
            elif current_price <= fvg.low:
                return 1.0  # Fully filled
            else:
                # Partial fill
                return (fvg.high - current_price) / fvg.size
        
        else:  # BEARISH FVG
            # For bearish FVG, price retracing up fills it
            if current_price <= fvg.low:
                return 0.0
            elif current_price >= fvg.high:
                return 1.0
            else:
                return (current_price - fvg.low) / fvg.size
    
    def update_fvg_status(
        self,
        fvgs: List[FairValueGap],
        data: pd.DataFrame,
        start_idx: int = 0
    ) -> List[FairValueGap]:
        """
        Update fill status of FVGs based on price action.
        
        Args:
            fvgs: List of FVGs to update
            data: OHLCV data to check against
            start_idx: Starting index in data
            
        Returns:
            Updated list of FVGs
        """
        for fvg in fvgs:
            if fvg.filled:
                continue
            
            # Check candles after FVG formation
            check_start = max(fvg.index + 2, start_idx)
            
            for i in range(check_start, len(data)):
                candle = data.iloc[i]
                
                if fvg.type == FVGType.BULLISH:
                    # Check if price touched the FVG zone
                    if candle['low'] <= fvg.high:
                        fill_pct = self.check_fvg_fill(fvg, candle['low'])
                        fvg.fill_percent = max(fvg.fill_percent, fill_pct)
                        
                        if candle['low'] <= fvg.low:
                            fvg.filled = True
                            break
                
                else:  # BEARISH
                    if candle['high'] >= fvg.low:
                        fill_pct = self.check_fvg_fill(fvg, candle['high'])
                        fvg.fill_percent = max(fvg.fill_percent, fill_pct)
                        
                        if candle['high'] >= fvg.high:
                            fvg.filled = True
                            break
        
        return fvgs
    
    def get_valid_entry_fvg(
        self,
        fvgs: List[FairValueGap],
        direction: SignalType,
        current_price: float,
        max_distance_atr: float = 3.0,
        current_atr: float = None
    ) -> Optional[FairValueGap]:
        """
        Get the nearest valid FVG for entry.
        
        Valid FVG Criteria:
        1. Not fully filled (fill_percent < 1.0)
        2. Correct direction (BULLISH for LONG, BEARISH for SHORT)
        3. Within reasonable distance from current price
        4. Created recently (not stale)
        
        Args:
            fvgs: List of detected FVGs
            direction: Trade direction
            current_price: Current market price
            max_distance_atr: Maximum distance in ATR
            current_atr: Current ATR value
            
        Returns:
            Best FairValueGap for entry or None
        """
        if direction == SignalType.LONG:
            valid_type = FVGType.BULLISH
        elif direction == SignalType.SHORT:
            valid_type = FVGType.BEARISH
        else:
            return None
        
        candidates = []
        
        for fvg in fvgs:
            # Check type and fill status
            if fvg.type != valid_type:
                continue
            
            if fvg.filled or fvg.fill_percent >= 0.9:
                continue
            
            # Check distance
            if valid_type == FVGType.BULLISH:
                # For longs, FVG should be below current price
                if fvg.high > current_price:
                    continue
                distance = current_price - fvg.midpoint
            else:
                # For shorts, FVG should be above current price
                if fvg.low < current_price:
                    continue
                distance = fvg.midpoint - current_price
            
            # Check max distance
            if current_atr and distance > max_distance_atr * current_atr:
                continue
            
            candidates.append((fvg, distance))
        
        if not candidates:
            return None
        
        # Return the nearest FVG
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def get_entry_price(
        self,
        fvg: FairValueGap,
        offset: float = None
    ) -> float:
        """
        Calculate entry price within FVG.
        
        Entry Offset Options:
        - 0.0 (0%): Entry at gap boundary (conservative)
        - 0.25 (25%): Entry at 25% into the gap
        - 0.5 (50%): Entry at midpoint (balanced)
        - 1.0 (100%): Entry at full fill (aggressive)
        
        Args:
            fvg: FairValueGap to enter
            offset: Entry offset (0-1, default from config)
            
        Returns:
            Entry price
        """
        offset = offset if offset is not None else self.entry_offset
        return fvg.get_entry_price(offset)
    
    def get_stop_loss_price(
        self,
        fvg: FairValueGap,
        buffer_atr: float = 0.2,
        current_atr: float = 0.001
    ) -> float:
        """
        Calculate stop loss price based on FVG invalidation.
        
        Stop Loss Logic:
        - BULLISH FVG: Stop below the gap low
        - BEARISH FVG: Stop above the gap high
        
        Args:
            fvg: FairValueGap being traded
            buffer_atr: Buffer in ATR beyond the gap
            current_atr: Current ATR value
            
        Returns:
            Stop loss price
        """
        buffer = buffer_atr * current_atr
        
        if fvg.type == FVGType.BULLISH:
            return fvg.low - buffer
        else:
            return fvg.high + buffer
    
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
        
        atr = pd.Series(tr).rolling(period).mean().values
        
        return atr
    
    def get_fvg_clusters(
        self,
        fvgs: List[FairValueGap],
        cluster_threshold: float = 0.001
    ) -> List[List[FairValueGap]]:
        """
        Group nearby FVGs into clusters (stronger zones).
        
        Clustered FVGs indicate:
        - Strong institutional interest
        - Higher probability reversal zones
        - Deeper potential retracements
        
        Args:
            fvgs: List of FVGs to cluster
            cluster_threshold: Price threshold for clustering
            
        Returns:
            List of FVG clusters
        """
        if not fvgs:
            return []
        
        # Sort by midpoint
        sorted_fvgs = sorted(fvgs, key=lambda x: x.midpoint)
        
        clusters = [[sorted_fvgs[0]]]
        
        for fvg in sorted_fvgs[1:]:
            last_cluster = clusters[-1]
            last_fvg = last_cluster[-1]
            
            # Check if FVG is close to last cluster
            if abs(fvg.midpoint - last_fvg.midpoint) < cluster_threshold * last_fvg.midpoint:
                last_cluster.append(fvg)
            else:
                clusters.append([fvg])
        
        return [c for c in clusters if len(c) > 0]
