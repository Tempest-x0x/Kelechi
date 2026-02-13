"""SIGHT Liquidity Analyzer - Pool identification and sweep detection."""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..core.base import LiquidityMapper
from ..core.types import (
    LiquidityPool, LiquidityType, LiquiditySweep, OHLCV, SignalType
)
from ..core.constants import SWING_LOOKBACK_CANDLES, DEFAULT_SWEEP_DEPTH_ATR


class LiquidityAnalyzer(LiquidityMapper):
    """
    Liquidity pool identification and sweep detection.
    
    Core responsibilities:
    1. Identify session highs/lows as liquidity pools
    2. Detect 48-candle swing highs/lows
    3. Find equal highs/lows (double tops/bottoms)
    4. Detect liquidity sweeps with rejection confirmation
    
    Liquidity Concept:
    - Resting orders accumulate above swing highs (buy stops)
    - Resting orders accumulate below swing lows (sell stops)
    - Smart money sweeps these pools before reversing
    """
    
    def __init__(
        self,
        default_lookback: int = SWING_LOOKBACK_CANDLES,
        sweep_depth_atr: float = DEFAULT_SWEEP_DEPTH_ATR
    ):
        super().__init__("LiquidityAnalyzer")
        self.default_lookback = default_lookback
        self.sweep_depth_atr = sweep_depth_atr
    
    def identify_pools(
        self,
        data: pd.DataFrame,
        lookback: int = None
    ) -> List[LiquidityPool]:
        """
        Identify liquidity pools from price data.
        
        Types of pools:
        1. Session highs/lows (previous day's H/L)
        2. Swing highs/lows (48-candle lookback)
        3. Equal highs/lows (double/triple tops/bottoms)
        4. Relative highs/lows (weekly/monthly)
        
        Args:
            data: OHLCV DataFrame
            lookback: Candle lookback for swing detection
            
        Returns:
            List of LiquidityPool objects
        """
        lookback = lookback or self.default_lookback
        pools = []
        
        # Detect swing-based pools
        swing_pools = self._detect_swing_pools(data, lookback)
        pools.extend(swing_pools)
        
        # Detect equal highs/lows
        equal_pools = self._detect_equal_levels(data)
        pools.extend(equal_pools)
        
        # Detect session highs/lows
        session_pools = self._detect_session_pools(data)
        pools.extend(session_pools)
        
        # Remove duplicates (pools at same price level)
        pools = self._merge_nearby_pools(pools, threshold=0.0001)
        
        # Sort by timestamp
        pools.sort(key=lambda x: x.index)
        
        return pools
    
    def _detect_swing_pools(
        self,
        data: pd.DataFrame,
        lookback: int
    ) -> List[LiquidityPool]:
        """
        Detect swing high/low based liquidity pools.
        
        Algorithm:
        - For each candle, check if it's the highest/lowest in lookback window
        - Mark as liquidity pool if it represents accumulated orders
        """
        pools = []
        highs = data['high'].values
        lows = data['low'].values
        n = len(data)
        
        # Rolling max/min for efficient detection
        rolling_high = pd.Series(highs).rolling(lookback, center=True).max().values
        rolling_low = pd.Series(lows).rolling(lookback, center=True).min().values
        
        for i in range(lookback // 2, n - lookback // 2):
            # Swing high = potential sell-side liquidity (buy stops above)
            if highs[i] == rolling_high[i]:
                pools.append(LiquidityPool(
                    type=LiquidityType.SWING_HIGH,
                    price=highs[i],
                    timestamp=data.index[i],
                    index=i,
                    strength=1.0
                ))
            
            # Swing low = potential buy-side liquidity (sell stops below)
            if lows[i] == rolling_low[i]:
                pools.append(LiquidityPool(
                    type=LiquidityType.SWING_LOW,
                    price=lows[i],
                    timestamp=data.index[i],
                    index=i,
                    strength=1.0
                ))
        
        return pools
    
    def _detect_equal_levels(
        self,
        data: pd.DataFrame,
        tolerance: float = 0.0002
    ) -> List[LiquidityPool]:
        """
        Detect equal highs/lows (double tops/bottoms).
        
        Equal levels represent strong liquidity pools because:
        - Market failed to break the level multiple times
        - More orders accumulate as level is retested
        - Eventual break leads to significant momentum
        
        Args:
            data: OHLCV DataFrame
            tolerance: Price tolerance for "equal" levels
        """
        pools = []
        highs = data['high'].values
        lows = data['low'].values
        n = len(data)
        
        # Find equal highs
        for i in range(n):
            for j in range(i + 10, min(i + 100, n)):  # Look 10-100 candles ahead
                if abs(highs[i] - highs[j]) < tolerance * highs[i]:
                    # Found equal highs
                    avg_price = (highs[i] + highs[j]) / 2
                    pools.append(LiquidityPool(
                        type=LiquidityType.EQUAL_HIGH,
                        price=avg_price,
                        timestamp=data.index[j],
                        index=j,
                        strength=2.0  # Equal levels are stronger
                    ))
                    break
        
        # Find equal lows
        for i in range(n):
            for j in range(i + 10, min(i + 100, n)):
                if abs(lows[i] - lows[j]) < tolerance * lows[i]:
                    avg_price = (lows[i] + lows[j]) / 2
                    pools.append(LiquidityPool(
                        type=LiquidityType.EQUAL_LOW,
                        price=avg_price,
                        timestamp=data.index[j],
                        index=j,
                        strength=2.0
                    ))
                    break
        
        return pools
    
    def _detect_session_pools(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """
        Detect previous session highs/lows as liquidity pools.
        
        Session boundaries:
        - Asia: 00:00-08:00 UTC
        - London: 08:00-16:00 UTC
        - New York: 13:00-21:00 UTC
        """
        pools = []
        
        # Group by date
        data_with_date = data.copy()
        data_with_date['date'] = data_with_date.index.date
        
        daily_groups = data_with_date.groupby('date')
        
        dates = list(daily_groups.groups.keys())
        
        for i, date in enumerate(dates[1:], 1):  # Start from second day
            prev_date = dates[i - 1]
            prev_day_data = daily_groups.get_group(prev_date)
            
            if len(prev_day_data) == 0:
                continue
            
            session_high = prev_day_data['high'].max()
            session_low = prev_day_data['low'].min()
            
            high_idx = prev_day_data['high'].idxmax()
            low_idx = prev_day_data['low'].idxmin()
            
            # Previous day high
            pools.append(LiquidityPool(
                type=LiquidityType.SESSION_HIGH,
                price=session_high,
                timestamp=high_idx,
                index=data.index.get_loc(high_idx),
                strength=1.5  # Session levels are significant
            ))
            
            # Previous day low
            pools.append(LiquidityPool(
                type=LiquidityType.SESSION_LOW,
                price=session_low,
                timestamp=low_idx,
                index=data.index.get_loc(low_idx),
                strength=1.5
            ))
        
        return pools
    
    def _merge_nearby_pools(
        self,
        pools: List[LiquidityPool],
        threshold: float
    ) -> List[LiquidityPool]:
        """Merge pools at similar price levels, keeping the strongest."""
        if not pools:
            return pools
        
        # Sort by price
        sorted_pools = sorted(pools, key=lambda x: x.price)
        merged = [sorted_pools[0]]
        
        for pool in sorted_pools[1:]:
            last = merged[-1]
            if abs(pool.price - last.price) < threshold * last.price:
                # Keep the stronger pool
                if pool.strength > last.strength:
                    merged[-1] = pool
            else:
                merged.append(pool)
        
        return merged
    
    def detect_sweep(
        self,
        data: pd.DataFrame,
        pools: List[LiquidityPool],
        current_idx: int,
        atr: Optional[np.ndarray] = None
    ) -> Optional[LiquiditySweep]:
        """
        Detect liquidity sweep events.
        
        Sweep Criteria:
        1. Price pierces through liquidity level (wick)
        2. Price closes back inside the level (rejection)
        3. Sweep depth meets minimum threshold
        
        For SHORT setup (sweep above pool):
        - High > pool price (pierces)
        - Close < pool price (rejects)
        
        For LONG setup (sweep below pool):
        - Low < pool price (pierces)
        - Close > pool price (rejects)
        
        Args:
            data: OHLCV DataFrame
            pools: List of liquidity pools to check
            current_idx: Current candle index
            atr: ATR values array
            
        Returns:
            LiquiditySweep if detected, None otherwise
        """
        if current_idx < 1 or current_idx >= len(data):
            return None
        
        candle = data.iloc[current_idx]
        
        # Calculate minimum sweep depth
        if atr is not None and current_idx < len(atr):
            min_depth = self.sweep_depth_atr * atr[current_idx]
        else:
            avg_range = (data['high'] - data['low']).rolling(14).mean()
            min_depth = self.sweep_depth_atr * avg_range.iloc[current_idx] if current_idx < len(avg_range) else 0
        
        # Check each unswept pool
        for pool in pools:
            if pool.swept:
                continue
            
            if pool.index >= current_idx:
                continue
            
            # Check for HIGH sweep (bearish setup)
            if pool.type in [LiquidityType.SWING_HIGH, LiquidityType.SESSION_HIGH, LiquidityType.EQUAL_HIGH]:
                if candle['high'] > pool.price and candle['close'] < pool.price:
                    sweep_depth = candle['high'] - pool.price
                    
                    if sweep_depth >= min_depth:
                        ohlcv = OHLCV(
                            timestamp=data.index[current_idx],
                            open=candle['open'],
                            high=candle['high'],
                            low=candle['low'],
                            close=candle['close'],
                            volume=candle['volume']
                        )
                        
                        return LiquiditySweep(
                            pool=pool,
                            sweep_candle=ohlcv,
                            sweep_timestamp=data.index[current_idx],
                            sweep_index=current_idx,
                            rejection=True,
                            sweep_depth_pips=sweep_depth
                        )
            
            # Check for LOW sweep (bullish setup)
            if pool.type in [LiquidityType.SWING_LOW, LiquidityType.SESSION_LOW, LiquidityType.EQUAL_LOW]:
                if candle['low'] < pool.price and candle['close'] > pool.price:
                    sweep_depth = pool.price - candle['low']
                    
                    if sweep_depth >= min_depth:
                        ohlcv = OHLCV(
                            timestamp=data.index[current_idx],
                            open=candle['open'],
                            high=candle['high'],
                            low=candle['low'],
                            close=candle['close'],
                            volume=candle['volume']
                        )
                        
                        return LiquiditySweep(
                            pool=pool,
                            sweep_candle=ohlcv,
                            sweep_timestamp=data.index[current_idx],
                            sweep_index=current_idx,
                            rejection=True,
                            sweep_depth_pips=sweep_depth
                        )
        
        return None
    
    def get_session_levels(
        self,
        data: pd.DataFrame,
        session: str
    ) -> Tuple[float, float]:
        """
        Get previous session's high and low.
        
        Args:
            data: OHLCV DataFrame
            session: Session name ('ASIA', 'LONDON', 'NEW_YORK')
            
        Returns:
            Tuple of (session_high, session_low)
        """
        session_hours = {
            'ASIA': (0, 8),
            'LONDON': (8, 16),
            'NEW_YORK': (13, 21)
        }
        
        if session not in session_hours:
            return (np.nan, np.nan)
        
        start_hour, end_hour = session_hours[session]
        
        # Get the previous session's data
        current_time = data.index[-1]
        prev_day = current_time - timedelta(days=1)
        
        # Filter to session hours
        mask = (
            (data.index.date == prev_day.date()) &
            (data.index.hour >= start_hour) &
            (data.index.hour < end_hour)
        )
        
        session_data = data[mask]
        
        if len(session_data) == 0:
            return (np.nan, np.nan)
        
        return (session_data['high'].max(), session_data['low'].min())
    
    def get_nearest_pool(
        self,
        pools: List[LiquidityPool],
        current_price: float,
        direction: SignalType
    ) -> Optional[LiquidityPool]:
        """
        Get the nearest unswept liquidity pool in the given direction.
        
        Args:
            pools: List of liquidity pools
            current_price: Current market price
            direction: Trade direction to find target
            
        Returns:
            Nearest liquidity pool or None
        """
        unswept = [p for p in pools if not p.swept]
        
        if direction == SignalType.LONG:
            # Look for pools above current price (targets)
            above = [p for p in unswept if p.price > current_price and 
                     p.type in [LiquidityType.SWING_HIGH, LiquidityType.SESSION_HIGH, LiquidityType.EQUAL_HIGH]]
            if above:
                return min(above, key=lambda p: p.price - current_price)
        
        elif direction == SignalType.SHORT:
            # Look for pools below current price (targets)
            below = [p for p in unswept if p.price < current_price and
                     p.type in [LiquidityType.SWING_LOW, LiquidityType.SESSION_LOW, LiquidityType.EQUAL_LOW]]
            if below:
                return min(below, key=lambda p: current_price - p.price)
        
        return None
    
    def mark_pool_swept(self, pools: List[LiquidityPool], pool_to_mark: LiquidityPool) -> None:
        """Mark a pool as swept in the pool list."""
        for i, pool in enumerate(pools):
            if pool.price == pool_to_mark.price and pool.type == pool_to_mark.type:
                pools[i].swept = True
                pools[i].sweep_timestamp = pool_to_mark.sweep_timestamp
                break
