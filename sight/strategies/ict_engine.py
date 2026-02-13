"""SIGHT ICT Engine - Core ICT model implementation (80% decision weight)."""
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseLogger
from ..core.types import (
    Timeframe, MarketBias, SignalType, TradeSetup, PairConfig,
    MarketStructure, FairValueGap, LiquidityPool, LiquiditySweep,
    DisplacementEvent, FVGType
)
from ..core.constants import (
    DEFAULT_HTF_EMA, DEFAULT_DISPLACEMENT_ATR, DEFAULT_FVG_MIN_SIZE_ATR,
    SWING_LOOKBACK_CANDLES, DEFAULT_SWEEP_DEPTH_ATR
)
from .market_structure import MarketStructureAnalyzer
from .liquidity import LiquidityAnalyzer
from .fvg import FVGDetector


class ICTEngine(BaseLogger):
    """
    ICT Narrative Engine - Core decision-making system.
    
    This engine contributes 80% of the trade decision weight.
    
    ICT Model Sequence:
    1. HTF Bias Determination (1H + 15m structure alignment)
    2. Liquidity Pool Mapping (session H/L, 48-candle swings)
    3. Liquidity Sweep Detection (pierce + rejection)
    4. Market Structure Shift (MSS) Confirmation
    5. FVG Entry Zone Identification
    
    Trade only when ALL criteria align:
    - HTF bias is clear (bullish or bearish)
    - Liquidity has been swept
    - MSS confirms direction
    - Valid FVG exists for entry
    """
    
    def __init__(
        self,
        config: PairConfig = None,
        structure_analyzer: MarketStructureAnalyzer = None,
        liquidity_analyzer: LiquidityAnalyzer = None,
        fvg_detector: FVGDetector = None
    ):
        super().__init__("ICTEngine")
        
        self.config = config or self._default_config()
        self.structure_analyzer = structure_analyzer or MarketStructureAnalyzer()
        self.liquidity_analyzer = liquidity_analyzer or LiquidityAnalyzer()
        self.fvg_detector = fvg_detector or FVGDetector()
        
        # State tracking
        self._cached_structures: Dict[Timeframe, MarketStructure] = {}
        self._cached_pools: List[LiquidityPool] = []
        self._cached_fvgs: List[FairValueGap] = []
    
    def _default_config(self) -> PairConfig:
        """Create default configuration."""
        return PairConfig(
            pair="DEFAULT",
            sweep_depth_atr_multiple=DEFAULT_SWEEP_DEPTH_ATR,
            sweep_lookback_candles=SWING_LOOKBACK_CANDLES,
            displacement_threshold_atr=DEFAULT_DISPLACEMENT_ATR,
            fvg_min_size_atr=DEFAULT_FVG_MIN_SIZE_ATR,
            htf_ema_period=DEFAULT_HTF_EMA
        )
    
    def update_config(self, config: PairConfig) -> None:
        """Update engine configuration."""
        self.config = config
        self.log_info(f"Config updated for {config.pair}")
    
    def analyze_htf_bias(
        self,
        h1_data: pd.DataFrame,
        m15_data: pd.DataFrame
    ) -> Tuple[MarketBias, bool, str]:
        """
        Determine Higher Timeframe bias from 1H and 15m structure.
        
        HTF Bias Rules:
        - BULLISH: Both 1H and 15m show HH/HL structure
        - BEARISH: Both 1H and 15m show LH/LL structure
        - NEUTRAL: Conflicting or unclear structure
        
        Trading Rule:
        - Only trade in the direction of aligned HTF bias
        
        Args:
            h1_data: 1-Hour OHLCV data
            m15_data: 15-Minute OHLCV data
            
        Returns:
            Tuple of (bias, is_aligned, direction_string)
        """
        # Analyze 1H structure
        h1_structure = self.structure_analyzer.determine_bias(h1_data, Timeframe.H1)
        self._cached_structures[Timeframe.H1] = h1_structure
        
        # Analyze 15m structure
        m15_structure = self.structure_analyzer.determine_bias(m15_data, Timeframe.M15)
        self._cached_structures[Timeframe.M15] = m15_structure
        
        # Check alignment
        is_aligned, direction = self.structure_analyzer.is_bias_aligned(
            h1_structure, m15_structure
        )
        
        if is_aligned:
            if direction == "LONG":
                return (MarketBias.BULLISH, True, "LONG")
            else:
                return (MarketBias.BEARISH, True, "SHORT")
        
        return (MarketBias.NEUTRAL, False, "NONE")
    
    def map_liquidity(
        self,
        data: pd.DataFrame,
        lookback: int = None
    ) -> List[LiquidityPool]:
        """
        Identify liquidity pools in price data.
        
        Liquidity Types:
        1. Previous session highs/lows
        2. 48-candle swing highs/lows
        3. Equal highs/lows (double tops/bottoms)
        
        Args:
            data: OHLCV data for liquidity mapping
            lookback: Candle lookback (default: 48)
            
        Returns:
            List of identified liquidity pools
        """
        lookback = lookback or self.config.sweep_lookback_candles
        
        pools = self.liquidity_analyzer.identify_pools(data, lookback)
        self._cached_pools = pools
        
        self.log_debug(f"Mapped {len(pools)} liquidity pools")
        
        return pools
    
    def detect_liquidity_sweep(
        self,
        data: pd.DataFrame,
        current_idx: int,
        atr: np.ndarray = None
    ) -> Optional[LiquiditySweep]:
        """
        Detect liquidity sweep with rejection.
        
        Sweep Criteria (Step A):
        1. Price must pierce the liquidity level (wick through)
        2. Price must reject (close back inside the range)
        3. Sweep depth must meet minimum threshold
        
        Args:
            data: OHLCV data
            current_idx: Current candle index to check
            atr: ATR array for threshold calculation
            
        Returns:
            LiquiditySweep if detected, None otherwise
        """
        if not self._cached_pools:
            self.map_liquidity(data)
        
        sweep = self.liquidity_analyzer.detect_sweep(
            data, 
            self._cached_pools, 
            current_idx,
            atr
        )
        
        if sweep:
            self.log_info(f"Liquidity sweep detected at {sweep.pool.price:.5f}")
            # Mark pool as swept
            self.liquidity_analyzer.mark_pool_swept(self._cached_pools, sweep.pool)
        
        return sweep
    
    def detect_mss(
        self,
        data: pd.DataFrame,
        htf_structure: MarketStructure,
        atr: np.ndarray = None
    ) -> Optional[DisplacementEvent]:
        """
        Detect Market Structure Shift (MSS) - Step B.
        
        MSS Criteria:
        1. Aggressive displacement candle (body > threshold * ATR)
        2. Closes through key structural level
        3. Direction aligns with expected reversal
        
        For BULLISH MSS:
        - In bearish structure, price breaks above recent lower high
        - Strong bullish candle with displacement
        
        For BEARISH MSS:
        - In bullish structure, price breaks below recent higher low
        - Strong bearish candle with displacement
        
        Args:
            data: Recent 1m OHLCV data
            htf_structure: Current HTF market structure
            atr: ATR values array
            
        Returns:
            DisplacementEvent if MSS detected
        """
        mss = self.structure_analyzer.detect_structure_shift(
            data,
            htf_structure,
            atr,
            self.config.displacement_threshold_atr
        )
        
        if mss:
            self.log_info(f"MSS detected: {mss.direction.name} at {mss.level_broken:.5f}")
        
        return mss
    
    def find_entry_fvg(
        self,
        data: pd.DataFrame,
        direction: SignalType,
        current_price: float,
        atr: np.ndarray = None
    ) -> Optional[FairValueGap]:
        """
        Find valid Fair Value Gap for entry - Step C.
        
        FVG Entry Rules:
        1. FVG must be created by displacement move
        2. FVG direction must match trade direction
        3. FVG must not be fully filled
        4. Entry at configurable offset (25%/50%/100%)
        
        Args:
            data: OHLCV data for FVG detection
            direction: Trade direction
            current_price: Current market price
            atr: ATR values array
            
        Returns:
            Valid FairValueGap for entry or None
        """
        # Detect FVGs
        fvgs = self.fvg_detector.detect_fvg(
            data,
            self.config.fvg_min_size_atr,
            atr
        )
        
        # Update fill status
        self.fvg_detector.update_fvg_status(fvgs, data)
        self._cached_fvgs = fvgs
        
        # Find valid entry FVG
        current_atr = atr[-1] if atr is not None and len(atr) > 0 else None
        
        entry_fvg = self.fvg_detector.get_valid_entry_fvg(
            fvgs,
            direction,
            current_price,
            max_distance_atr=3.0,
            current_atr=current_atr
        )
        
        if entry_fvg:
            self.log_info(f"Entry FVG found: {entry_fvg.type.name} at {entry_fvg.midpoint:.5f}")
        
        return entry_fvg
    
    def generate_setup(
        self,
        pair: str,
        h1_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        m1_data: pd.DataFrame
    ) -> Optional[TradeSetup]:
        """
        Generate complete trade setup using ICT model.
        
        ICT Model Sequence:
        1. Analyze HTF bias (1H + 15m alignment)
        2. Map liquidity pools
        3. Detect liquidity sweep
        4. Confirm MSS (Market Structure Shift)
        5. Find FVG entry zone
        6. Calculate risk parameters
        
        Args:
            pair: Currency pair
            h1_data: 1-Hour OHLCV data
            m15_data: 15-Minute OHLCV data
            m1_data: 1-Minute OHLCV data (execution timeframe)
            
        Returns:
            TradeSetup if all criteria met, None otherwise
        """
        # Step 1: Analyze HTF Bias
        htf_bias, is_aligned, direction = self.analyze_htf_bias(h1_data, m15_data)
        
        if not is_aligned:
            self.log_debug(f"{pair}: HTF bias not aligned, skipping")
            return None
        
        signal_type = SignalType.LONG if direction == "LONG" else SignalType.SHORT
        
        # Step 2: Map Liquidity (use 15m for pool identification)
        pools = self.map_liquidity(m15_data)
        
        if not pools:
            self.log_debug(f"{pair}: No liquidity pools found")
            return None
        
        # Calculate ATR for thresholds
        atr = self._calculate_atr(m1_data)
        
        # Step 3: Detect Liquidity Sweep
        current_idx = len(m1_data) - 1
        sweep = self.detect_liquidity_sweep(m1_data, current_idx, atr)
        
        if not sweep:
            self.log_debug(f"{pair}: No liquidity sweep detected")
            return None
        
        # Validate sweep direction matches expected trade
        sweep_is_bearish = sweep.pool.type.name.endswith("HIGH")
        sweep_is_bullish = sweep.pool.type.name.endswith("LOW")
        
        if signal_type == SignalType.LONG and not sweep_is_bullish:
            return None
        if signal_type == SignalType.SHORT and not sweep_is_bearish:
            return None
        
        # Step 4: Detect MSS
        htf_structure = self._cached_structures.get(Timeframe.M15)
        if htf_structure is None:
            return None
        
        mss = self.detect_mss(m1_data, htf_structure, atr)
        
        if mss is None:
            self.log_debug(f"{pair}: No MSS confirmation")
            return None
        
        if mss.direction != signal_type:
            self.log_debug(f"{pair}: MSS direction mismatch")
            return None
        
        # Step 5: Find FVG Entry
        current_price = m1_data['close'].iloc[-1]
        entry_fvg = self.find_entry_fvg(m1_data, signal_type, current_price, atr)
        
        if entry_fvg is None:
            self.log_debug(f"{pair}: No valid FVG for entry")
            return None
        
        # Step 6: Calculate Entry/SL/TP
        entry_price = self.fvg_detector.get_entry_price(entry_fvg, self.config.fvg_entry_offset)
        current_atr = atr[-1] if len(atr) > 0 else current_price * 0.001
        
        if signal_type == SignalType.LONG:
            stop_loss = entry_fvg.low - (0.2 * current_atr)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * self.config.default_risk_reward)
        else:
            stop_loss = entry_fvg.high + (0.2 * current_atr)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * self.config.default_risk_reward)
        
        # Calculate RR ratio
        risk_pips = abs(entry_price - stop_loss)
        reward_pips = abs(take_profit - entry_price)
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        # Build TradeSetup
        setup = TradeSetup(
            signal=signal_type,
            timestamp=m1_data.index[-1],
            htf_bias=htf_bias,
            htf_structure=htf_structure,
            liquidity_sweep=sweep,
            mss_event=mss,
            entry_fvg=entry_fvg,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            is_valid=True
        )
        
        self.log_info(f"{pair}: Trade setup generated - {signal_type.name} @ {entry_price:.5f}")
        
        return setup
    
    def validate_setup(self, setup: TradeSetup) -> Tuple[bool, List[str]]:
        """
        Validate trade setup meets all ICT criteria.
        
        Validation Rules:
        1. HTF bias is defined (not NEUTRAL)
        2. Liquidity sweep occurred
        3. MSS confirmed
        4. FVG exists and not fully filled
        5. RR ratio meets minimum (2.2)
        
        Args:
            setup: TradeSetup to validate
            
        Returns:
            Tuple of (is_valid, list of invalidation reasons)
        """
        reasons = []
        
        # Check HTF bias
        if setup.htf_bias == MarketBias.NEUTRAL:
            reasons.append("HTF bias is neutral")
        
        if setup.htf_bias == MarketBias.UNDEFINED:
            reasons.append("HTF bias is undefined")
        
        # Check liquidity sweep
        if setup.liquidity_sweep is None:
            reasons.append("No liquidity sweep")
        elif not setup.liquidity_sweep.rejection:
            reasons.append("Sweep without rejection")
        
        # Check MSS
        if setup.mss_event is None:
            reasons.append("No MSS confirmation")
        elif not setup.mss_event.closes_through_level:
            reasons.append("MSS did not close through level")
        
        # Check FVG
        if setup.entry_fvg is None:
            reasons.append("No entry FVG")
        elif setup.entry_fvg.filled:
            reasons.append("Entry FVG already filled")
        
        # Check RR ratio
        if setup.risk_reward_ratio < self.config.default_risk_reward:
            reasons.append(f"RR ratio {setup.risk_reward_ratio:.2f} below minimum {self.config.default_risk_reward}")
        
        is_valid = len(reasons) == 0
        
        return (is_valid, reasons)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ATR for the data."""
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
    
    def get_narrative_summary(self) -> Dict:
        """
        Get summary of current ICT narrative state.
        
        Returns:
            Dict with current analysis state
        """
        return {
            "h1_bias": self._cached_structures.get(Timeframe.H1, MarketStructure(MarketBias.UNDEFINED)).bias.name,
            "m15_bias": self._cached_structures.get(Timeframe.M15, MarketStructure(MarketBias.UNDEFINED)).bias.name,
            "liquidity_pools": len(self._cached_pools),
            "unswept_pools": len([p for p in self._cached_pools if not p.swept]),
            "active_fvgs": len([f for f in self._cached_fvgs if not f.filled])
        }
