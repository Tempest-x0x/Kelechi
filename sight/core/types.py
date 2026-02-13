"""SIGHT Type Definitions - Production-grade type system for trading engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from datetime import datetime
import numpy as np
import pandas as pd


class Timeframe(Enum):
    """Supported trading timeframes."""
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"

    @property
    def minutes(self) -> int:
        mapping = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "1H": 60, "4H": 240, "1D": 1440, "1W": 10080
        }
        return mapping[self.value]


class MarketBias(Enum):
    """Market directional bias."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()
    UNDEFINED = auto()


class SignalType(Enum):
    """Trade signal types."""
    LONG = auto()
    SHORT = auto()
    NO_SIGNAL = auto()


class OrderType(Enum):
    """Order execution types."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class TradeStatus(Enum):
    """Trade lifecycle status."""
    PENDING = auto()
    ACTIVE = auto()
    PARTIAL = auto()
    CLOSED = auto()
    CANCELLED = auto()
    EXPIRED = auto()


class LiquidityType(Enum):
    """Liquidity pool classification."""
    SESSION_HIGH = auto()
    SESSION_LOW = auto()
    SWING_HIGH = auto()
    SWING_LOW = auto()
    EQUAL_HIGH = auto()
    EQUAL_LOW = auto()
    RELATIVE_HIGH = auto()
    RELATIVE_LOW = auto()


class FVGType(Enum):
    """Fair Value Gap types."""
    BULLISH = auto()  # Gap created by bullish displacement
    BEARISH = auto()  # Gap created by bearish displacement


@dataclass(frozen=True)
class OHLCV:
    """Immutable OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def range_size(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def body_percent(self) -> float:
        if self.range_size == 0:
            return 0.0
        return self.body_size / self.range_size


@dataclass
class SwingPoint:
    """Market structure swing point."""
    timestamp: datetime
    price: float
    index: int
    is_high: bool
    strength: int = 1  # Number of candles on each side defining the swing
    
    @property
    def type_str(self) -> str:
        return "HIGH" if self.is_high else "LOW"


@dataclass
class MarketStructure:
    """Market structure state container."""
    bias: MarketBias
    last_higher_high: Optional[SwingPoint] = None
    last_higher_low: Optional[SwingPoint] = None
    last_lower_high: Optional[SwingPoint] = None
    last_lower_low: Optional[SwingPoint] = None
    swing_points: List[SwingPoint] = field(default_factory=list)
    structure_break_price: Optional[float] = None
    confidence_score: float = 0.0


@dataclass
class LiquidityPool:
    """Liquidity accumulation zone."""
    type: LiquidityType
    price: float
    timestamp: datetime
    index: int
    strength: float = 1.0  # Multiple touches increase strength
    swept: bool = False
    sweep_timestamp: Optional[datetime] = None


@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) structure."""
    type: FVGType
    high: float  # Upper bound of gap
    low: float   # Lower bound of gap
    midpoint: float
    timestamp: datetime
    index: int
    filled: bool = False
    fill_percent: float = 0.0
    creation_candle_size: float = 0.0
    
    @property
    def size(self) -> float:
        return self.high - self.low
    
    def get_entry_price(self, offset_pct: float = 0.5) -> float:
        """Get entry price at specified offset (0=low, 0.5=mid, 1=high)."""
        return self.low + (self.size * offset_pct)


@dataclass
class DisplacementEvent:
    """Market Structure Shift (MSS) displacement event."""
    timestamp: datetime
    index: int
    direction: SignalType
    displacement_size: float  # ATR multiple or pip size
    candle_count: int = 1
    closes_through_level: bool = False
    level_broken: Optional[float] = None


@dataclass
class LiquiditySweep:
    """Liquidity sweep event."""
    pool: LiquidityPool
    sweep_candle: OHLCV
    sweep_timestamp: datetime
    sweep_index: int
    rejection: bool = False  # Price closed back inside
    sweep_depth_pips: float = 0.0


@dataclass
class TradeSetup:
    """Complete trade setup with all ICT components."""
    signal: SignalType
    timestamp: datetime
    
    # HTF Context
    htf_bias: MarketBias
    htf_structure: MarketStructure
    
    # Entry Components
    liquidity_sweep: Optional[LiquiditySweep] = None
    mss_event: Optional[DisplacementEvent] = None
    entry_fvg: Optional[FairValueGap] = None
    
    # Price Levels
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Confluence
    bollinger_touch: bool = False
    keltner_touch: bool = False
    confluence_score: float = 0.0
    
    # Risk Parameters
    risk_reward_ratio: float = 0.0
    position_size: float = 0.0
    risk_percent: float = 0.01
    
    # Validation
    is_valid: bool = False
    invalidation_reasons: List[str] = field(default_factory=list)


@dataclass
class Trade:
    """Executed trade record."""
    id: str
    pair: str
    setup: TradeSetup
    status: TradeStatus = TradeStatus.PENDING
    
    # Execution
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    actual_entry: float = 0.0
    actual_exit: float = 0.0
    
    # Results
    pnl_pips: float = 0.0
    pnl_currency: float = 0.0
    actual_rr: float = 0.0
    slippage_pips: float = 0.0
    
    # Metadata
    notes: str = ""


@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    pair: str
    timeframe: Timeframe
    start_date: datetime
    end_date: datetime
    
    # Core Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Profitability
    profit_factor: float = 0.0
    expected_value: float = 0.0
    total_pnl: float = 0.0
    
    # Risk Metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Statistical
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade Distribution
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    @property
    def meets_targets(self) -> bool:
        return self.win_rate >= 0.70 and self.avg_rr >= 2.2


@dataclass
class PairConfig:
    """Optimized configuration for a currency pair."""
    pair: str
    
    # Sweep Parameters
    sweep_depth_atr_multiple: float = 0.5
    sweep_lookback_candles: int = 48
    
    # Displacement Parameters
    displacement_threshold_atr: float = 1.5
    displacement_body_percent_min: float = 0.7
    
    # FVG Parameters
    fvg_min_size_atr: float = 0.3
    fvg_entry_offset: float = 0.5  # 0=low, 0.5=mid, 1=high
    
    # HTF Parameters
    htf_ema_period: int = 200
    swing_strength: int = 3
    
    # Risk Parameters
    default_risk_reward: float = 2.2
    max_spread_percent_of_target: float = 0.15
    max_daily_trades: int = 2
    
    # Confluence Parameters
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_multiple: float = 1.5
    
    # Validation Results
    backtest_win_rate: float = 0.0
    backtest_profit_factor: float = 0.0
    validation_passed: bool = False
    optimization_iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "sweep_depth_atr_multiple": self.sweep_depth_atr_multiple,
            "sweep_lookback_candles": self.sweep_lookback_candles,
            "displacement_threshold_atr": self.displacement_threshold_atr,
            "displacement_body_percent_min": self.displacement_body_percent_min,
            "fvg_min_size_atr": self.fvg_min_size_atr,
            "fvg_entry_offset": self.fvg_entry_offset,
            "htf_ema_period": self.htf_ema_period,
            "swing_strength": self.swing_strength,
            "default_risk_reward": self.default_risk_reward,
            "max_spread_percent_of_target": self.max_spread_percent_of_target,
            "max_daily_trades": self.max_daily_trades,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "kc_period": self.kc_period,
            "kc_atr_multiple": self.kc_atr_multiple,
            "backtest_win_rate": self.backtest_win_rate,
            "backtest_profit_factor": self.backtest_profit_factor,
            "validation_passed": self.validation_passed,
            "optimization_iterations": self.optimization_iterations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairConfig":
        return cls(**data)


# Type aliases for clarity
PriceLevel = float
PipValue = float
ATRValue = float
EquityCurve = List[float]
TradeHistory = List[Trade]
ConfigDict = Dict[str, Any]
