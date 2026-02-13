"""SIGHT Base Classes - Abstract interfaces and base implementations."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .types import (
    Timeframe, MarketBias, SignalType, TradeSetup, Trade,
    BacktestResult, PairConfig, MarketStructure, FairValueGap,
    LiquidityPool, LiquiditySweep, DisplacementEvent, OHLCV
)


class BaseLogger:
    """Standardized logging mixin for all SIGHT components."""
    
    def __init__(self, name: str = None):
        self._logger = logging.getLogger(name or self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            ))
            self._logger.addHandler(handler)
    
    def log_info(self, msg: str) -> None:
        self._logger.info(msg)
    
    def log_debug(self, msg: str) -> None:
        self._logger.debug(msg)
    
    def log_warning(self, msg: str) -> None:
        self._logger.warning(msg)
    
    def log_error(self, msg: str) -> None:
        self._logger.error(msg)


class DataProvider(ABC, BaseLogger):
    """Abstract interface for market data providers."""
    
    @abstractmethod
    def get_ohlcv(
        self, 
        pair: str, 
        timeframe: Timeframe,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Retrieve OHLCV data for specified pair and timeframe."""
        pass
    
    @abstractmethod
    def get_latest_candle(self, pair: str, timeframe: Timeframe) -> OHLCV:
        """Get the most recent completed candle."""
        pass
    
    @abstractmethod
    def get_spread(self, pair: str) -> float:
        """Get current bid-ask spread in pips."""
        pass


class StructureAnalyzer(ABC, BaseLogger):
    """Abstract interface for market structure analysis."""
    
    @abstractmethod
    def detect_swings(
        self, 
        data: pd.DataFrame, 
        strength: int = 3
    ) -> List[Any]:
        """Detect swing highs and lows."""
        pass
    
    @abstractmethod
    def determine_bias(
        self, 
        data: pd.DataFrame,
        timeframe: Timeframe
    ) -> MarketStructure:
        """Determine current market bias from structure."""
        pass
    
    @abstractmethod
    def detect_structure_shift(
        self, 
        data: pd.DataFrame,
        current_structure: MarketStructure
    ) -> Optional[DisplacementEvent]:
        """Detect market structure shift (MSS)."""
        pass


class LiquidityMapper(ABC, BaseLogger):
    """Abstract interface for liquidity analysis."""
    
    @abstractmethod
    def identify_pools(
        self, 
        data: pd.DataFrame,
        lookback: int = 48
    ) -> List[LiquidityPool]:
        """Identify liquidity pools (highs/lows)."""
        pass
    
    @abstractmethod
    def detect_sweep(
        self, 
        data: pd.DataFrame,
        pools: List[LiquidityPool],
        current_idx: int
    ) -> Optional[LiquiditySweep]:
        """Detect liquidity sweep events."""
        pass
    
    @abstractmethod
    def get_session_levels(
        self, 
        data: pd.DataFrame,
        session: str
    ) -> Tuple[float, float]:
        """Get previous session high and low."""
        pass


class GapAnalyzer(ABC, BaseLogger):
    """Abstract interface for Fair Value Gap analysis."""
    
    @abstractmethod
    def detect_fvg(
        self, 
        data: pd.DataFrame,
        min_size_atr: float = 0.3
    ) -> List[FairValueGap]:
        """Detect Fair Value Gaps."""
        pass
    
    @abstractmethod
    def check_fvg_fill(
        self, 
        fvg: FairValueGap,
        current_price: float
    ) -> float:
        """Check FVG fill percentage."""
        pass
    
    @abstractmethod
    def get_valid_entry_fvg(
        self, 
        fvgs: List[FairValueGap],
        direction: SignalType,
        current_price: float
    ) -> Optional[FairValueGap]:
        """Get the nearest valid FVG for entry."""
        pass


class ConfluenceChecker(ABC, BaseLogger):
    """Abstract interface for confluence validation."""
    
    @abstractmethod
    def check_bollinger_touch(
        self, 
        data: pd.DataFrame,
        direction: SignalType
    ) -> bool:
        """Check if price touched Bollinger Band boundary."""
        pass
    
    @abstractmethod
    def check_keltner_touch(
        self, 
        data: pd.DataFrame,
        direction: SignalType
    ) -> bool:
        """Check if price touched Keltner Channel boundary."""
        pass
    
    @abstractmethod
    def calculate_confluence_score(
        self, 
        setup: TradeSetup
    ) -> float:
        """Calculate overall confluence score."""
        pass


class SignalGenerator(ABC, BaseLogger):
    """Abstract interface for trade signal generation."""
    
    @abstractmethod
    def generate_signal(
        self, 
        pair: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        config: PairConfig
    ) -> Optional[TradeSetup]:
        """Generate trade setup from market data."""
        pass
    
    @abstractmethod
    def validate_setup(self, setup: TradeSetup) -> bool:
        """Validate trade setup meets all criteria."""
        pass


class RiskManager(ABC, BaseLogger):
    """Abstract interface for risk management."""
    
    @abstractmethod
    def calculate_position_size(
        self, 
        account_balance: float,
        risk_percent: float,
        stop_loss_pips: float,
        pair: str
    ) -> float:
        """Calculate position size based on risk parameters."""
        pass
    
    @abstractmethod
    def validate_risk_reward(
        self, 
        setup: TradeSetup,
        min_rr: float = 2.2
    ) -> bool:
        """Validate setup meets minimum risk-reward."""
        pass
    
    @abstractmethod
    def check_daily_limit(
        self, 
        pair: str,
        trades_today: int
    ) -> bool:
        """Check if daily trade limit reached."""
        pass


class ExecutionEngine(ABC, BaseLogger):
    """Abstract interface for trade execution."""
    
    @abstractmethod
    def execute_trade(self, setup: TradeSetup) -> Trade:
        """Execute trade from validated setup."""
        pass
    
    @abstractmethod
    def modify_trade(
        self, 
        trade: Trade,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> Trade:
        """Modify existing trade."""
        pass
    
    @abstractmethod
    def close_trade(
        self, 
        trade: Trade,
        reason: str = ""
    ) -> Trade:
        """Close existing trade."""
        pass


class PerformanceMonitor(ABC, BaseLogger):
    """Abstract interface for performance monitoring."""
    
    @abstractmethod
    def calculate_z_score(self, trades: List[Trade]) -> float:
        """Calculate Z-score for streak dependency."""
        pass
    
    @abstractmethod
    def calculate_expectancy(self, trades: List[Trade]) -> float:
        """Calculate trade expectancy value."""
        pass
    
    @abstractmethod
    def check_confidence(
        self, 
        trades: List[Trade],
        lookback: int = 50
    ) -> Tuple[bool, str]:
        """Check trading confidence based on recent performance."""
        pass


class BacktestEngine(ABC, BaseLogger):
    """Abstract interface for backtesting."""
    
    @abstractmethod
    def run_backtest(
        self, 
        pair: str,
        start_date: datetime,
        end_date: datetime,
        config: PairConfig
    ) -> BacktestResult:
        """Run backtest for specified parameters."""
        pass
    
    @abstractmethod
    def calculate_metrics(
        self, 
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        pass


class Optimizer(ABC, BaseLogger):
    """Abstract interface for parameter optimization."""
    
    @abstractmethod
    def optimize_pair(
        self, 
        pair: str,
        train_start: datetime,
        train_end: datetime
    ) -> PairConfig:
        """Optimize parameters for a single pair."""
        pass
    
    @abstractmethod
    def walk_forward_validate(
        self, 
        pair: str,
        config: PairConfig,
        val_start: datetime,
        val_end: datetime
    ) -> Tuple[bool, BacktestResult]:
        """Validate optimized parameters on out-of-sample data."""
        pass


@dataclass
class EngineState:
    """Global engine state container."""
    is_running: bool = False
    is_trading_enabled: bool = True
    current_pair: Optional[str] = None
    active_trades: Dict[str, Trade] = None
    daily_trade_counts: Dict[str, int] = None
    performance_paused_pairs: List[str] = None
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        self.active_trades = self.active_trades or {}
        self.daily_trade_counts = self.daily_trade_counts or {}
        self.performance_paused_pairs = self.performance_paused_pairs or []
    
    def reset_daily_counts(self):
        self.daily_trade_counts = {pair: 0 for pair in self.daily_trade_counts}


class SIGHTEngine(BaseLogger):
    """Main SIGHT trading engine orchestrator."""
    
    def __init__(
        self,
        data_provider: DataProvider,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
        performance_monitor: PerformanceMonitor,
        configs: Dict[str, PairConfig]
    ):
        super().__init__("SIGHTEngine")
        self.data_provider = data_provider
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.performance_monitor = performance_monitor
        self.configs = configs
        self.state = EngineState()
        self.trade_history: List[Trade] = []
    
    def start(self) -> None:
        """Start the trading engine."""
        self.state.is_running = True
        self.log_info("SIGHT Engine started")
    
    def stop(self) -> None:
        """Stop the trading engine."""
        self.state.is_running = False
        self.log_info("SIGHT Engine stopped")
    
    def process_tick(self, pair: str) -> Optional[Trade]:
        """Process single market tick for given pair."""
        if not self.state.is_running:
            return None
        
        if pair in self.state.performance_paused_pairs:
            return None
        
        config = self.configs.get(pair)
        if not config:
            self.log_warning(f"No config found for {pair}")
            return None
        
        # Implementation would be filled in by concrete classes
        return None
