"""SIGHT Main Engine - Complete trading system orchestration."""
from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path

from .core.base import BaseLogger, EngineState
from .core.types import (
    Timeframe, Trade, TradeSetup, PairConfig, TradeStatus
)
from .core.constants import SUPPORTED_PAIRS, CONFIG_DIR

# Data Pipeline
from .data.pipeline import DataPipeline
from .data.provider import ParquetDataProvider

# Strategies
from .strategies.ict_engine import ICTEngine
from .strategies.signal_generator import ICTSignalGenerator

# Analysis
from .analysis.confluence import ConfluenceFilter

# Monitoring
from .monitoring.performance_tracker import PerformanceTracker
from .monitoring.stress_test import StressTest
from .monitoring.slippage_monitor import SlippageMonitor
from .monitoring.dashboard import DashboardMetrics

# Optimization
from .optimization.backtest import BacktestEngine
from .optimization.optimizer import ParameterOptimizer

# Execution
from .execution.order_manager import OrderManager
from .execution.risk_manager import RiskManager
from .execution.position_tracker import PositionTracker

# Config
from .config.loader import ConfigLoader


class SIGHTEngine(BaseLogger):
    """
    SIGHT - Institutional ICT Trading Engine
    
    Main orchestrator that integrates all components:
    - Data Pipeline (Phase 1)
    - ICT Narrative Engine (Phase 2) - 80% decision weight
    - Confluence Filter (Phase 3) - 20% decision weight
    - Performance Monitoring (Phase 4)
    - Optimization Engine (Phase 5)
    - Execution Layer (Phase 6)
    
    Performance Targets:
    - Minimum 70% Aggregate Win Rate
    - Minimum 1:2.2 Risk-to-Reward
    - 12 Currency Pairs
    - 1-2 Trades per Day
    - Walk-Forward Validated
    """
    
    def __init__(
        self,
        data_dir: str = "data/parquet",
        config_dir: str = CONFIG_DIR,
        initial_equity: float = 100000.0
    ):
        super().__init__("SIGHTEngine")
        
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.state = EngineState()
        self.state.active_trades = {}
        self.state.daily_trade_counts = {pair: 0 for pair in SUPPORTED_PAIRS}
        
        # Initialize components
        self._init_components(initial_equity)
        
        # Load configurations
        self.pair_configs: Dict[str, PairConfig] = {}
        
        self.log_info("SIGHT Engine initialized")
    
    def _init_components(self, initial_equity: float) -> None:
        """Initialize all engine components."""
        # Data Provider
        self.data_provider = ParquetDataProvider(str(self.data_dir))
        
        # Config Loader
        self.config_loader = ConfigLoader(str(self.config_dir))
        
        # ICT Engine (80% weight)
        self.ict_engine = ICTEngine()
        
        # Confluence Filter (20% weight)
        self.confluence_filter = ConfluenceFilter()
        
        # Signal Generator
        self.signal_generator = ICTSignalGenerator(
            ict_engine=self.ict_engine,
            confluence_filter=self.confluence_filter
        )
        
        # Risk Manager
        self.risk_manager = RiskManager()
        
        # Order Manager
        self.order_manager = OrderManager(
            risk_manager=self.risk_manager,
            config_path=str(self.config_dir / "pair_config.json")
        )
        self.order_manager.account_balance = initial_equity
        
        # Position Tracker
        self.position_tracker = PositionTracker()
        
        # Performance Monitoring
        self.performance_tracker = PerformanceTracker()
        self.stress_tester = StressTest()
        self.slippage_monitor = SlippageMonitor()
        self.dashboard = DashboardMetrics(initial_equity)
        
        # Backtest Engine
        self.backtest_engine = BacktestEngine(
            data_provider=self.data_provider,
            ict_engine=self.ict_engine,
            confluence_filter=self.confluence_filter
        )
        
        # Optimizer
        self.optimizer = ParameterOptimizer(
            backtest_engine=self.backtest_engine,
            output_dir=str(self.config_dir)
        )
    
    def load_configs(self, filename: str = "pair_config.json") -> None:
        """Load pair configurations."""
        self.pair_configs = self.config_loader.load_pair_configs(filename)
        self.order_manager.load_pair_configs(str(self.config_dir / filename))
        self.log_info(f"Loaded {len(self.pair_configs)} pair configurations")
    
    def start(self) -> None:
        """Start the trading engine."""
        self.state.is_running = True
        self.state.is_trading_enabled = True
        self.log_info("SIGHT Engine started")
    
    def stop(self) -> None:
        """Stop the trading engine."""
        self.state.is_running = False
        self.log_info("SIGHT Engine stopped")
    
    def process_pair(
        self,
        pair: str,
        current_time: datetime = None
    ) -> Optional[Trade]:
        """
        Process a single pair for potential trade entry.
        
        Args:
            pair: Currency pair to process
            current_time: Current timestamp (for backtesting)
            
        Returns:
            Executed Trade or None
        """
        if not self.state.is_running or not self.state.is_trading_enabled:
            return None
        
        # Check if pair is blocked
        if self.slippage_monitor.is_pair_blocked(pair):
            return None
        
        # Check if already has open position
        if self.position_tracker.has_position(pair):
            return None
        
        # Check daily limit
        if not self.risk_manager.check_daily_limit(pair):
            return None
        
        # Get config
        config = self.pair_configs.get(pair, PairConfig(pair=pair))
        
        current_time = current_time or datetime.now()
        
        # Get data windows
        try:
            h1_data = self.data_provider.get_ohlcv(
                pair, Timeframe.H1,
                current_time - datetime.timedelta(days=30),
                current_time
            )
            m15_data = self.data_provider.get_ohlcv(
                pair, Timeframe.M15,
                current_time - datetime.timedelta(days=7),
                current_time
            )
            m1_data = self.data_provider.get_ohlcv(
                pair, Timeframe.M1,
                current_time - datetime.timedelta(days=1),
                current_time
            )
        except Exception as e:
            self.log_error(f"Error loading data for {pair}: {e}")
            return None
        
        if len(h1_data) < 50 or len(m15_data) < 100 or len(m1_data) < 100:
            return None
        
        # Generate signal
        setup = self.signal_generator.generate_signal_mtf(
            pair, h1_data, m15_data, m1_data, config
        )
        
        if setup is None or not setup.is_valid:
            return None
        
        # Add pair to setup
        setup.pair = pair
        
        # Check slippage conditions
        spread = self.data_provider.get_spread(pair)
        sl_pips = self.risk_manager.get_stop_loss_pips(setup, pair)
        
        can_trade, reason = self.slippage_monitor.check_entry_conditions(
            pair, spread, 0.5, sl_pips, config
        )
        
        if not can_trade:
            self.log_debug(f"{pair}: {reason}")
            return None
        
        # Execute trade
        trade = self.order_manager.execute_trade(setup)
        
        if trade:
            self.position_tracker.add_position(trade)
            self.dashboard.update_trade(trade)
        
        return trade
    
    def update_positions(self, pair: str, high: float, low: float, close: float) -> List[Trade]:
        """
        Update positions based on market data.
        
        Args:
            pair: Currency pair
            high: Current candle high
            low: Current candle low
            close: Current close price
            
        Returns:
            List of closed trades
        """
        # Update price in tracker
        self.position_tracker.update_price(pair, close)
        
        # Check for SL/TP hits
        closed_trades = self.order_manager.update_from_market_data(pair, high, low, close)
        
        for trade in closed_trades:
            # Update tracking
            self.position_tracker.remove_position(pair)
            self.performance_tracker.add_trade(trade)
            self.dashboard.update_trade(trade)
            
            # Record slippage
            if trade.slippage_pips != 0:
                self.slippage_monitor.record_slippage(
                    pair, trade.setup.entry_price, trade.actual_entry,
                    self.data_provider.get_spread(pair),
                    abs(trade.setup.entry_price - trade.setup.stop_loss) * 10000,
                    trade.setup.signal.name == "LONG"
                )
        
        return closed_trades
    
    def run_optimization(
        self,
        pairs: List[str] = None,
        train_start: datetime = None,
        train_end: datetime = None,
        val_start: datetime = None,
        val_end: datetime = None
    ) -> Dict:
        """
        Run parameter optimization for all pairs.
        
        Args:
            pairs: Optional list of pairs (default: all)
            train_start: Training period start
            train_end: Training period end
            val_start: Validation start
            val_end: Validation end
            
        Returns:
            Optimization results
        """
        self.log_info("Starting optimization run")
        
        results = self.optimizer.optimize_all_pairs(
            train_start, train_end, val_start, val_end
        )
        
        # Save configs
        self.optimizer.save_configs(results)
        
        # Generate report
        report = self.optimizer.generate_report(results)
        
        # Save report
        report_path = self.config_dir / "optimization_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.log_info(f"Optimization complete. Report saved to {report_path}")
        
        return results
    
    def run_backtest(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        config: PairConfig = None
    ):
        """
        Run backtest for a single pair.
        
        Args:
            pair: Currency pair
            start: Start date
            end: End date
            config: Optional config (default: loaded config)
            
        Returns:
            BacktestResult
        """
        config = config or self.pair_configs.get(pair, PairConfig(pair=pair))
        return self.backtest_engine.run_backtest(pair, start, end, config)
    
    def run_stress_test(self) -> Dict:
        """
        Run Monte Carlo stress test on recent trades.
        
        Returns:
            StressTestResult as dict
        """
        trades = self.performance_tracker.trade_history
        
        if len(trades) < 20:
            self.log_warning("Insufficient trades for stress test")
            return {"error": "Insufficient trades"}
        
        result = self.stress_tester.run_stress_test(
            trades,
            self.order_manager.account_balance
        )
        
        # Generate and save report
        report = self.stress_tester.generate_report(result)
        report_path = self.config_dir / "stress_test_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return result.__dict__
    
    def get_dashboard(self) -> str:
        """Get formatted dashboard display."""
        return self.dashboard.format_dashboard()
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        report = self.performance_tracker.generate_report()
        return {
            'timestamp': report.timestamp.isoformat(),
            'total_trades': report.total_trades,
            'win_rate': report.win_rate,
            'profit_factor': report.profit_factor,
            'expected_value': report.expected_value,
            'z_score': report.z_score,
            'max_drawdown': report.max_drawdown,
            'is_healthy': report.is_healthy,
            'warnings': report.warnings,
            'recommendations': report.recommendations
        }
    
    def export_state(self) -> Dict:
        """Export current engine state."""
        return {
            'is_running': self.state.is_running,
            'is_trading_enabled': self.state.is_trading_enabled,
            'account': self.order_manager.get_account_state(),
            'open_positions': [
                p.__dict__ for p in self.position_tracker.get_all_positions()
            ],
            'performance': self.get_performance_report(),
            'blocked_pairs': self.slippage_monitor.get_blocked_pairs()
        }


def create_engine(
    historical_data_dir: str = "historical_data",
    output_dir: str = "data/parquet",
    config_dir: str = "config",
    run_pipeline: bool = True
) -> SIGHTEngine:
    """
    Factory function to create and initialize SIGHT engine.
    
    Args:
        historical_data_dir: Directory with ZIP files
        output_dir: Directory for Parquet files
        config_dir: Directory for configurations
        run_pipeline: Whether to run data pipeline
        
    Returns:
        Initialized SIGHTEngine
    """
    # Run data pipeline if needed
    if run_pipeline:
        pipeline = DataPipeline(
            historical_data_dir=historical_data_dir,
            output_dir=output_dir
        )
        pipeline.run_pipeline()
    
    # Create engine
    engine = SIGHTEngine(
        data_dir=output_dir,
        config_dir=config_dir
    )
    
    # Load configs
    engine.load_configs()
    
    return engine
