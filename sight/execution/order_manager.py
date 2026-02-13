"""SIGHT Order Manager - Trade execution and order management."""
from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path
import uuid

from ..core.base import ExecutionEngine, BaseLogger
from ..core.types import (
    Trade, TradeSetup, TradeStatus, PairConfig, SignalType, OrderType
)
from ..core.constants import CONFIG_DIR, SUPPORTED_PAIRS
from .risk_manager import RiskManager


class OrderManager(ExecutionEngine):
    """
    Order management and trade execution.
    
    Responsibilities:
    - Load pair-specific configurations
    - Execute trades from validated setups
    - Track open positions
    - Handle order modifications and closures
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        config_path: str = None
    ):
        super().__init__("OrderManager")
        self.risk_manager = risk_manager
        self.config_path = Path(config_path) if config_path else Path(CONFIG_DIR) / "pair_config.json"
        
        # Pair configurations
        self.pair_configs: Dict[str, PairConfig] = {}
        
        # Active trades
        self.active_trades: Dict[str, Trade] = {}
        
        # Trade history
        self.closed_trades: List[Trade] = []
        
        # Account state
        self.account_balance: float = 100000.0
    
    def load_pair_configs(self, config_path: str = None) -> None:
        """
        Load pair-specific configurations from JSON.
        
        Args:
            config_path: Path to config file (optional)
        """
        path = Path(config_path) if config_path else self.config_path
        
        if not path.exists():
            self.log_warning(f"Config file not found: {path}")
            self._load_default_configs()
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for pair, config_data in data.get('pairs', {}).items():
                # Remove optimization_summary if present
                config_data.pop('optimization_summary', None)
                
                self.pair_configs[pair] = PairConfig.from_dict(config_data)
            
            self.log_info(f"Loaded configs for {len(self.pair_configs)} pairs")
            
        except Exception as e:
            self.log_error(f"Error loading configs: {e}")
            self._load_default_configs()
    
    def _load_default_configs(self) -> None:
        """Load default configurations for all pairs."""
        for pair in SUPPORTED_PAIRS:
            self.pair_configs[pair] = PairConfig(pair=pair)
        self.log_info("Loaded default configurations")
    
    def get_pair_config(self, pair: str) -> Optional[PairConfig]:
        """Get configuration for a pair."""
        if pair not in self.pair_configs:
            self.log_warning(f"No config for {pair}, using default")
            return PairConfig(pair=pair)
        return self.pair_configs[pair]
    
    def execute_trade(self, setup: TradeSetup) -> Optional[Trade]:
        """
        Execute trade from validated setup.
        
        Process:
        1. Validate risk parameters
        2. Calculate position size
        3. Create trade record
        4. Simulate execution (or send to broker)
        
        Args:
            setup: Validated trade setup
            
        Returns:
            Executed Trade or None if failed
        """
        if not setup.is_valid:
            self.log_warning("Cannot execute invalid setup")
            return None
        
        # Get pair from setup (assumed to be set elsewhere or derived)
        # For now, we'll extract from context or use a default
        pair = getattr(setup, 'pair', 'EURUSD')
        
        config = self.get_pair_config(pair)
        
        # Validate risk
        is_valid, reason, metrics = self.risk_manager.validate_setup_risk(
            setup, pair, self.account_balance, config
        )
        
        if not is_valid:
            self.log_warning(f"Risk validation failed: {reason}")
            return None
        
        # Create trade
        trade = Trade(
            id=self._generate_trade_id(),
            pair=pair,
            setup=setup,
            status=TradeStatus.ACTIVE,
            entry_time=datetime.now(),
            actual_entry=setup.entry_price,
            position_size=metrics['position_size']
        )
        
        # Update tracking
        self.active_trades[trade.id] = trade
        self.risk_manager.increment_daily_count(pair)
        self.risk_manager.track_open_risk(pair, metrics['risk_amount'])
        
        self.log_info(f"Trade executed: {trade.id} {pair} "
                     f"{'LONG' if setup.signal == SignalType.LONG else 'SHORT'} "
                     f"@ {setup.entry_price:.5f}")
        
        return trade
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        return f"SIG{uuid.uuid4().hex[:8].upper()}"
    
    def modify_trade(
        self,
        trade: Trade,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> Trade:
        """
        Modify existing trade SL/TP.
        
        Args:
            trade: Trade to modify
            new_sl: New stop loss level
            new_tp: New take profit level
            
        Returns:
            Modified trade
        """
        if trade.id not in self.active_trades:
            self.log_error(f"Trade {trade.id} not found in active trades")
            return trade
        
        if new_sl is not None:
            trade.setup.stop_loss = new_sl
            self.log_info(f"Modified {trade.id} SL to {new_sl:.5f}")
        
        if new_tp is not None:
            trade.setup.take_profit = new_tp
            self.log_info(f"Modified {trade.id} TP to {new_tp:.5f}")
        
        return trade
    
    def close_trade(
        self,
        trade: Trade,
        reason: str = "",
        exit_price: float = None
    ) -> Trade:
        """
        Close an active trade.
        
        Args:
            trade: Trade to close
            reason: Closure reason
            exit_price: Exit price (optional, defaults to current)
            
        Returns:
            Closed trade
        """
        if trade.id not in self.active_trades:
            self.log_error(f"Trade {trade.id} not found")
            return trade
        
        # Calculate PnL
        entry = trade.actual_entry
        exit_px = exit_price or entry  # Would be current price in live
        
        if trade.setup.signal == SignalType.LONG:
            pnl_points = exit_px - entry
        else:
            pnl_points = entry - exit_px
        
        # Simplified PnL calculation
        trade.actual_exit = exit_px
        trade.exit_time = datetime.now()
        trade.pnl_pips = pnl_points * 10000  # Simplified
        trade.pnl_currency = trade.pnl_pips * 10  # Simplified
        trade.status = TradeStatus.CLOSED
        trade.notes = reason
        
        # Update balance
        self.account_balance += trade.pnl_currency
        
        # Move to closed trades
        del self.active_trades[trade.id]
        self.closed_trades.append(trade)
        
        # Release risk
        config = self.get_pair_config(trade.pair)
        sl_pips = abs(entry - trade.setup.stop_loss) * 10000
        risk_amount = sl_pips * 10 * trade.position_size
        self.risk_manager.release_risk(trade.pair, risk_amount)
        
        self.log_info(f"Closed {trade.id}: PnL {trade.pnl_currency:+.2f} ({reason})")
        
        return trade
    
    def check_trade_status(
        self,
        trade: Trade,
        current_high: float,
        current_low: float
    ) -> Optional[str]:
        """
        Check if trade hit SL or TP.
        
        Args:
            trade: Trade to check
            current_high: Current candle high
            current_low: Current candle low
            
        Returns:
            'SL', 'TP', or None
        """
        setup = trade.setup
        
        if setup.signal == SignalType.LONG:
            if current_low <= setup.stop_loss:
                return 'SL'
            if current_high >= setup.take_profit:
                return 'TP'
        else:
            if current_high >= setup.stop_loss:
                return 'SL'
            if current_low <= setup.take_profit:
                return 'TP'
        
        return None
    
    def get_active_trades(self) -> List[Trade]:
        """Get all active trades."""
        return list(self.active_trades.values())
    
    def get_active_trade_for_pair(self, pair: str) -> Optional[Trade]:
        """Get active trade for specific pair."""
        for trade in self.active_trades.values():
            if trade.pair == pair:
                return trade
        return None
    
    def has_open_position(self, pair: str) -> bool:
        """Check if there's an open position for pair."""
        return self.get_active_trade_for_pair(pair) is not None
    
    def get_closed_trades(
        self,
        pair: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Trade]:
        """
        Get closed trades with optional filters.
        
        Args:
            pair: Filter by pair
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Filtered list of trades
        """
        trades = self.closed_trades
        
        if pair:
            trades = [t for t in trades if t.pair == pair]
        
        if start_date:
            trades = [t for t in trades if t.exit_time and t.exit_time >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.exit_time and t.exit_time <= end_date]
        
        return trades
    
    def get_account_state(self) -> Dict:
        """Get current account state summary."""
        total_open_pnl = sum(
            t.pnl_currency for t in self.active_trades.values() 
            if hasattr(t, 'pnl_currency')
        )
        
        return {
            'balance': self.account_balance,
            'equity': self.account_balance + total_open_pnl,
            'open_trades': len(self.active_trades),
            'total_trades': len(self.closed_trades),
            'open_risk': self.risk_manager.get_total_risk_exposure()
        }
    
    def update_from_market_data(
        self,
        pair: str,
        high: float,
        low: float,
        close: float
    ) -> List[Trade]:
        """
        Update active trades based on market data.
        
        Args:
            pair: Currency pair
            high: Current candle high
            low: Current candle low
            close: Current candle close
            
        Returns:
            List of trades that were closed
        """
        closed = []
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade.pair != pair:
                continue
            
            status = self.check_trade_status(trade, high, low)
            
            if status == 'SL':
                self.close_trade(trade, "Stop Loss Hit", trade.setup.stop_loss)
                closed.append(trade)
            elif status == 'TP':
                self.close_trade(trade, "Take Profit Hit", trade.setup.take_profit)
                closed.append(trade)
        
        return closed
