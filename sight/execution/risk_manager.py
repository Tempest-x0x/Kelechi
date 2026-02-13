"""SIGHT Risk Manager - Position sizing and risk controls."""
from typing import Dict, Optional, Tuple
from datetime import datetime, date

from ..core.base import RiskManager as BaseRiskManager
from ..core.types import TradeSetup, PairConfig, SignalType
from ..core.constants import (
    DEFAULT_RISK_PERCENT, MAX_RISK_PERCENT, TARGET_RISK_REWARD,
    MAX_DAILY_TRADES, PIP_VALUES, POINT_TO_PIP
)


class RiskManager(BaseRiskManager):
    """
    Risk management for SIGHT trading engine.
    
    Responsibilities:
    - Position size calculation
    - Risk-reward validation
    - Daily trade limit enforcement
    - Maximum risk exposure tracking
    """
    
    def __init__(
        self,
        default_risk_percent: float = DEFAULT_RISK_PERCENT,
        max_risk_percent: float = MAX_RISK_PERCENT,
        max_daily_trades: int = MAX_DAILY_TRADES
    ):
        super().__init__("RiskManager")
        self.default_risk_percent = default_risk_percent
        self.max_risk_percent = max_risk_percent
        self.max_daily_trades = max_daily_trades
        
        # Daily trade tracking
        self.daily_trade_counts: Dict[str, Dict[date, int]] = {}
        
        # Risk exposure tracking
        self.open_risk: Dict[str, float] = {}  # pair -> risk amount
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        stop_loss_pips: float,
        pair: str
    ) -> float:
        """
        Calculate position size based on fixed fractional risk.
        
        Formula:
        Position Size (lots) = Risk Amount / (Stop Loss in Pips × Pip Value)
        
        Args:
            account_balance: Current account balance
            risk_percent: Risk as decimal (e.g., 0.01 = 1%)
            stop_loss_pips: Stop loss distance in pips
            pair: Currency pair
            
        Returns:
            Position size in lots
        """
        # Validate inputs
        if stop_loss_pips <= 0:
            self.log_error("Invalid stop loss distance")
            return 0.0
        
        # Cap risk percentage
        risk_percent = min(risk_percent, self.max_risk_percent)
        
        # Calculate risk amount
        risk_amount = account_balance * risk_percent
        
        # Get pip value for pair
        pip_value = PIP_VALUES.get(pair, 10.0)  # Default $10 per pip per lot
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to standard lot sizes
        position_size = self._round_position_size(position_size)
        
        self.log_debug(f"Position size: {position_size:.2f} lots "
                      f"(Risk: ${risk_amount:.2f}, SL: {stop_loss_pips:.1f} pips)")
        
        return position_size
    
    def _round_position_size(self, size: float) -> float:
        """Round to nearest micro lot (0.01)."""
        return round(size, 2)
    
    def validate_risk_reward(
        self,
        setup: TradeSetup,
        min_rr: float = TARGET_RISK_REWARD
    ) -> bool:
        """
        Validate that setup meets minimum risk-reward requirement.
        
        Args:
            setup: Trade setup to validate
            min_rr: Minimum required RR ratio
            
        Returns:
            True if RR meets requirement
        """
        if setup.risk_reward_ratio < min_rr:
            self.log_debug(f"RR {setup.risk_reward_ratio:.2f} below minimum {min_rr}")
            return False
        
        return True
    
    def check_daily_limit(
        self,
        pair: str,
        trades_today: int = None
    ) -> bool:
        """
        Check if daily trade limit has been reached.
        
        Args:
            pair: Currency pair
            trades_today: Override for today's trade count
            
        Returns:
            True if can still trade
        """
        today = datetime.now().date()
        
        if pair not in self.daily_trade_counts:
            self.daily_trade_counts[pair] = {}
        
        current_count = trades_today
        if current_count is None:
            current_count = self.daily_trade_counts[pair].get(today, 0)
        
        if current_count >= self.max_daily_trades:
            self.log_debug(f"{pair}: Daily limit reached ({current_count}/{self.max_daily_trades})")
            return False
        
        return True
    
    def increment_daily_count(self, pair: str) -> None:
        """Increment daily trade count for pair."""
        today = datetime.now().date()
        
        if pair not in self.daily_trade_counts:
            self.daily_trade_counts[pair] = {}
        
        current = self.daily_trade_counts[pair].get(today, 0)
        self.daily_trade_counts[pair][today] = current + 1
    
    def reset_daily_counts(self) -> None:
        """Reset all daily trade counts (call at day start)."""
        for pair in self.daily_trade_counts:
            self.daily_trade_counts[pair] = {}
        self.log_info("Daily trade counts reset")
    
    def get_stop_loss_pips(
        self,
        setup: TradeSetup,
        pair: str
    ) -> float:
        """
        Calculate stop loss distance in pips.
        
        Args:
            setup: Trade setup
            pair: Currency pair
            
        Returns:
            Stop loss distance in pips
        """
        price_diff = abs(setup.entry_price - setup.stop_loss)
        pip_mult = POINT_TO_PIP.get(pair, 10000)
        
        return price_diff * pip_mult
    
    def get_take_profit_pips(
        self,
        setup: TradeSetup,
        pair: str
    ) -> float:
        """
        Calculate take profit distance in pips.
        
        Args:
            setup: Trade setup
            pair: Currency pair
            
        Returns:
            Take profit distance in pips
        """
        price_diff = abs(setup.take_profit - setup.entry_price)
        pip_mult = POINT_TO_PIP.get(pair, 10000)
        
        return price_diff * pip_mult
    
    def validate_setup_risk(
        self,
        setup: TradeSetup,
        pair: str,
        account_balance: float,
        config: PairConfig = None
    ) -> Tuple[bool, str, Dict]:
        """
        Comprehensive risk validation for a trade setup.
        
        Checks:
        1. Daily trade limit
        2. Risk-reward ratio
        3. Maximum risk exposure
        4. Position size validity
        
        Args:
            setup: Trade setup to validate
            pair: Currency pair
            account_balance: Current balance
            config: Pair configuration
            
        Returns:
            Tuple of (is_valid, reason, risk_metrics)
        """
        # Check daily limit
        if not self.check_daily_limit(pair):
            return (False, "Daily trade limit reached", {})
        
        # Get risk parameters
        risk_pct = config.max_spread_percent_of_target if config else self.default_risk_percent
        min_rr = config.default_risk_reward if config else TARGET_RISK_REWARD
        
        # Check RR ratio
        if not self.validate_risk_reward(setup, min_rr):
            return (False, f"RR {setup.risk_reward_ratio:.2f} below minimum {min_rr}", {})
        
        # Calculate position size
        sl_pips = self.get_stop_loss_pips(setup, pair)
        tp_pips = self.get_take_profit_pips(setup, pair)
        
        if sl_pips <= 0:
            return (False, "Invalid stop loss distance", {})
        
        position_size = self.calculate_position_size(
            account_balance, self.default_risk_percent, sl_pips, pair
        )
        
        if position_size <= 0:
            return (False, "Invalid position size calculated", {})
        
        # Calculate risk amount
        pip_value = PIP_VALUES.get(pair, 10.0)
        risk_amount = sl_pips * pip_value * position_size
        
        metrics = {
            'position_size': position_size,
            'stop_loss_pips': sl_pips,
            'take_profit_pips': tp_pips,
            'risk_amount': risk_amount,
            'risk_percent': risk_amount / account_balance,
            'potential_profit': tp_pips * pip_value * position_size,
            'risk_reward_ratio': setup.risk_reward_ratio
        }
        
        return (True, "Risk validation passed", metrics)
    
    def track_open_risk(self, pair: str, risk_amount: float) -> None:
        """Add risk amount for open position."""
        if pair not in self.open_risk:
            self.open_risk[pair] = 0.0
        self.open_risk[pair] += risk_amount
    
    def release_risk(self, pair: str, risk_amount: float) -> None:
        """Release risk amount when position closed."""
        if pair in self.open_risk:
            self.open_risk[pair] = max(0, self.open_risk[pair] - risk_amount)
    
    def get_total_risk_exposure(self) -> float:
        """Get total risk across all open positions."""
        return sum(self.open_risk.values())
    
    def get_pair_risk_exposure(self, pair: str) -> float:
        """Get risk for specific pair."""
        return self.open_risk.get(pair, 0.0)
