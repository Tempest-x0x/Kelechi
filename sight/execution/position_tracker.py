"""SIGHT Position Tracker - Track and manage open positions."""
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from ..core.base import BaseLogger
from ..core.types import Trade, TradeStatus, SignalType


@dataclass
class PositionSummary:
    """Summary of a position."""
    pair: str
    direction: str
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pips: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    duration_minutes: int


class PositionTracker(BaseLogger):
    """
    Real-time position tracking and monitoring.
    
    Tracks:
    - Open positions by pair
    - Unrealized PnL
    - Position duration
    - Risk exposure
    """
    
    def __init__(self):
        super().__init__("PositionTracker")
        self.positions: Dict[str, Trade] = {}
        self.current_prices: Dict[str, float] = {}
    
    def add_position(self, trade: Trade) -> None:
        """Add new position to tracker."""
        if trade.pair in self.positions:
            self.log_warning(f"Position already exists for {trade.pair}")
            return
        
        self.positions[trade.pair] = trade
        self.log_info(f"Position added: {trade.pair} {trade.setup.signal.name}")
    
    def remove_position(self, pair: str) -> Optional[Trade]:
        """Remove position from tracker."""
        if pair not in self.positions:
            return None
        
        trade = self.positions.pop(pair)
        self.log_info(f"Position removed: {pair}")
        return trade
    
    def update_price(self, pair: str, price: float) -> None:
        """Update current price for a pair."""
        self.current_prices[pair] = price
    
    def get_position(self, pair: str) -> Optional[Trade]:
        """Get position for pair."""
        return self.positions.get(pair)
    
    def has_position(self, pair: str) -> bool:
        """Check if position exists for pair."""
        return pair in self.positions
    
    def get_position_summary(self, pair: str) -> Optional[PositionSummary]:
        """Get detailed summary of a position."""
        trade = self.positions.get(pair)
        if not trade:
            return None
        
        current_price = self.current_prices.get(pair, trade.actual_entry)
        
        # Calculate unrealized PnL
        if trade.setup.signal == SignalType.LONG:
            pnl_points = current_price - trade.actual_entry
        else:
            pnl_points = trade.actual_entry - current_price
        
        pnl_pips = pnl_points * 10000  # Simplified
        pnl_currency = pnl_pips * 10 * trade.position_size
        
        # Calculate duration
        duration = datetime.now() - trade.entry_time
        duration_minutes = int(duration.total_seconds() / 60)
        
        return PositionSummary(
            pair=pair,
            direction="LONG" if trade.setup.signal == SignalType.LONG else "SHORT",
            entry_price=trade.actual_entry,
            current_price=current_price,
            position_size=trade.position_size,
            unrealized_pnl=pnl_currency,
            unrealized_pnl_pips=pnl_pips,
            stop_loss=trade.setup.stop_loss,
            take_profit=trade.setup.take_profit,
            entry_time=trade.entry_time,
            duration_minutes=duration_minutes
        )
    
    def get_all_positions(self) -> List[PositionSummary]:
        """Get summaries of all positions."""
        summaries = []
        for pair in self.positions:
            summary = self.get_position_summary(pair)
            if summary:
                summaries.append(summary)
        return summaries
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        total = 0.0
        for pair in self.positions:
            summary = self.get_position_summary(pair)
            if summary:
                total += summary.unrealized_pnl
        return total
    
    def get_exposure_by_direction(self) -> Dict[str, float]:
        """Get total exposure by direction."""
        long_exposure = 0.0
        short_exposure = 0.0
        
        for trade in self.positions.values():
            if trade.setup.signal == SignalType.LONG:
                long_exposure += trade.position_size
            else:
                short_exposure += trade.position_size
        
        return {
            'long': long_exposure,
            'short': short_exposure,
            'net': long_exposure - short_exposure
        }
    
    def format_position_display(self) -> str:
        """Format positions for display."""
        if not self.positions:
            return "No open positions"
        
        lines = ["OPEN POSITIONS", "-" * 50]
        
        for summary in self.get_all_positions():
            pnl_str = f"+{summary.unrealized_pnl:.2f}" if summary.unrealized_pnl >= 0 else f"{summary.unrealized_pnl:.2f}"
            lines.append(
                f"{summary.pair:8s} {summary.direction:5s} "
                f"@ {summary.entry_price:.5f} → {summary.current_price:.5f} "
                f"PnL: {pnl_str} ({summary.duration_minutes}min)"
            )
        
        total_pnl = self.get_total_unrealized_pnl()
        lines.append("-" * 50)
        lines.append(f"Total Unrealized PnL: ${total_pnl:+.2f}")
        
        return "\n".join(lines)
