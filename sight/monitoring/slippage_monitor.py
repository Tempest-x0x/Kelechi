"""SIGHT Slippage Monitor - Spread and slippage guards."""
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date
from dataclasses import dataclass, field

from ..core.base import BaseLogger
from ..core.types import Trade, PairConfig
from ..core.constants import (
    SUPPORTED_PAIRS, MAX_SPREAD_PERCENT_OF_TARGET, TARGET_RISK_REWARD
)


@dataclass
class SlippageEvent:
    """Record of a slippage event."""
    timestamp: datetime
    pair: str
    expected_price: float
    actual_price: float
    slippage_pips: float
    spread_pips: float
    combined_cost_pips: float
    target_profit_pips: float
    cost_percent_of_target: float
    action_taken: str


@dataclass
class PairSlippageStats:
    """Slippage statistics for a single pair."""
    pair: str
    total_events: int = 0
    avg_slippage_pips: float = 0.0
    max_slippage_pips: float = 0.0
    avg_spread_pips: float = 0.0
    max_spread_pips: float = 0.0
    stop_count: int = 0
    events: List[SlippageEvent] = field(default_factory=list)


class SlippageMonitor(BaseLogger):
    """
    Slippage and Spread Guard for execution quality.
    
    Rule:
    If Spread + Slippage > 15% of 2.2R target profit:
    → Stop trading that pair for the day
    
    Purpose:
    - Monitor execution quality
    - Prevent trading during unfavorable conditions
    - Track slippage patterns by pair
    - Generate alerts for excessive costs
    """
    
    def __init__(
        self,
        max_cost_percent: float = MAX_SPREAD_PERCENT_OF_TARGET,
        target_rr: float = TARGET_RISK_REWARD
    ):
        super().__init__("SlippageMonitor")
        self.max_cost_percent = max_cost_percent
        self.target_rr = target_rr
        
        # Daily state tracking
        self.blocked_pairs: Dict[date, List[str]] = {}
        
        # Statistics by pair
        self.pair_stats: Dict[str, PairSlippageStats] = {
            pair: PairSlippageStats(pair=pair) for pair in SUPPORTED_PAIRS
        }
        
        # Event history
        self.events: List[SlippageEvent] = []
    
    def check_entry_conditions(
        self,
        pair: str,
        spread_pips: float,
        expected_slippage_pips: float,
        stop_loss_pips: float,
        config: PairConfig = None
    ) -> Tuple[bool, str]:
        """
        Check if entry conditions are acceptable.
        
        Calculation:
        1. Combined cost = Spread + Expected Slippage
        2. Target profit = Stop Loss × Target RR (2.2)
        3. Cost percent = Combined cost / Target profit
        
        If Cost percent > 15%: Block entry
        
        Args:
            pair: Currency pair
            spread_pips: Current bid-ask spread in pips
            expected_slippage_pips: Expected slippage in pips
            stop_loss_pips: Stop loss distance in pips
            config: Optional pair config for custom thresholds
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if pair is blocked today
        today = datetime.now().date()
        if today in self.blocked_pairs and pair in self.blocked_pairs[today]:
            return (False, f"{pair} blocked for today due to excessive slippage")
        
        # Calculate costs
        combined_cost = spread_pips + expected_slippage_pips
        target_rr = config.default_risk_reward if config else self.target_rr
        target_profit_pips = stop_loss_pips * target_rr
        
        if target_profit_pips <= 0:
            return (False, "Invalid stop loss distance")
        
        cost_percent = combined_cost / target_profit_pips
        max_percent = config.max_spread_percent_of_target if config else self.max_cost_percent
        
        if cost_percent > max_percent:
            reason = (f"Cost {cost_percent:.1%} exceeds {max_percent:.1%} threshold "
                     f"(Spread: {spread_pips:.1f}, Slip: {expected_slippage_pips:.1f}, "
                     f"Target: {target_profit_pips:.1f})")
            self.log_warning(reason)
            return (False, reason)
        
        return (True, f"Cost {cost_percent:.1%} acceptable")
    
    def record_slippage(
        self,
        pair: str,
        expected_price: float,
        actual_price: float,
        spread_pips: float,
        stop_loss_pips: float,
        is_long: bool
    ) -> SlippageEvent:
        """
        Record actual slippage from executed trade.
        
        Args:
            pair: Currency pair
            expected_price: Expected entry price
            actual_price: Actual filled price
            spread_pips: Spread at execution
            stop_loss_pips: Stop loss distance
            is_long: True if long trade
            
        Returns:
            SlippageEvent record
        """
        # Calculate slippage
        price_diff = actual_price - expected_price
        slippage_pips = abs(price_diff) * self._get_pip_multiplier(pair)
        
        # Determine if slippage was favorable or adverse
        if is_long:
            slippage_pips = slippage_pips if price_diff > 0 else -slippage_pips
        else:
            slippage_pips = slippage_pips if price_diff < 0 else -slippage_pips
        
        # Calculate cost metrics
        combined_cost = spread_pips + abs(slippage_pips)
        target_profit_pips = stop_loss_pips * self.target_rr
        cost_percent = combined_cost / target_profit_pips if target_profit_pips > 0 else 0
        
        # Determine action
        action = "ALLOWED"
        if cost_percent > self.max_cost_percent:
            action = "PAIR_BLOCKED"
            self._block_pair(pair)
        
        event = SlippageEvent(
            timestamp=datetime.now(),
            pair=pair,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_pips=slippage_pips,
            spread_pips=spread_pips,
            combined_cost_pips=combined_cost,
            target_profit_pips=target_profit_pips,
            cost_percent_of_target=cost_percent,
            action_taken=action
        )
        
        # Update statistics
        self._update_stats(pair, event)
        self.events.append(event)
        
        if action == "PAIR_BLOCKED":
            self.log_warning(f"BLOCKED {pair} for the day: cost {cost_percent:.1%} > {self.max_cost_percent:.1%}")
        
        return event
    
    def _block_pair(self, pair: str) -> None:
        """Block a pair for the current day."""
        today = datetime.now().date()
        
        if today not in self.blocked_pairs:
            self.blocked_pairs[today] = []
        
        if pair not in self.blocked_pairs[today]:
            self.blocked_pairs[today].append(pair)
            self.pair_stats[pair].stop_count += 1
    
    def _update_stats(self, pair: str, event: SlippageEvent) -> None:
        """Update pair statistics with new event."""
        stats = self.pair_stats[pair]
        stats.total_events += 1
        stats.events.append(event)
        
        # Update averages
        events = stats.events
        stats.avg_slippage_pips = sum(e.slippage_pips for e in events) / len(events)
        stats.avg_spread_pips = sum(e.spread_pips for e in events) / len(events)
        stats.max_slippage_pips = max(abs(e.slippage_pips) for e in events)
        stats.max_spread_pips = max(e.spread_pips for e in events)
    
    def _get_pip_multiplier(self, pair: str) -> float:
        """Get pip multiplier for pair (10000 for most, 100 for JPY)."""
        if 'JPY' in pair:
            return 100
        return 10000
    
    def is_pair_blocked(self, pair: str, check_date: date = None) -> bool:
        """Check if pair is blocked for trading."""
        check_date = check_date or datetime.now().date()
        return check_date in self.blocked_pairs and pair in self.blocked_pairs[check_date]
    
    def get_blocked_pairs(self, check_date: date = None) -> List[str]:
        """Get list of pairs blocked for the day."""
        check_date = check_date or datetime.now().date()
        return self.blocked_pairs.get(check_date, [])
    
    def reset_daily_blocks(self) -> None:
        """Reset daily blocked pairs (call at start of trading day)."""
        today = datetime.now().date()
        if today in self.blocked_pairs:
            del self.blocked_pairs[today]
        self.log_info("Daily pair blocks reset")
    
    def get_pair_stats(self, pair: str) -> PairSlippageStats:
        """Get slippage statistics for a pair."""
        return self.pair_stats.get(pair, PairSlippageStats(pair=pair))
    
    def get_execution_quality_score(self, pair: str) -> float:
        """
        Calculate execution quality score for a pair.
        
        Score: 0.0 (poor) to 1.0 (excellent)
        
        Based on:
        - Average slippage vs threshold
        - Average spread vs typical
        - Block frequency
        """
        stats = self.pair_stats.get(pair)
        if not stats or stats.total_events == 0:
            return 1.0  # No data, assume good
        
        # Penalize for high average slippage
        slippage_penalty = min(abs(stats.avg_slippage_pips) / 2.0, 1.0)
        
        # Penalize for high spread
        spread_penalty = min(stats.avg_spread_pips / 3.0, 1.0)
        
        # Penalize for blocks
        block_penalty = min(stats.stop_count / 5, 1.0)
        
        score = 1.0 - (slippage_penalty * 0.4 + spread_penalty * 0.3 + block_penalty * 0.3)
        
        return max(score, 0.0)
    
    def generate_report(self) -> Dict:
        """Generate slippage monitoring report."""
        today = datetime.now().date()
        
        # Today's events
        today_events = [e for e in self.events if e.timestamp.date() == today]
        
        # Pair rankings by execution quality
        pair_quality = {
            pair: self.get_execution_quality_score(pair)
            for pair in SUPPORTED_PAIRS
        }
        
        ranked_pairs = sorted(pair_quality.items(), key=lambda x: -x[1])
        
        return {
            'date': today.isoformat(),
            'blocked_pairs': self.get_blocked_pairs(),
            'total_events_today': len(today_events),
            'total_events_all_time': len(self.events),
            'pair_quality_ranking': [
                {'pair': p, 'score': f"{s:.2f}"} for p, s in ranked_pairs
            ],
            'pair_stats': {
                pair: {
                    'total_events': stats.total_events,
                    'avg_slippage': f"{stats.avg_slippage_pips:.2f}",
                    'max_slippage': f"{stats.max_slippage_pips:.2f}",
                    'avg_spread': f"{stats.avg_spread_pips:.2f}",
                    'blocks': stats.stop_count
                }
                for pair, stats in self.pair_stats.items()
                if stats.total_events > 0
            }
        }
