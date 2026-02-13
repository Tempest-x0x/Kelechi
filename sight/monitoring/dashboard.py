"""SIGHT Dashboard Metrics - Live performance visualization."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..core.base import BaseLogger
from ..core.types import Trade, TradeStatus
from ..core.constants import TARGET_WIN_RATE, TARGET_PROFIT_FACTOR, TARGET_RISK_REWARD


@dataclass
class DashboardSnapshot:
    """Point-in-time dashboard metrics."""
    timestamp: datetime
    
    # Core Metrics
    win_rate: float
    profit_factor: float
    z_score: float
    expected_value: float
    
    # Target Comparison
    win_rate_vs_target: float
    pf_vs_target: float
    rr_vs_benchmark: float
    
    # Equity
    current_equity: float
    equity_peak: float
    current_drawdown: float
    
    # Activity
    trades_today: int
    trades_this_week: int
    active_positions: int
    
    # Health Indicators
    is_healthy: bool
    alerts: List[str]


class DashboardMetrics(BaseLogger):
    """
    Live dashboard metrics for SIGHT trading engine.
    
    Display Metrics (as specified):
    - Live Win Rate (target 70%)
    - Profit Factor (>2.0)
    - Z-Score
    - Equity Curve vs 1:2.2 Benchmark
    
    Additional Metrics:
    - Expected Value
    - Current Drawdown
    - Active Positions
    - System Health
    """
    
    def __init__(self, starting_equity: float = 100000.0):
        super().__init__("DashboardMetrics")
        self.starting_equity = starting_equity
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), starting_equity)]
        self.benchmark_curve: List[Tuple[datetime, float]] = [(datetime.now(), starting_equity)]
        self.trade_history: List[Trade] = []
        self.active_trades: Dict[str, Trade] = {}
    
    def update_trade(self, trade: Trade) -> None:
        """Update dashboard with new/modified trade."""
        if trade.status == TradeStatus.ACTIVE:
            self.active_trades[trade.id] = trade
        elif trade.status == TradeStatus.CLOSED:
            if trade.id in self.active_trades:
                del self.active_trades[trade.id]
            self.trade_history.append(trade)
            self._update_equity(trade)
            self._update_benchmark(trade)
    
    def _update_equity(self, trade: Trade) -> None:
        """Update equity curve with completed trade."""
        last_equity = self.equity_curve[-1][1]
        new_equity = last_equity + trade.pnl_currency
        self.equity_curve.append((trade.exit_time or datetime.now(), new_equity))
    
    def _update_benchmark(self, trade: Trade) -> None:
        """
        Update benchmark curve (1:2.2 RR strategy).
        
        Benchmark assumes:
        - Same trades taken
        - Fixed 1:2.2 RR on all trades
        - Standard risk per trade
        """
        last_benchmark = self.benchmark_curve[-1][1]
        risk_amount = last_benchmark * 0.01  # 1% risk
        
        if trade.pnl_currency > 0:
            # Win at 2.2R
            benchmark_pnl = risk_amount * TARGET_RISK_REWARD
        else:
            # Loss at 1R
            benchmark_pnl = -risk_amount
        
        new_benchmark = last_benchmark + benchmark_pnl
        self.benchmark_curve.append((trade.exit_time or datetime.now(), new_benchmark))
    
    def calculate_win_rate(self, trades: List[Trade] = None) -> float:
        """Calculate current win rate."""
        trades = trades or self.trade_history
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t.pnl_currency > 0)
        return wins / len(trades)
    
    def calculate_profit_factor(self, trades: List[Trade] = None) -> float:
        """Calculate profit factor."""
        trades = trades or self.trade_history
        if not trades:
            return 0.0
        
        gross_profit = sum(t.pnl_currency for t in trades if t.pnl_currency > 0)
        gross_loss = sum(abs(t.pnl_currency) for t in trades if t.pnl_currency < 0)
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def calculate_z_score(self, trades: List[Trade] = None) -> float:
        """Calculate Z-Score for streak dependency."""
        trades = trades or self.trade_history
        
        if len(trades) < 10:
            return 0.0
        
        results = [1 if t.pnl_currency > 0 else 0 for t in trades]
        n = len(results)
        wins = sum(results)
        losses = n - wins
        
        if wins == 0 or losses == 0:
            return 0.0
        
        # Count runs
        r = 1
        for i in range(1, n):
            if results[i] != results[i-1]:
                r += 1
        
        p = 2 * wins * losses
        
        import numpy as np
        numerator = n * (r - 0.5) - p
        denominator_inner = (p * (p - n)) / (n - 1)
        if denominator_inner < 0:
            denominator_inner = abs(denominator_inner)
        denominator = np.sqrt(denominator_inner)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def calculate_expected_value(self, trades: List[Trade] = None) -> float:
        """Calculate trade expectancy."""
        trades = trades or self.trade_history
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t.pnl_currency > 0]
        losses = [t for t in trades if t.pnl_currency <= 0]
        
        win_pct = len(wins) / len(trades)
        avg_win = sum(t.pnl_currency for t in wins) / len(wins) if wins else 0
        avg_loss = sum(abs(t.pnl_currency) for t in losses) / len(losses) if losses else 0
        
        return (win_pct * avg_win) - ((1 - win_pct) * avg_loss)
    
    def get_current_drawdown(self) -> Tuple[float, float]:
        """Get current drawdown and peak equity."""
        if not self.equity_curve:
            return (0.0, self.starting_equity)
        
        equities = [e[1] for e in self.equity_curve]
        peak = max(equities)
        current = equities[-1]
        
        drawdown = (peak - current) / peak if peak > 0 else 0.0
        
        return (drawdown, peak)
    
    def get_trades_count(self, period: str = 'today') -> int:
        """Get trade count for period."""
        now = datetime.now()
        
        if period == 'today':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return len(self.trade_history)
        
        return sum(1 for t in self.trade_history 
                   if t.exit_time and t.exit_time >= start)
    
    def get_snapshot(self) -> DashboardSnapshot:
        """Get current dashboard snapshot."""
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()
        z_score = self.calculate_z_score()
        ev = self.calculate_expected_value()
        
        drawdown, peak = self.get_current_drawdown()
        current_equity = self.equity_curve[-1][1] if self.equity_curve else self.starting_equity
        
        # Calculate vs targets
        win_rate_diff = win_rate - TARGET_WIN_RATE
        pf_diff = profit_factor - TARGET_PROFIT_FACTOR
        
        # Calculate RR vs benchmark
        if len(self.equity_curve) > 1 and len(self.benchmark_curve) > 1:
            actual_return = (self.equity_curve[-1][1] - self.starting_equity) / self.starting_equity
            bench_return = (self.benchmark_curve[-1][1] - self.starting_equity) / self.starting_equity
            rr_vs_bench = actual_return - bench_return
        else:
            rr_vs_bench = 0.0
        
        # Health check
        alerts = []
        is_healthy = True
        
        if win_rate < 0.60:
            alerts.append(f"Win rate {win_rate:.1%} below 60%")
            is_healthy = False
        
        if profit_factor < 1.5:
            alerts.append(f"Profit factor {profit_factor:.2f} below 1.5")
            is_healthy = False
        
        if abs(z_score) > 1.96:
            alerts.append(f"Z-Score {z_score:.2f} indicates streak dependency")
        
        if drawdown > 0.15:
            alerts.append(f"Drawdown {drawdown:.1%} exceeds 15%")
            is_healthy = False
        
        if ev < 1.0 and len(self.trade_history) >= 20:
            alerts.append(f"Expected value {ev:.2f} below threshold")
            is_healthy = False
        
        return DashboardSnapshot(
            timestamp=datetime.now(),
            win_rate=win_rate,
            profit_factor=profit_factor,
            z_score=z_score,
            expected_value=ev,
            win_rate_vs_target=win_rate_diff,
            pf_vs_target=pf_diff,
            rr_vs_benchmark=rr_vs_bench,
            current_equity=current_equity,
            equity_peak=peak,
            current_drawdown=drawdown,
            trades_today=self.get_trades_count('today'),
            trades_this_week=self.get_trades_count('week'),
            active_positions=len(self.active_trades),
            is_healthy=is_healthy,
            alerts=alerts
        )
    
    def format_dashboard(self, snapshot: DashboardSnapshot = None) -> str:
        """Format dashboard as text display."""
        s = snapshot or self.get_snapshot()
        
        # Status indicator
        status = "✅ HEALTHY" if s.is_healthy else "⚠️ WARNING"
        
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║               SIGHT TRADING ENGINE DASHBOARD              ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Status: {status:<48}║",
            f"║  Time: {s.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<50}║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  PERFORMANCE METRICS                                      ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Win Rate:      {s.win_rate:>6.1%}  (Target: {TARGET_WIN_RATE:.0%})  {self._delta_str(s.win_rate_vs_target)}            ║",
            f"║  Profit Factor: {s.profit_factor:>6.2f}  (Target: {TARGET_PROFIT_FACTOR:.1f})  {self._delta_str(s.pf_vs_target, fmt='.2f')}             ║",
            f"║  Z-Score:       {s.z_score:>6.2f}  (< 1.96 = OK)                    ║",
            f"║  Expected Value:{s.expected_value:>7.2f}  (> 1.0 = OK)                     ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  EQUITY STATUS                                            ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Current Equity:  ${s.current_equity:>12,.2f}                       ║",
            f"║  Equity Peak:     ${s.equity_peak:>12,.2f}                       ║",
            f"║  Drawdown:        {s.current_drawdown:>6.2%}                                 ║",
            f"║  vs Benchmark:    {self._delta_str(s.rr_vs_benchmark)}                                ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  ACTIVITY                                                 ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Trades Today:    {s.trades_today:>3}                                       ║",
            f"║  Trades This Week:{s.trades_this_week:>3}                                       ║",
            f"║  Active Positions:{s.active_positions:>3}                                       ║",
        ]
        
        if s.alerts:
            lines.extend([
                "╠══════════════════════════════════════════════════════════╣",
                "║  ALERTS                                                   ║",
                "╠══════════════════════════════════════════════════════════╣",
            ])
            for alert in s.alerts[:5]:  # Max 5 alerts
                lines.append(f"║  ⚠ {alert:<54}║")
        
        lines.append("╚══════════════════════════════════════════════════════════╝")
        
        return "\n".join(lines)
    
    def _delta_str(self, value: float, fmt: str = '.1%') -> str:
        """Format delta with +/- prefix."""
        if value > 0:
            return f"+{value:{fmt}}"
        else:
            return f"{value:{fmt}}"
    
    def get_equity_curve_data(self) -> Dict:
        """Get equity curve data for plotting."""
        return {
            'timestamps': [e[0].isoformat() for e in self.equity_curve],
            'equity': [e[1] for e in self.equity_curve],
            'benchmark': [b[1] for b in self.benchmark_curve]
        }
    
    def export_metrics(self) -> Dict:
        """Export all metrics as JSON-serializable dict."""
        snapshot = self.get_snapshot()
        
        return {
            'timestamp': snapshot.timestamp.isoformat(),
            'metrics': {
                'win_rate': snapshot.win_rate,
                'profit_factor': snapshot.profit_factor,
                'z_score': snapshot.z_score,
                'expected_value': snapshot.expected_value
            },
            'targets': {
                'win_rate_target': TARGET_WIN_RATE,
                'win_rate_vs_target': snapshot.win_rate_vs_target,
                'profit_factor_target': TARGET_PROFIT_FACTOR,
                'profit_factor_vs_target': snapshot.pf_vs_target
            },
            'equity': {
                'current': snapshot.current_equity,
                'peak': snapshot.equity_peak,
                'drawdown': snapshot.current_drawdown,
                'vs_benchmark': snapshot.rr_vs_benchmark
            },
            'activity': {
                'trades_today': snapshot.trades_today,
                'trades_this_week': snapshot.trades_this_week,
                'active_positions': snapshot.active_positions
            },
            'health': {
                'is_healthy': snapshot.is_healthy,
                'alerts': snapshot.alerts
            }
        }
    
    def to_json(self) -> str:
        """Export metrics as JSON string."""
        return json.dumps(self.export_metrics(), indent=2)
