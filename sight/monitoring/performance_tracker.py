"""SIGHT Performance Tracker - Z-Score and Expectancy monitoring."""
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.base import PerformanceMonitor
from ..core.types import Trade, TradeStatus
from ..core.constants import (
    Z_SCORE_THRESHOLD, CONFIDENCE_WARNING_THRESHOLD,
    LOOKBACK_TRADES_FOR_CONFIDENCE, MIN_EXPECTED_VALUE,
    TARGET_WIN_RATE, TARGET_PROFIT_FACTOR
)


@dataclass
class PerformanceReport:
    """Performance metrics report."""
    timestamp: datetime
    total_trades: int
    win_rate: float
    profit_factor: float
    expected_value: float
    z_score: float
    max_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    is_healthy: bool
    warnings: List[str]
    recommendations: List[str]


class PerformanceTracker(PerformanceMonitor):
    """
    Performance tracking with Z-Score and Expectancy analysis.
    
    Key Metrics:
    1. Z-Score - Streak dependency detection
    2. Expectancy (EV) - Daily performance audit
    3. Confidence monitoring - Rolling win rate
    4. Drawdown tracking - Risk management
    
    Safety Rules:
    - Z > 1.96: Significant streak dependency (95% confidence)
    - Win rate < 60% over 50 trades: Confidence warning
    - EV < 1.0: Pause trading, generate degradation report
    """
    
    def __init__(self):
        super().__init__("PerformanceTracker")
        self.trade_history: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_performance: Dict[str, Dict] = {}
        self.starting_equity: float = 100000.0
    
    def add_trade(self, trade: Trade) -> None:
        """Add completed trade to history."""
        if trade.status == TradeStatus.CLOSED:
            self.trade_history.append(trade)
            self._update_equity_curve(trade)
            self._update_daily_performance(trade)
    
    def _update_equity_curve(self, trade: Trade) -> None:
        """Update equity curve with trade result."""
        if not self.equity_curve:
            self.equity_curve.append(self.starting_equity)
        
        last_equity = self.equity_curve[-1]
        new_equity = last_equity + trade.pnl_currency
        self.equity_curve.append(new_equity)
    
    def _update_daily_performance(self, trade: Trade) -> None:
        """Update daily performance tracking."""
        if trade.exit_time is None:
            return
        
        date_key = trade.exit_time.strftime("%Y-%m-%d")
        
        if date_key not in self.daily_performance:
            self.daily_performance[date_key] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }
        
        day = self.daily_performance[date_key]
        day['trades'] += 1
        day['pnl'] += trade.pnl_currency
        
        if trade.pnl_currency > 0:
            day['wins'] += 1
            day['gross_profit'] += trade.pnl_currency
        else:
            day['losses'] += 1
            day['gross_loss'] += abs(trade.pnl_currency)
    
    def calculate_z_score(self, trades: List[Trade] = None) -> float:
        """
        Calculate Z-Score for streak dependency analysis.
        
        Z-Score Formula:
        Z = (N × (R - 0.5) - P) / sqrt((P × (P - N)) / (N - 1))
        
        Where:
        - N = total trades
        - R = streak count (number of streaks)
        - P = 2 × wins × losses
        
        Interpretation:
        - Z > 1.96: Significant positive streak dependency (95% CI)
        - Z < -1.96: Significant negative streak dependency
        - |Z| <= 1.96: No significant streak dependency
        
        Args:
            trades: List of trades (default: all history)
            
        Returns:
            Z-Score value
        """
        trades = trades or self.trade_history
        
        if len(trades) < 10:
            self.log_debug("Insufficient trades for Z-Score calculation")
            return 0.0
        
        # Count wins and losses
        results = [1 if t.pnl_currency > 0 else 0 for t in trades]
        n = len(results)
        wins = sum(results)
        losses = n - wins
        
        if wins == 0 or losses == 0:
            return 0.0
        
        # Count streaks (runs)
        r = 1  # Start with 1 streak
        for i in range(1, n):
            if results[i] != results[i-1]:
                r += 1
        
        # Calculate P
        p = 2 * wins * losses
        
        # Calculate Z-Score
        numerator = n * (r - 0.5) - p
        
        denominator_inner = (p * (p - n)) / (n - 1)
        if denominator_inner < 0:
            denominator_inner = abs(denominator_inner)
        
        denominator = np.sqrt(denominator_inner)
        
        if denominator == 0:
            return 0.0
        
        z_score = numerator / denominator
        
        self.log_debug(f"Z-Score: {z_score:.4f} (N={n}, R={r}, W={wins}, L={losses})")
        
        return z_score
    
    def calculate_expectancy(self, trades: List[Trade] = None) -> float:
        """
        Calculate trade expectancy (Expected Value).
        
        EV Formula:
        EV = (Win% × AvgWin) - (Loss% × AvgLoss)
        
        Rules:
        - EV < 1.0: Pause trading
        - Output "Performance Degradation Report"
        
        Args:
            trades: List of trades (default: all history)
            
        Returns:
            Expected Value
        """
        trades = trades or self.trade_history
        
        if len(trades) < 5:
            return 0.0
        
        wins = [t for t in trades if t.pnl_currency > 0]
        losses = [t for t in trades if t.pnl_currency <= 0]
        
        total = len(trades)
        win_pct = len(wins) / total
        loss_pct = len(losses) / total
        
        avg_win = np.mean([t.pnl_currency for t in wins]) if wins else 0.0
        avg_loss = np.mean([abs(t.pnl_currency) for t in losses]) if losses else 0.0
        
        ev = (win_pct * avg_win) - (loss_pct * avg_loss)
        
        self.log_debug(f"Expectancy: {ev:.2f} (Win%={win_pct:.2%}, AvgWin={avg_win:.2f})")
        
        return ev
    
    def check_confidence(
        self,
        trades: List[Trade] = None,
        lookback: int = LOOKBACK_TRADES_FOR_CONFIDENCE
    ) -> Tuple[bool, str]:
        """
        Check trading confidence based on recent performance.
        
        Rule: If win rate over last 50 trades < 60%, log "Confidence Warning"
        
        Args:
            trades: List of trades
            lookback: Number of recent trades to analyze
            
        Returns:
            Tuple of (is_confident, message)
        """
        trades = trades or self.trade_history
        
        if len(trades) < lookback:
            return (True, f"Insufficient data ({len(trades)}/{lookback} trades)")
        
        recent_trades = trades[-lookback:]
        wins = sum(1 for t in recent_trades if t.pnl_currency > 0)
        win_rate = wins / len(recent_trades)
        
        if win_rate < CONFIDENCE_WARNING_THRESHOLD:
            message = f"Confidence Warning: Win rate {win_rate:.1%} < {CONFIDENCE_WARNING_THRESHOLD:.1%} over last {lookback} trades"
            self.log_warning(message)
            return (False, message)
        
        return (True, f"Confidence OK: {win_rate:.1%} win rate over last {lookback} trades")
    
    def calculate_drawdown(self, equity_curve: List[float] = None) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: Equity values (default: internal curve)
            
        Returns:
            Tuple of (max_drawdown_percent, max_duration_periods)
        """
        curve = equity_curve or self.equity_curve
        
        if len(curve) < 2:
            return (0.0, 0)
        
        equity = np.array(curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        
        max_dd = np.max(drawdowns)
        
        # Calculate duration
        max_duration = 0
        current_duration = 0
        
        for i in range(1, len(equity)):
            if equity[i] < running_max[i]:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return (max_dd, max_duration)
    
    def calculate_profit_factor(self, trades: List[Trade] = None) -> float:
        """
        Calculate profit factor.
        
        Profit Factor = Gross Profit / Gross Loss
        
        Target: > 2.0
        
        Args:
            trades: List of trades
            
        Returns:
            Profit factor
        """
        trades = trades or self.trade_history
        
        gross_profit = sum(t.pnl_currency for t in trades if t.pnl_currency > 0)
        gross_loss = sum(abs(t.pnl_currency) for t in trades if t.pnl_currency < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_streak_info(self, trades: List[Trade] = None) -> Dict[str, int]:
        """
        Get current and maximum winning/losing streaks.
        
        Args:
            trades: List of trades
            
        Returns:
            Dict with streak information
        """
        trades = trades or self.trade_history
        
        if not trades:
            return {
                'current_streak': 0,
                'current_streak_type': 'none',
                'max_win_streak': 0,
                'max_loss_streak': 0
            }
        
        results = [1 if t.pnl_currency > 0 else -1 for t in trades]
        
        # Current streak
        current_streak = 1
        for i in range(len(results) - 1, 0, -1):
            if results[i] == results[i-1]:
                current_streak += 1
            else:
                break
        
        current_type = 'win' if results[-1] == 1 else 'loss'
        
        # Max streaks
        max_win = 0
        max_loss = 0
        streak = 1
        
        for i in range(1, len(results)):
            if results[i] == results[i-1]:
                streak += 1
            else:
                if results[i-1] == 1:
                    max_win = max(max_win, streak)
                else:
                    max_loss = max(max_loss, streak)
                streak = 1
        
        # Final streak
        if results[-1] == 1:
            max_win = max(max_win, streak)
        else:
            max_loss = max(max_loss, streak)
        
        return {
            'current_streak': current_streak,
            'current_streak_type': current_type,
            'max_win_streak': max_win,
            'max_loss_streak': max_loss
        }
    
    def generate_daily_audit(self, date: datetime = None) -> Dict:
        """
        Generate daily expectancy audit.
        
        Rule: If EV < 1.0, pause trading and output report.
        
        Args:
            date: Date to audit (default: today)
            
        Returns:
            Daily audit report
        """
        date = date or datetime.now()
        date_key = date.strftime("%Y-%m-%d")
        
        if date_key not in self.daily_performance:
            return {'date': date_key, 'status': 'no_trades'}
        
        day = self.daily_performance[date_key]
        
        # Calculate daily EV
        if day['trades'] > 0:
            win_pct = day['wins'] / day['trades']
            avg_win = day['gross_profit'] / day['wins'] if day['wins'] > 0 else 0
            avg_loss = day['gross_loss'] / day['losses'] if day['losses'] > 0 else 0
            ev = (win_pct * avg_win) - ((1 - win_pct) * avg_loss)
        else:
            ev = 0.0
        
        # Calculate daily profit factor
        if day['gross_loss'] > 0:
            pf = day['gross_profit'] / day['gross_loss']
        else:
            pf = float('inf') if day['gross_profit'] > 0 else 0.0
        
        status = 'healthy' if ev >= MIN_EXPECTED_VALUE else 'degraded'
        
        audit = {
            'date': date_key,
            'trades': day['trades'],
            'wins': day['wins'],
            'losses': day['losses'],
            'pnl': day['pnl'],
            'win_rate': day['wins'] / day['trades'] if day['trades'] > 0 else 0,
            'profit_factor': pf,
            'expected_value': ev,
            'status': status
        }
        
        if status == 'degraded':
            self.log_warning(f"Performance Degradation Report - {date_key}: EV={ev:.2f}")
            audit['recommendation'] = 'Pause trading until system review'
        
        return audit
    
    def generate_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report.
        
        Returns:
            PerformanceReport with all metrics
        """
        trades = self.trade_history
        warnings = []
        recommendations = []
        
        # Basic metrics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl_currency > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        # Advanced metrics
        profit_factor = self.calculate_profit_factor(trades)
        expected_value = self.calculate_expectancy(trades)
        z_score = self.calculate_z_score(trades)
        max_dd, dd_duration = self.calculate_drawdown()
        streak_info = self.get_streak_info(trades)
        
        # Check conditions
        is_confident, confidence_msg = self.check_confidence(trades)
        
        is_healthy = True
        
        # Z-Score check
        if abs(z_score) > Z_SCORE_THRESHOLD:
            warnings.append(f"Z-Score {z_score:.2f} indicates streak dependency")
            recommendations.append("Review position sizing and risk management")
        
        # Confidence check
        if not is_confident:
            warnings.append(confidence_msg)
            recommendations.append("Consider reducing position sizes")
            is_healthy = False
        
        # EV check
        if expected_value < MIN_EXPECTED_VALUE:
            warnings.append(f"Expected Value {expected_value:.2f} below threshold")
            recommendations.append("PAUSE TRADING - System review required")
            is_healthy = False
        
        # Win rate check
        if win_rate < TARGET_WIN_RATE:
            warnings.append(f"Win rate {win_rate:.1%} below target {TARGET_WIN_RATE:.1%}")
        
        # Profit factor check
        if profit_factor < TARGET_PROFIT_FACTOR:
            warnings.append(f"Profit factor {profit_factor:.2f} below target {TARGET_PROFIT_FACTOR}")
        
        return PerformanceReport(
            timestamp=datetime.now(),
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expected_value=expected_value,
            z_score=z_score,
            max_drawdown=max_dd,
            consecutive_wins=streak_info['max_win_streak'],
            consecutive_losses=streak_info['max_loss_streak'],
            is_healthy=is_healthy,
            warnings=warnings,
            recommendations=recommendations
        )
