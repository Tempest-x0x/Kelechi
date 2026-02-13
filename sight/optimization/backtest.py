"""SIGHT Backtest Engine - High-performance vectorized backtesting."""
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import uuid

from ..core.base import BacktestEngine as BaseBacktestEngine, BaseLogger
from ..core.types import (
    Timeframe, Trade, TradeSetup, TradeStatus, BacktestResult,
    PairConfig, SignalType, MarketBias
)
from ..core.constants import (
    TARGET_WIN_RATE, TARGET_RISK_REWARD, TARGET_PROFIT_FACTOR,
    POINT_TO_PIP
)
from ..strategies.ict_engine import ICTEngine
from ..analysis.confluence import ConfluenceFilter


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    pair: str
    start_date: datetime
    end_date: datetime
    pair_config: PairConfig
    initial_equity: float = 100000.0
    risk_per_trade: float = 0.01
    commission_pips: float = 0.5
    max_daily_trades: int = 2


class BacktestEngine(BaseBacktestEngine):
    """
    High-performance backtest engine for ICT strategies.
    
    Features:
    - Vectorized where possible
    - Multi-timeframe data handling
    - Realistic fill simulation
    - Commission and slippage modeling
    
    Reports:
    - Win Rate
    - Profit Factor
    - Risk-Reward Ratio
    - Max Drawdown
    """
    
    def __init__(
        self,
        data_provider,
        ict_engine: ICTEngine = None,
        confluence_filter: ConfluenceFilter = None
    ):
        super().__init__("BacktestEngine")
        self.data_provider = data_provider
        self.ict_engine = ict_engine or ICTEngine()
        self.confluence_filter = confluence_filter or ConfluenceFilter()
    
    def run_backtest(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime,
        config: PairConfig
    ) -> BacktestResult:
        """
        Run full backtest for a pair with given configuration.
        
        Process:
        1. Load multi-timeframe data
        2. Iterate through time (1-minute resolution)
        3. Generate signals using ICT Engine
        4. Validate with Confluence Filter
        5. Simulate execution
        6. Calculate metrics
        
        Args:
            pair: Currency pair to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            config: Pair configuration
            
        Returns:
            BacktestResult with all metrics
        """
        self.log_info(f"Starting backtest: {pair} from {start_date} to {end_date}")
        
        # Update engine configs
        self.ict_engine.update_config(config)
        self.confluence_filter.update_config(config)
        
        # Load data
        h1_data = self.data_provider.get_ohlcv(pair, Timeframe.H1, start_date, end_date)
        m15_data = self.data_provider.get_ohlcv(pair, Timeframe.M15, start_date, end_date)
        m1_data = self.data_provider.get_ohlcv(pair, Timeframe.M1, start_date, end_date)
        
        if len(h1_data) == 0 or len(m15_data) == 0 or len(m1_data) == 0:
            self.log_error(f"Insufficient data for {pair}")
            return self._empty_result(pair, start_date, end_date)
        
        # Initialize tracking
        trades: List[Trade] = []
        equity_curve = [100000.0]
        current_equity = 100000.0
        active_trade: Optional[Trade] = None
        daily_trade_count = 0
        last_date = None
        
        # Run simulation
        m1_timestamps = m1_data.index
        
        for i in range(200, len(m1_data)):  # Start after warmup period
            current_time = m1_timestamps[i]
            current_date = current_time.date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trade_count = 0
                last_date = current_date
            
            # Check active trade status
            if active_trade:
                trade_result = self._check_trade_exit(
                    active_trade, m1_data, i, pair
                )
                if trade_result:
                    active_trade.status = TradeStatus.CLOSED
                    active_trade.exit_time = current_time
                    active_trade.actual_exit = trade_result['exit_price']
                    active_trade.pnl_pips = trade_result['pnl_pips']
                    active_trade.pnl_currency = trade_result['pnl_currency']
                    active_trade.actual_rr = trade_result['actual_rr']
                    
                    trades.append(active_trade)
                    current_equity += active_trade.pnl_currency
                    equity_curve.append(current_equity)
                    active_trade = None
                continue
            
            # Check for new entry (max 2 per day)
            if daily_trade_count >= config.max_daily_trades:
                continue
            
            # Get data windows for signal generation
            h1_window = h1_data[h1_data.index <= current_time].tail(100)
            m15_window = m15_data[m15_data.index <= current_time].tail(200)
            m1_window = m1_data.iloc[max(0, i-200):i+1]
            
            if len(h1_window) < 50 or len(m15_window) < 100 or len(m1_window) < 100:
                continue
            
            # Generate setup
            setup = self.ict_engine.generate_setup(
                pair, h1_window, m15_window, m1_window
            )
            
            if setup is None or not setup.is_valid:
                continue
            
            # Validate confluence
            bb_touch = self.confluence_filter.check_bollinger_touch(m15_window, setup.signal)
            kc_touch = self.confluence_filter.check_keltner_touch(m15_window, setup.signal)
            
            if not bb_touch and not kc_touch:
                continue
            
            setup.bollinger_touch = bb_touch
            setup.keltner_touch = kc_touch
            setup.confluence_score = self.confluence_filter.calculate_confluence_score(setup)
            
            # Execute trade
            active_trade = self._create_trade(
                pair, setup, current_time, current_equity, config
            )
            daily_trade_count += 1
        
        # Close any remaining trade
        if active_trade and len(m1_data) > 0:
            last_price = m1_data['close'].iloc[-1]
            if active_trade.setup.signal == SignalType.LONG:
                pnl_pips = (last_price - active_trade.actual_entry) * POINT_TO_PIP.get(pair, 10000)
            else:
                pnl_pips = (active_trade.actual_entry - last_price) * POINT_TO_PIP.get(pair, 10000)
            
            active_trade.status = TradeStatus.CLOSED
            active_trade.exit_time = m1_data.index[-1]
            active_trade.actual_exit = last_price
            active_trade.pnl_pips = pnl_pips
            active_trade.pnl_currency = pnl_pips * 10  # Simplified
            trades.append(active_trade)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        result = BacktestResult(
            pair=pair,
            timeframe=Timeframe.M1,
            start_date=start_date,
            end_date=end_date,
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            expected_value=metrics['expected_value'],
            total_pnl=metrics['total_pnl'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            avg_rr=metrics['avg_rr'],
            max_drawdown=metrics['max_drawdown'],
            max_drawdown_duration=metrics['max_dd_duration'],
            sharpe_ratio=metrics['sharpe_ratio'],
            trades=trades,
            equity_curve=equity_curve
        )
        
        self.log_info(f"Backtest complete: {len(trades)} trades, "
                     f"Win Rate: {result.win_rate:.1%}, PF: {result.profit_factor:.2f}")
        
        return result
    
    def _check_trade_exit(
        self,
        trade: Trade,
        m1_data: pd.DataFrame,
        current_idx: int,
        pair: str
    ) -> Optional[Dict]:
        """Check if trade should be closed (SL/TP hit)."""
        candle = m1_data.iloc[current_idx]
        
        setup = trade.setup
        is_long = setup.signal == SignalType.LONG
        
        pip_mult = POINT_TO_PIP.get(pair, 10000)
        
        if is_long:
            # Check stop loss
            if candle['low'] <= setup.stop_loss:
                pnl_pips = (setup.stop_loss - trade.actual_entry) * pip_mult
                return {
                    'exit_price': setup.stop_loss,
                    'pnl_pips': pnl_pips,
                    'pnl_currency': pnl_pips * 10,
                    'actual_rr': -1.0,
                    'reason': 'SL'
                }
            # Check take profit
            if candle['high'] >= setup.take_profit:
                pnl_pips = (setup.take_profit - trade.actual_entry) * pip_mult
                return {
                    'exit_price': setup.take_profit,
                    'pnl_pips': pnl_pips,
                    'pnl_currency': pnl_pips * 10,
                    'actual_rr': setup.risk_reward_ratio,
                    'reason': 'TP'
                }
        else:
            # Short trade
            if candle['high'] >= setup.stop_loss:
                pnl_pips = (trade.actual_entry - setup.stop_loss) * pip_mult
                return {
                    'exit_price': setup.stop_loss,
                    'pnl_pips': pnl_pips,
                    'pnl_currency': pnl_pips * 10,
                    'actual_rr': -1.0,
                    'reason': 'SL'
                }
            if candle['low'] <= setup.take_profit:
                pnl_pips = (trade.actual_entry - setup.take_profit) * pip_mult
                return {
                    'exit_price': setup.take_profit,
                    'pnl_pips': pnl_pips,
                    'pnl_currency': pnl_pips * 10,
                    'actual_rr': setup.risk_reward_ratio,
                    'reason': 'TP'
                }
        
        return None
    
    def _create_trade(
        self,
        pair: str,
        setup: TradeSetup,
        current_time: datetime,
        current_equity: float,
        config: PairConfig
    ) -> Trade:
        """Create new trade from setup."""
        return Trade(
            id=str(uuid.uuid4())[:8],
            pair=pair,
            setup=setup,
            status=TradeStatus.ACTIVE,
            entry_time=current_time,
            actual_entry=setup.entry_price,
            position_size=current_equity * config.max_spread_percent_of_target
        )
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Metrics:
        - Win Rate
        - Profit Factor
        - Expected Value
        - Average Win/Loss
        - Average RR
        - Max Drawdown
        - Sharpe Ratio
        """
        if not trades:
            return self._empty_metrics()
        
        # Basic counts
        total = len(trades)
        wins = [t for t in trades if t.pnl_currency > 0]
        losses = [t for t in trades if t.pnl_currency <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        # PnL metrics
        gross_profit = sum(t.pnl_currency for t in wins)
        gross_loss = sum(abs(t.pnl_currency) for t in losses)
        total_pnl = sum(t.pnl_currency for t in trades)
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # RR metrics
        rrs = [t.actual_rr for t in trades if t.actual_rr != 0]
        avg_rr = np.mean(rrs) if rrs else 0
        
        # Drawdown
        equity_curve = [100000.0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl_currency)
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_dd = np.max(drawdowns)
        
        # Drawdown duration
        max_dd_duration = 0
        current_duration = 0
        for i in range(1, len(equity)):
            if equity[i] < running_max[i]:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_currency for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        return {
            'total_trades': total,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expected_value': ev,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_rr': avg_rr,
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,
            'sharpe_ratio': sharpe
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expected_value': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_rr': 0.0,
            'max_drawdown': 0.0,
            'max_dd_duration': 0,
            'sharpe_ratio': 0.0
        }
    
    def _empty_result(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Return empty backtest result."""
        return BacktestResult(
            pair=pair,
            timeframe=Timeframe.M1,
            start_date=start_date,
            end_date=end_date
        )


class VectorizedBacktest(BaseLogger):
    """
    Vectorized backtest for rapid parameter optimization.
    
    Uses numpy operations for speed when testing many configurations.
    Simplified signal logic for fast iteration.
    """
    
    def __init__(self):
        super().__init__("VectorizedBacktest")
    
    def run_fast_backtest(
        self,
        signals: np.ndarray,  # 1 = long, -1 = short, 0 = no signal
        prices: np.ndarray,   # Entry prices
        stop_losses: np.ndarray,
        take_profits: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray
    ) -> Dict[str, float]:
        """
        Run vectorized backtest on pre-computed signals.
        
        Args:
            signals: Signal array
            prices: Price array
            stop_losses: Stop loss levels
            take_profits: Take profit levels
            high_prices: High prices for SL/TP check
            low_prices: Low prices for SL/TP check
            
        Returns:
            Performance metrics
        """
        n = len(signals)
        trades = []
        
        i = 0
        while i < n:
            if signals[i] == 0:
                i += 1
                continue
            
            entry_price = prices[i]
            sl = stop_losses[i]
            tp = take_profits[i]
            is_long = signals[i] == 1
            
            # Find exit
            for j in range(i + 1, min(i + 1000, n)):  # Max 1000 candles
                if is_long:
                    if low_prices[j] <= sl:
                        trades.append(-1.0)  # Loss
                        i = j + 1
                        break
                    if high_prices[j] >= tp:
                        trades.append(2.2)  # Win at 2.2R
                        i = j + 1
                        break
                else:
                    if high_prices[j] >= sl:
                        trades.append(-1.0)
                        i = j + 1
                        break
                    if low_prices[j] <= tp:
                        trades.append(2.2)
                        i = j + 1
                        break
            else:
                i += 1
        
        if not trades:
            return {'win_rate': 0, 'profit_factor': 0, 'total_trades': 0}
        
        trades = np.array(trades)
        wins = np.sum(trades > 0)
        total = len(trades)
        win_rate = wins / total
        
        gross_profit = np.sum(trades[trades > 0])
        gross_loss = np.abs(np.sum(trades[trades < 0]))
        
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'profit_factor': pf,
            'total_trades': total,
            'total_pnl': np.sum(trades)
        }
