"""SIGHT Stress Test - Monte Carlo simulation for risk assessment."""
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from ..core.base import BaseLogger
from ..core.types import Trade
from ..core.constants import MONTE_CARLO_ITERATIONS, MONTE_CARLO_TRADE_SAMPLE


@dataclass
class StressTestResult:
    """Monte Carlo stress test results."""
    timestamp: datetime
    n_trades: int
    n_simulations: int
    
    # Probability of Ruin
    probability_of_ruin: float
    ruin_threshold: float
    
    # Drawdown Analysis
    median_max_drawdown: float
    drawdown_95_percentile: float
    drawdown_99_percentile: float
    
    # Return Distribution
    median_final_equity: float
    equity_5_percentile: float
    equity_95_percentile: float
    
    # Confidence Intervals
    ci_95_lower: float
    ci_95_upper: float
    
    # Summary
    risk_rating: str
    recommendations: List[str]


class StressTest(BaseLogger):
    """
    Monte Carlo Stress Test for portfolio risk assessment.
    
    Purpose:
    - Simulate thousands of possible equity paths
    - Calculate Probability of Ruin
    - Determine 95% Confidence Interval for drawdown
    
    Process:
    1. Take last 100 trades
    2. Shuffle 10,000 times
    3. Simulate 10,000 equity paths
    4. Calculate statistics
    
    Output:
    - Probability of Ruin
    - 95% Drawdown Confidence Interval
    """
    
    def __init__(
        self,
        n_simulations: int = MONTE_CARLO_ITERATIONS,
        trade_sample: int = MONTE_CARLO_TRADE_SAMPLE,
        ruin_threshold: float = 0.50  # 50% drawdown = ruin
    ):
        super().__init__("StressTest")
        self.n_simulations = n_simulations
        self.trade_sample = trade_sample
        self.ruin_threshold = ruin_threshold
    
    def run_stress_test(
        self,
        trades: List[Trade],
        starting_equity: float = 100000.0
    ) -> StressTestResult:
        """
        Run Monte Carlo stress test on trade history.
        
        Algorithm:
        1. Extract trade returns (PnL as % of equity)
        2. For each simulation:
           a. Shuffle trade order
           b. Simulate equity curve
           c. Record max drawdown
           d. Check for ruin condition
        3. Calculate statistics across all simulations
        
        Args:
            trades: List of completed trades
            starting_equity: Starting account equity
            
        Returns:
            StressTestResult with all metrics
        """
        self.log_info(f"Starting Monte Carlo stress test: {self.n_simulations} simulations")
        
        # Get trade returns
        trade_returns = self._extract_returns(trades, starting_equity)
        
        if len(trade_returns) < 10:
            self.log_warning("Insufficient trades for stress test")
            return self._empty_result()
        
        # Limit to sample size
        if len(trade_returns) > self.trade_sample:
            trade_returns = trade_returns[-self.trade_sample:]
        
        # Run simulations
        max_drawdowns = []
        final_equities = []
        ruin_count = 0
        
        for i in range(self.n_simulations):
            # Shuffle returns
            shuffled = np.random.permutation(trade_returns)
            
            # Simulate equity curve
            equity_curve = self._simulate_equity(shuffled, starting_equity)
            
            # Calculate max drawdown
            max_dd = self._calculate_max_drawdown(equity_curve)
            max_drawdowns.append(max_dd)
            
            # Check ruin
            if max_dd >= self.ruin_threshold:
                ruin_count += 1
            
            # Record final equity
            final_equities.append(equity_curve[-1])
        
        # Calculate statistics
        max_drawdowns = np.array(max_drawdowns)
        final_equities = np.array(final_equities)
        
        probability_of_ruin = ruin_count / self.n_simulations
        
        # Drawdown percentiles
        median_dd = np.median(max_drawdowns)
        dd_95 = np.percentile(max_drawdowns, 95)
        dd_99 = np.percentile(max_drawdowns, 99)
        
        # Equity percentiles
        median_equity = np.median(final_equities)
        equity_5 = np.percentile(final_equities, 5)
        equity_95 = np.percentile(final_equities, 95)
        
        # 95% CI for drawdown
        ci_lower = np.percentile(max_drawdowns, 2.5)
        ci_upper = np.percentile(max_drawdowns, 97.5)
        
        # Risk rating
        risk_rating, recommendations = self._assess_risk(
            probability_of_ruin, dd_95, median_dd
        )
        
        result = StressTestResult(
            timestamp=datetime.now(),
            n_trades=len(trade_returns),
            n_simulations=self.n_simulations,
            probability_of_ruin=probability_of_ruin,
            ruin_threshold=self.ruin_threshold,
            median_max_drawdown=median_dd,
            drawdown_95_percentile=dd_95,
            drawdown_99_percentile=dd_99,
            median_final_equity=median_equity,
            equity_5_percentile=equity_5,
            equity_95_percentile=equity_95,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            risk_rating=risk_rating,
            recommendations=recommendations
        )
        
        self.log_info(f"Stress test complete: P(Ruin)={probability_of_ruin:.2%}, "
                     f"95% DD={dd_95:.2%}")
        
        return result
    
    def _extract_returns(
        self,
        trades: List[Trade],
        starting_equity: float
    ) -> np.ndarray:
        """Extract trade returns as percentage of equity."""
        returns = []
        current_equity = starting_equity
        
        for trade in trades:
            pct_return = trade.pnl_currency / current_equity
            returns.append(pct_return)
            current_equity += trade.pnl_currency
        
        return np.array(returns)
    
    def _simulate_equity(
        self,
        returns: np.ndarray,
        starting_equity: float
    ) -> np.ndarray:
        """Simulate equity curve from shuffled returns."""
        equity = [starting_equity]
        
        for ret in returns:
            new_equity = equity[-1] * (1 + ret)
            equity.append(max(new_equity, 0))  # Equity can't go negative
        
        return np.array(equity)
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        return np.max(drawdowns)
    
    def _assess_risk(
        self,
        probability_of_ruin: float,
        dd_95: float,
        median_dd: float
    ) -> Tuple[str, List[str]]:
        """
        Assess overall risk level and provide recommendations.
        
        Risk Levels:
        - LOW: P(Ruin) < 1%, 95% DD < 15%
        - MODERATE: P(Ruin) < 5%, 95% DD < 25%
        - HIGH: P(Ruin) < 10%, 95% DD < 35%
        - CRITICAL: P(Ruin) >= 10% or 95% DD >= 35%
        """
        recommendations = []
        
        if probability_of_ruin < 0.01 and dd_95 < 0.15:
            rating = "LOW"
            recommendations.append("Risk parameters are healthy")
        
        elif probability_of_ruin < 0.05 and dd_95 < 0.25:
            rating = "MODERATE"
            recommendations.append("Consider tightening stop losses")
            recommendations.append("Monitor position sizing")
        
        elif probability_of_ruin < 0.10 and dd_95 < 0.35:
            rating = "HIGH"
            recommendations.append("Reduce position sizes by 25%")
            recommendations.append("Review trade selection criteria")
            recommendations.append("Consider reducing max daily trades")
        
        else:
            rating = "CRITICAL"
            recommendations.append("PAUSE TRADING IMMEDIATELY")
            recommendations.append("Reduce position sizes by 50%")
            recommendations.append("Full system review required")
            recommendations.append("Consider paper trading until metrics improve")
        
        return (rating, recommendations)
    
    def _empty_result(self) -> StressTestResult:
        """Return empty result for insufficient data."""
        return StressTestResult(
            timestamp=datetime.now(),
            n_trades=0,
            n_simulations=0,
            probability_of_ruin=0.0,
            ruin_threshold=self.ruin_threshold,
            median_max_drawdown=0.0,
            drawdown_95_percentile=0.0,
            drawdown_99_percentile=0.0,
            median_final_equity=0.0,
            equity_5_percentile=0.0,
            equity_95_percentile=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
            risk_rating="UNKNOWN",
            recommendations=["Insufficient trade data for stress test"]
        )
    
    def run_scenario_analysis(
        self,
        trades: List[Trade],
        scenarios: Dict[str, float],
        starting_equity: float = 100000.0
    ) -> Dict[str, StressTestResult]:
        """
        Run stress tests under different market scenarios.
        
        Scenarios adjust trade returns to simulate different conditions:
        - "normal": No adjustment
        - "volatile": Returns multiplied by factor
        - "adverse": Negative trades worse, positive trades reduced
        
        Args:
            trades: Trade history
            scenarios: Dict of scenario_name -> adjustment_factor
            starting_equity: Starting equity
            
        Returns:
            Dict of scenario results
        """
        results = {}
        
        for scenario_name, factor in scenarios.items():
            # Adjust returns based on scenario
            adjusted_trades = self._adjust_trades(trades, factor)
            result = self.run_stress_test(adjusted_trades, starting_equity)
            results[scenario_name] = result
        
        return results
    
    def _adjust_trades(self, trades: List[Trade], factor: float) -> List[Trade]:
        """Adjust trade PnL for scenario analysis."""
        adjusted = []
        for trade in trades:
            adjusted_trade = Trade(
                id=trade.id,
                pair=trade.pair,
                setup=trade.setup,
                status=trade.status,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                actual_entry=trade.actual_entry,
                actual_exit=trade.actual_exit,
                pnl_pips=trade.pnl_pips * factor,
                pnl_currency=trade.pnl_currency * factor,
                actual_rr=trade.actual_rr,
                slippage_pips=trade.slippage_pips
            )
            adjusted.append(adjusted_trade)
        return adjusted
    
    def generate_report(self, result: StressTestResult) -> str:
        """Generate human-readable stress test report."""
        lines = [
            "=" * 60,
            "SIGHT MONTE CARLO STRESS TEST REPORT",
            "=" * 60,
            f"Generated: {result.timestamp.isoformat()}",
            f"Trades Analyzed: {result.n_trades}",
            f"Simulations Run: {result.n_simulations:,}",
            "",
            "PROBABILITY OF RUIN",
            "-" * 40,
            f"Ruin Threshold: {result.ruin_threshold:.0%} drawdown",
            f"Probability of Ruin: {result.probability_of_ruin:.2%}",
            "",
            "DRAWDOWN ANALYSIS (95% Confidence Interval)",
            "-" * 40,
            f"Median Max Drawdown: {result.median_max_drawdown:.2%}",
            f"95th Percentile: {result.drawdown_95_percentile:.2%}",
            f"99th Percentile: {result.drawdown_99_percentile:.2%}",
            f"95% CI: [{result.ci_95_lower:.2%}, {result.ci_95_upper:.2%}]",
            "",
            "EQUITY DISTRIBUTION",
            "-" * 40,
            f"Median Final Equity: ${result.median_final_equity:,.2f}",
            f"5th Percentile: ${result.equity_5_percentile:,.2f}",
            f"95th Percentile: ${result.equity_95_percentile:,.2f}",
            "",
            f"RISK RATING: {result.risk_rating}",
            "-" * 40,
            "Recommendations:"
        ]
        
        for rec in result.recommendations:
            lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
