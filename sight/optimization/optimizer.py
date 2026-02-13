"""SIGHT Parameter Optimizer - Grid search and walk-forward validation."""
from typing import List, Dict, Tuple, Optional, Iterator
import json
from datetime import datetime
from pathlib import Path
from itertools import product
from dataclasses import dataclass
import copy

from ..core.base import Optimizer, BaseLogger
from ..core.types import PairConfig, BacktestResult
from ..core.constants import (
    SUPPORTED_PAIRS, TARGET_WIN_RATE, TARGET_RISK_REWARD,
    VALIDATION_WIN_RATE, MAX_OPTIMIZATION_ITERATIONS,
    SWEEP_DEPTH_RANGE, DISPLACEMENT_RANGE, FVG_OFFSET_RANGE,
    EMA_PERIOD_RANGE, BB_PERIOD_RANGE, KC_PERIOD_RANGE,
    TRAIN_START_YEAR, TRAIN_END_YEAR,
    VALIDATION_START_YEAR, VALIDATION_END_YEAR,
    CONFIG_DIR
)
from .backtest import BacktestEngine


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    pair: str
    best_config: PairConfig
    train_result: BacktestResult
    validation_result: Optional[BacktestResult]
    validation_passed: bool
    iterations_run: int
    flagged_for_review: bool
    failure_reason: Optional[str] = None


class ParameterOptimizer(Optimizer):
    """
    Parameter optimizer using grid search.
    
    Optimization Loop:
    ```
    FOR each pair in 12_PAIRS:
        WHILE WinRate < 70% OR RRR < 2.2:
            Adjust Hyperparameters
            Backtest last 5 years
            IF Iterations > 100:
                Flag for Manual Review
        Save optimal config to pair_config.json
    ```
    
    Hyperparameters to Tune:
    - Sweep Depth (ATR multiples)
    - Displacement Strength (ATR multiples)
    - FVG Offset (entry level)
    - HTF EMA Period (100/200/600)
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        output_dir: str = CONFIG_DIR
    ):
        super().__init__("ParameterOptimizer")
        self.backtest_engine = backtest_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter search space
        self.param_grid = {
            'sweep_depth_atr_multiple': SWEEP_DEPTH_RANGE,
            'displacement_threshold_atr': DISPLACEMENT_RANGE,
            'fvg_entry_offset': FVG_OFFSET_RANGE,
            'htf_ema_period': EMA_PERIOD_RANGE,
            'bb_period': BB_PERIOD_RANGE,
            'kc_period': KC_PERIOD_RANGE
        }
    
    def _generate_param_combinations(self) -> Iterator[Dict]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        for combo in product(*values):
            yield dict(zip(keys, combo))
    
    def optimize_pair(
        self,
        pair: str,
        train_start: datetime,
        train_end: datetime
    ) -> PairConfig:
        """
        Optimize parameters for a single pair.
        
        Algorithm:
        1. Generate all parameter combinations
        2. For each combination:
           a. Create PairConfig
           b. Run backtest
           c. Check if targets met
           d. Track best result
        3. If iterations > 100 without meeting targets: flag for review
        4. Return best config
        
        Args:
            pair: Currency pair
            train_start: Training period start
            train_end: Training period end
            
        Returns:
            Best PairConfig found
        """
        self.log_info(f"Optimizing {pair}: {train_start.date()} to {train_end.date()}")
        
        best_config = None
        best_score = -float('inf')
        best_result = None
        iteration = 0
        targets_met = False
        
        base_config = PairConfig(pair=pair)
        
        for params in self._generate_param_combinations():
            iteration += 1
            
            # Create config with current parameters
            config = copy.deepcopy(base_config)
            for key, value in params.items():
                setattr(config, key, value)
            
            # Run backtest
            try:
                result = self.backtest_engine.run_backtest(
                    pair, train_start, train_end, config
                )
            except Exception as e:
                self.log_error(f"Backtest failed for {pair}: {e}")
                continue
            
            # Calculate score (combined metric)
            score = self._calculate_optimization_score(result)
            
            # Check if targets met
            if result.win_rate >= TARGET_WIN_RATE and result.avg_rr >= TARGET_RISK_REWARD:
                targets_met = True
                if score > best_score:
                    best_score = score
                    best_config = config
                    best_result = result
                    self.log_info(f"  Iteration {iteration}: Targets met! "
                                 f"WR={result.win_rate:.1%}, RR={result.avg_rr:.2f}")
            elif score > best_score:
                best_score = score
                best_config = config
                best_result = result
            
            # Early stop if targets met and we've explored enough
            if targets_met and iteration >= 20:
                break
            
            # Check iteration limit
            if iteration >= MAX_OPTIMIZATION_ITERATIONS:
                self.log_warning(f"{pair}: Max iterations reached without meeting targets")
                break
            
            # Progress logging
            if iteration % 10 == 0:
                self.log_debug(f"  Iteration {iteration}: Best WR={best_result.win_rate:.1%}, "
                              f"Score={best_score:.4f}")
        
        # Update config with results
        if best_config:
            best_config.backtest_win_rate = best_result.win_rate
            best_config.backtest_profit_factor = best_result.profit_factor
            best_config.optimization_iterations = iteration
        
        return best_config
    
    def _calculate_optimization_score(self, result: BacktestResult) -> float:
        """
        Calculate composite optimization score.
        
        Score components:
        - Win rate (40% weight)
        - Profit factor (30% weight)
        - Risk-reward ratio (20% weight)
        - Trade count (10% weight) - penalize too few trades
        """
        if result.total_trades < 10:
            return -1.0  # Not enough trades
        
        wr_score = result.win_rate * 0.4
        pf_score = min(result.profit_factor / 5.0, 1.0) * 0.3
        rr_score = min(result.avg_rr / 4.0, 1.0) * 0.2
        trade_score = min(result.total_trades / 200, 1.0) * 0.1
        
        return wr_score + pf_score + rr_score + trade_score
    
    def walk_forward_validate(
        self,
        pair: str,
        config: PairConfig,
        val_start: datetime,
        val_end: datetime
    ) -> Tuple[bool, BacktestResult]:
        """
        Validate optimized parameters on out-of-sample data.
        
        Walk-Forward Validation:
        - Train: 2010-2020
        - Validate: 2021-2025
        
        Validation passes only if:
        - Win Rate >= 65%
        
        Args:
            pair: Currency pair
            config: Optimized config to validate
            val_start: Validation start date
            val_end: Validation end date
            
        Returns:
            Tuple of (passed, result)
        """
        self.log_info(f"Walk-forward validation: {pair} from {val_start.date()} to {val_end.date()}")
        
        try:
            result = self.backtest_engine.run_backtest(
                pair, val_start, val_end, config
            )
        except Exception as e:
            self.log_error(f"Validation backtest failed: {e}")
            return (False, BacktestResult(
                pair=pair, timeframe=None, 
                start_date=val_start, end_date=val_end
            ))
        
        # Check validation criteria
        passed = result.win_rate >= VALIDATION_WIN_RATE
        
        if passed:
            self.log_info(f"  Validation PASSED: WR={result.win_rate:.1%}")
            config.validation_passed = True
        else:
            self.log_warning(f"  Validation FAILED: WR={result.win_rate:.1%} < {VALIDATION_WIN_RATE:.1%}")
            config.validation_passed = False
        
        return (passed, result)
    
    def optimize_all_pairs(
        self,
        train_start: datetime = None,
        train_end: datetime = None,
        val_start: datetime = None,
        val_end: datetime = None
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize all supported pairs with walk-forward validation.
        
        Args:
            train_start: Training period start
            train_end: Training period end
            val_start: Validation period start
            val_end: Validation period end
            
        Returns:
            Dict mapping pair to OptimizationResult
        """
        # Default dates
        train_start = train_start or datetime(TRAIN_START_YEAR, 1, 1)
        train_end = train_end or datetime(TRAIN_END_YEAR, 12, 31)
        val_start = val_start or datetime(VALIDATION_START_YEAR, 1, 1)
        val_end = val_end or datetime(VALIDATION_END_YEAR, 12, 31)
        
        results = {}
        
        for pair in SUPPORTED_PAIRS:
            self.log_info(f"\n{'='*60}")
            self.log_info(f"OPTIMIZING {pair}")
            self.log_info(f"{'='*60}")
            
            # Optimize on training data
            config = self.optimize_pair(pair, train_start, train_end)
            
            if config is None:
                results[pair] = OptimizationResult(
                    pair=pair,
                    best_config=PairConfig(pair=pair),
                    train_result=BacktestResult(pair=pair, timeframe=None, 
                                               start_date=train_start, end_date=train_end),
                    validation_result=None,
                    validation_passed=False,
                    iterations_run=0,
                    flagged_for_review=True,
                    failure_reason="Optimization failed - no valid config found"
                )
                continue
            
            # Get training result
            train_result = self.backtest_engine.run_backtest(
                pair, train_start, train_end, config
            )
            
            # Validate on out-of-sample
            val_passed, val_result = self.walk_forward_validate(
                pair, config, val_start, val_end
            )
            
            # Check if flagged for review
            flagged = (
                config.optimization_iterations >= MAX_OPTIMIZATION_ITERATIONS or
                not val_passed
            )
            
            failure_reason = None
            if not val_passed:
                failure_reason = f"Validation win rate {val_result.win_rate:.1%} < {VALIDATION_WIN_RATE:.1%}"
            elif config.backtest_win_rate < TARGET_WIN_RATE:
                failure_reason = f"Training win rate {config.backtest_win_rate:.1%} < {TARGET_WIN_RATE:.1%}"
            
            results[pair] = OptimizationResult(
                pair=pair,
                best_config=config,
                train_result=train_result,
                validation_result=val_result,
                validation_passed=val_passed,
                iterations_run=config.optimization_iterations,
                flagged_for_review=flagged,
                failure_reason=failure_reason
            )
        
        return results
    
    def save_configs(
        self,
        results: Dict[str, OptimizationResult],
        filename: str = "pair_config.json"
    ) -> Path:
        """
        Save optimized configurations to JSON file.
        
        Args:
            results: Optimization results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output = {
            "generated": datetime.now().isoformat(),
            "pairs": {}
        }
        
        for pair, result in results.items():
            config = result.best_config
            output["pairs"][pair] = {
                **config.to_dict(),
                "optimization_summary": {
                    "iterations_run": result.iterations_run,
                    "validation_passed": result.validation_passed,
                    "flagged_for_review": result.flagged_for_review,
                    "failure_reason": result.failure_reason,
                    "train_win_rate": result.train_result.win_rate if result.train_result else 0,
                    "validation_win_rate": result.validation_result.win_rate if result.validation_result else 0
                }
            }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.log_info(f"Saved configs to {filepath}")
        return filepath
    
    def generate_report(self, results: Dict[str, OptimizationResult]) -> str:
        """Generate human-readable optimization report."""
        lines = [
            "=" * 70,
            "SIGHT PARAMETER OPTIMIZATION REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "SUMMARY",
            "-" * 70,
        ]
        
        passed = sum(1 for r in results.values() if r.validation_passed)
        total = len(results)
        
        lines.append(f"Pairs Optimized: {total}")
        lines.append(f"Validation Passed: {passed}/{total}")
        lines.append(f"Flagged for Review: {sum(1 for r in results.values() if r.flagged_for_review)}")
        lines.append("")
        
        lines.append("PAIR RESULTS")
        lines.append("-" * 70)
        
        for pair, result in sorted(results.items()):
            status = "✅ PASS" if result.validation_passed else "❌ FAIL"
            flag = " [REVIEW]" if result.flagged_for_review else ""
            
            train_wr = result.train_result.win_rate if result.train_result else 0
            val_wr = result.validation_result.win_rate if result.validation_result else 0
            
            lines.append(f"\n{pair}: {status}{flag}")
            lines.append(f"  Training Win Rate:   {train_wr:.1%}")
            lines.append(f"  Validation Win Rate: {val_wr:.1%}")
            lines.append(f"  Iterations: {result.iterations_run}")
            
            if result.failure_reason:
                lines.append(f"  Failure: {result.failure_reason}")
            
            # Key parameters
            cfg = result.best_config
            lines.append(f"  Parameters:")
            lines.append(f"    Sweep Depth: {cfg.sweep_depth_atr_multiple}")
            lines.append(f"    Displacement: {cfg.displacement_threshold_atr}")
            lines.append(f"    FVG Offset: {cfg.fvg_entry_offset}")
            lines.append(f"    EMA Period: {cfg.htf_ema_period}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class WalkForwardValidator(BaseLogger):
    """
    Walk-Forward Validation framework.
    
    Process:
    1. Split data into training and validation windows
    2. Optimize on training window
    3. Test on validation window
    4. Slide window forward and repeat
    
    This tests strategy robustness across different market regimes.
    """
    
    def __init__(
        self,
        optimizer: ParameterOptimizer,
        train_window_years: int = 3,
        validation_window_months: int = 6
    ):
        super().__init__("WalkForwardValidator")
        self.optimizer = optimizer
        self.train_years = train_window_years
        self.val_months = validation_window_months
    
    def run_walk_forward(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[PairConfig, BacktestResult]]:
        """
        Run walk-forward validation with sliding windows.
        
        Args:
            pair: Currency pair
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of (config, validation_result) tuples
        """
        results = []
        
        current_start = start_date
        
        while current_start < end_date:
            # Define windows
            train_end = current_start.replace(year=current_start.year + self.train_years)
            val_start = train_end
            val_end = val_start.replace(month=val_start.month + self.val_months) \
                      if val_start.month + self.val_months <= 12 \
                      else val_start.replace(year=val_start.year + 1, 
                                            month=(val_start.month + self.val_months) % 12)
            
            if val_end > end_date:
                break
            
            self.log_info(f"Window: Train {current_start.date()}-{train_end.date()}, "
                         f"Validate {val_start.date()}-{val_end.date()}")
            
            # Optimize and validate
            config = self.optimizer.optimize_pair(pair, current_start, train_end)
            passed, result = self.optimizer.walk_forward_validate(pair, config, val_start, val_end)
            
            results.append((config, result))
            
            # Slide window
            current_start = val_start
        
        return results
