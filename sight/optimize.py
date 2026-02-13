#!/usr/bin/env python3
"""
SIGHT Optimization Runner

Run this script to:
1. Optimize parameters for all 12 currency pairs
2. Run walk-forward validation
3. Generate pair_config.json
4. Create optimization report

Usage:
    python -m sight.optimize

Loop Logic:
    FOR each pair in 12_PAIRS:
        WHILE WinRate < 70% OR RRR < 2.2:
            Adjust Hyperparameters
            Backtest last 5 years
            IF Iterations > 100:
                Flag for Manual Review
        Save optimal config to pair_config.json

Walk-Forward Validation:
    Train: 2010-2020
    Validate: 2021-2025
    Pass if: Win Rate >= 65%
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sight.data.provider import ParquetDataProvider
from sight.strategies.ict_engine import ICTEngine
from sight.analysis.confluence import ConfluenceFilter
from sight.optimization.backtest import BacktestEngine
from sight.optimization.optimizer import ParameterOptimizer
from sight.core.constants import (
    SUPPORTED_PAIRS, TRAIN_START_YEAR, TRAIN_END_YEAR,
    VALIDATION_START_YEAR, VALIDATION_END_YEAR,
    TARGET_WIN_RATE, TARGET_RISK_REWARD
)


def main():
    print("=" * 70)
    print("SIGHT PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"\nTargets:")
    print(f"  Win Rate:    >= {TARGET_WIN_RATE:.0%}")
    print(f"  Risk-Reward: >= {TARGET_RISK_REWARD}")
    print(f"\nPairs: {len(SUPPORTED_PAIRS)}")
    print(f"Training:   {TRAIN_START_YEAR} - {TRAIN_END_YEAR}")
    print(f"Validation: {VALIDATION_START_YEAR} - {VALIDATION_END_YEAR}")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    
    data_provider = ParquetDataProvider("data/parquet")
    ict_engine = ICTEngine()
    confluence_filter = ConfluenceFilter()
    
    backtest_engine = BacktestEngine(
        data_provider=data_provider,
        ict_engine=ict_engine,
        confluence_filter=confluence_filter
    )
    
    optimizer = ParameterOptimizer(
        backtest_engine=backtest_engine,
        output_dir="config"
    )
    
    # Define date ranges
    train_start = datetime(TRAIN_START_YEAR, 1, 1)
    train_end = datetime(TRAIN_END_YEAR, 12, 31)
    val_start = datetime(VALIDATION_START_YEAR, 1, 1)
    val_end = datetime(VALIDATION_END_YEAR, 12, 31)
    
    # Run optimization
    print("\n[2/4] Running optimization loop...")
    print("-" * 70)
    
    results = optimizer.optimize_all_pairs(
        train_start, train_end,
        val_start, val_end
    )
    
    # Save configurations
    print("\n[3/4] Saving configurations...")
    config_path = optimizer.save_configs(results)
    print(f"  Saved to: {config_path}")
    
    # Generate report
    print("\n[4/4] Generating report...")
    report = optimizer.generate_report(results)
    
    report_path = Path("config") / "optimization_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r.validation_passed)
    flagged = sum(1 for r in results.values() if r.flagged_for_review)
    
    print(f"\nPairs Optimized:    {len(results)}")
    print(f"Validation Passed:  {passed}/{len(results)}")
    print(f"Flagged for Review: {flagged}")
    
    print("\nPair Results:")
    print("-" * 70)
    
    for pair, result in sorted(results.items()):
        status = "✅ PASS" if result.validation_passed else "❌ FAIL"
        flag = " [REVIEW]" if result.flagged_for_review else ""
        
        train_wr = result.train_result.win_rate if result.train_result else 0
        val_wr = result.validation_result.win_rate if result.validation_result else 0
        
        print(f"  {pair}: {status}{flag}")
        print(f"    Train WR: {train_wr:.1%}, Val WR: {val_wr:.1%}")
        
        if result.failure_reason:
            print(f"    Reason: {result.failure_reason}")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    # Return exit code based on results
    if passed < len(results) // 2:
        print("\n⚠️  WARNING: Less than 50% of pairs passed validation")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
