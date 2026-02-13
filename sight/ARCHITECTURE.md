# SIGHT - Institutional ICT Trading Engine

## Architecture Overview

```
sight/
├── core/                    # Core types, constants, base classes
│   ├── types.py            # Type definitions (Trade, Setup, etc.)
│   ├── constants.py        # Global configuration constants
│   └── base.py             # Abstract base classes
│
├── data/                    # Data Pipeline (Phase 1)
│   ├── pipeline.py         # ZIP extraction, Parquet conversion
│   ├── provider.py         # Data access layer
│   └── resampler.py        # OHLCV resampling utilities
│
├── strategies/              # ICT Narrative Engine (Phase 2) - 80% Weight
│   ├── ict_engine.py       # Main ICT model orchestration
│   ├── market_structure.py # HH/HL/LH/LL detection
│   ├── liquidity.py        # Pool identification, sweep detection
│   ├── fvg.py              # Fair Value Gap detection
│   └── signal_generator.py # Combined signal generation
│
├── analysis/                # Statistical Confluence (Phase 3) - 20% Weight
│   ├── indicators.py       # Bollinger/Keltner calculations
│   └── confluence.py       # Boundary touch validation
│
├── monitoring/              # Performance Monitoring (Phase 4)
│   ├── performance_tracker.py  # Z-Score, Expectancy
│   ├── stress_test.py          # Monte Carlo simulation
│   ├── slippage_monitor.py     # Execution quality
│   └── dashboard.py            # Live metrics display
│
├── optimization/            # Optimization Engine (Phase 5)
│   ├── backtest.py         # Vectorized backtesting
│   └── optimizer.py        # Grid search, walk-forward
│
├── execution/               # Execution Layer (Phase 6)
│   ├── order_manager.py    # Trade execution
│   ├── risk_manager.py     # Position sizing
│   └── position_tracker.py # Open position tracking
│
├── config/                  # Configuration Management
│   └── loader.py           # Config loading utilities
│
├── engine.py               # Main orchestrator
├── optimize.py             # Optimization runner
└── run_pipeline.py         # Data pipeline runner
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Win Rate | ≥ 70% |
| Risk-Reward | ≥ 1:2.2 |
| Profit Factor | > 2.0 |
| Max Daily Trades | 2 |
| Validation Win Rate | ≥ 65% |

## ICT Model Sequence (80% Decision Weight)

```
1. HTF Bias Analysis (1H + 15m alignment)
   └─→ Determine bullish/bearish bias
   
2. Liquidity Mapping
   └─→ Session H/L, 48-candle swings, equal levels

3. Liquidity Sweep Detection
   └─→ Price pierces level + rejects (closes inside)
   
4. Market Structure Shift (MSS)
   └─→ Displacement candle breaks structure level
   
5. Fair Value Gap Entry
   └─→ Enter at 50% / 25% / 100% of FVG
```

## Confluence Filter (20% Decision Weight)

Valid setup requires:
- Price touches Bollinger Band boundary, OR
- Price touches Keltner Channel boundary

## Safety Mechanisms

### Z-Score Monitor
```
Z = (N × (R − 0.5) − P) / sqrt((P × (P − N)) / (N − 1))

Where:
  N = total trades
  R = streak count
  P = 2 × wins × losses

If Z > 1.96 → Significant streak dependency
If win rate < 60% over 50 trades → Confidence Warning
```

### Expectancy Audit
```
EV = (Win% × AvgWin) − (Loss% × AvgLoss)

If EV < 1.0 → Pause trading
```

### Monte Carlo Stress Test
- 10,000 simulations
- Output: Probability of Ruin, 95% Drawdown CI

### Slippage Guard
```
If Spread + Slippage > 15% of target profit:
  → Block pair for the day
```

## Optimization Loop

```python
FOR each pair in 12_PAIRS:
    WHILE WinRate < 70% OR RRR < 2.2:
        Adjust Hyperparameters
        Backtest last 5 years
        IF Iterations > 100:
            Flag for Manual Review
    Save optimal config to pair_config.json
```

### Hyperparameters
- Sweep Depth (ATR multiples)
- Displacement Strength (ATR multiples)
- FVG Entry Offset (0-1)
- HTF EMA Period (100/200/600)

### Walk-Forward Validation
- Train: 2010-2020
- Validate: 2021-2025
- Pass if: Win Rate ≥ 65%

## Usage

### 1. Run Data Pipeline
```bash
python -m sight.run_pipeline
```

### 2. Run Optimization
```bash
python -m sight.optimize
```

### 3. Use Engine
```python
from sight.engine import SIGHTEngine, create_engine

# Create engine with data pipeline
engine = create_engine(
    historical_data_dir="historical_data",
    output_dir="data/parquet",
    config_dir="config"
)

# Start trading
engine.start()

# Process a pair
trade = engine.process_pair("EURUSD")

# Get dashboard
print(engine.get_dashboard())

# Run stress test
stress_result = engine.run_stress_test()
```

## Configuration Example

```json
{
  "pair": "EURUSD",
  "sweep_depth_atr_multiple": 0.5,
  "sweep_lookback_candles": 48,
  "displacement_threshold_atr": 1.5,
  "fvg_entry_offset": 0.5,
  "htf_ema_period": 200,
  "default_risk_reward": 2.2,
  "bb_period": 20,
  "kc_period": 20,
  "backtest_win_rate": 0.72,
  "validation_passed": true
}
```
