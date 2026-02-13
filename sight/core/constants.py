"""SIGHT Constants - Global configuration and trading parameters."""
from typing import List, Dict

# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================
TARGET_WIN_RATE: float = 0.70  # 70% minimum
TARGET_RISK_REWARD: float = 2.2  # 1:2.2 RRR
TARGET_PROFIT_FACTOR: float = 2.0  # Minimum profit factor
VALIDATION_WIN_RATE: float = 0.65  # Walk-forward validation threshold
MAX_OPTIMIZATION_ITERATIONS: int = 100
MAX_DAILY_TRADES: int = 2

# ============================================================================
# CURRENCY PAIRS
# ============================================================================
SUPPORTED_PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "EURJPY", "GBPJPY",
    "AUDJPY", "EURCHF", "XAUUSD", "NZDUSD"
]

# Pip value configurations (per standard lot)
PIP_VALUES: Dict[str, float] = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "USDJPY": 9.09,
    "USDCHF": 10.0, "AUDUSD": 10.0, "USDCAD": 7.69,
    "EURJPY": 9.09, "GBPJPY": 9.09, "AUDJPY": 9.09,
    "EURCHF": 10.0, "XAUUSD": 10.0, "NZDUSD": 10.0
}

# Point to pip conversion (4 or 2 decimal pairs)
POINT_TO_PIP: Dict[str, float] = {
    "EURUSD": 10.0, "GBPUSD": 10.0, "USDJPY": 100.0,
    "USDCHF": 10.0, "AUDUSD": 10.0, "USDCAD": 10.0,
    "EURJPY": 100.0, "GBPJPY": 100.0, "AUDJPY": 100.0,
    "EURCHF": 10.0, "XAUUSD": 10.0, "NZDUSD": 10.0
}

# ============================================================================
# ICT ENGINE DEFAULTS
# ============================================================================
# Swing Detection
DEFAULT_SWING_STRENGTH: int = 3  # Candles on each side
SWING_LOOKBACK_CANDLES: int = 48  # 48 candles for liquidity pools

# Market Structure
HTF_EMA_PERIODS: List[int] = [100, 200, 600]  # Hyperparameter options
DEFAULT_HTF_EMA: int = 200

# Liquidity Sweep
DEFAULT_SWEEP_DEPTH_ATR: float = 0.5  # Minimum sweep depth in ATR
SWEEP_REJECTION_MAX_CANDLES: int = 3  # Max candles for rejection confirmation

# Displacement (MSS)
DEFAULT_DISPLACEMENT_ATR: float = 1.5  # Minimum displacement in ATR
DISPLACEMENT_BODY_PERCENT: float = 0.7  # Minimum body to range ratio

# Fair Value Gap
DEFAULT_FVG_MIN_SIZE_ATR: float = 0.3  # Minimum FVG size in ATR
FVG_ENTRY_OFFSETS: List[float] = [0.25, 0.5, 1.0]  # Entry level options

# ============================================================================
# CONFLUENCE FILTER DEFAULTS
# ============================================================================
BOLLINGER_PERIOD: int = 20
BOLLINGER_STD: float = 2.0
KELTNER_PERIOD: int = 20
KELTNER_ATR_MULTIPLE: float = 1.5

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
DEFAULT_RISK_PERCENT: float = 0.01  # 1% risk per trade
MAX_RISK_PERCENT: float = 0.02  # 2% maximum
MAX_SPREAD_PERCENT_OF_TARGET: float = 0.15  # 15% of target profit

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================
Z_SCORE_THRESHOLD: float = 1.96  # 95% confidence level
CONFIDENCE_WARNING_THRESHOLD: float = 0.60  # 60% win rate over last 50
LOOKBACK_TRADES_FOR_CONFIDENCE: int = 50
MIN_EXPECTED_VALUE: float = 1.0  # Minimum EV before pause
MONTE_CARLO_ITERATIONS: int = 10000
MONTE_CARLO_TRADE_SAMPLE: int = 100

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
# Hyperparameter search ranges
SWEEP_DEPTH_RANGE: List[float] = [0.3, 0.5, 0.7, 1.0]
DISPLACEMENT_RANGE: List[float] = [1.0, 1.5, 2.0, 2.5]
FVG_OFFSET_RANGE: List[float] = [0.25, 0.5, 0.75, 1.0]
EMA_PERIOD_RANGE: List[int] = [100, 200, 600]
BB_PERIOD_RANGE: List[int] = [14, 20, 30]
KC_PERIOD_RANGE: List[int] = [14, 20, 30]

# Walk-Forward Validation
TRAIN_START_YEAR: int = 2010
TRAIN_END_YEAR: int = 2020
VALIDATION_START_YEAR: int = 2021
VALIDATION_END_YEAR: int = 2025

# ============================================================================
# DATA PIPELINE
# ============================================================================
PARQUET_COMPRESSION: str = "snappy"
RESAMPLE_TIMEFRAMES: Dict[str, str] = {
    "15min": "15min",
    "1H": "1h"  # lowercase 'h' for pandas 2.0+
}

# ============================================================================
# SESSION TIMES (UTC)
# ============================================================================
TRADING_SESSIONS: Dict[str, Dict[str, int]] = {
    "ASIA": {"start": 0, "end": 8},
    "LONDON": {"start": 7, "end": 16},
    "NEW_YORK": {"start": 12, "end": 21},
    "KILLZONE_LONDON": {"start": 7, "end": 10},
    "KILLZONE_NY": {"start": 12, "end": 15}
}

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_DIR: str = "data"
HISTORICAL_DATA_DIR: str = "historical_data"
PARQUET_DIR: str = "data/parquet"
CONFIG_DIR: str = "config"
LOGS_DIR: str = "logs"
REPORTS_DIR: str = "reports"
