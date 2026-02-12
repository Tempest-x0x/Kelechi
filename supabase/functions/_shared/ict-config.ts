/**
 * Institutional Order Flow (ICT/SMC) Configuration
 * 
 * Central configuration for the high-confluence trading system.
 * Target Metrics: 70% Win Rate | 1:2.2 RRR | 1-2 Trades per Pair/Day
 */

// ============================================
// CORE TRADING PARAMETERS
// ============================================
export const ICT_CONFIG = {
  // Risk-Reward Ratio
  RRR_RATIO: 2.2,
  
  // Target performance metrics
  TARGET_WIN_RATE: 70,
  TARGET_PROFIT_FACTOR: 2.0,
  MAX_TRADES_PER_PAIR_PER_DAY: 2,
  
  // Minimum confidence for trade execution
  MIN_CONFIDENCE_THRESHOLD: 65,
  
  // ATR multipliers
  ATR_PERIOD: 14,
  STOP_LOSS_ATR_MULT: 1.0,
  TAKE_PROFIT_ATR_MULT: 2.2,  // Aligned with RRR_RATIO
  
  // Position expiry (hours)
  POSITION_EXPIRY_HOURS: 4,
} as const;

// ============================================
// TRADING SESSION KILLZONES (UTC)
// ============================================
export const KILLZONES = {
  LONDON: {
    name: 'London',
    start: { hour: 7, minute: 0 },
    end: { hour: 10, minute: 0 },
  },
  NEW_YORK: {
    name: 'New York',
    start: { hour: 12, minute: 0 },
    end: { hour: 15, minute: 0 },
  },
} as const;

export type KillzoneName = keyof typeof KILLZONES;

// ============================================
// HIGHER TIMEFRAME (HTF) TREND MATRIX
// ============================================
export const HTF_CONFIG = {
  // Timeframes for multi-timeframe analysis
  TREND_TIMEFRAME: '1h',      // 1-Hour for 200 EMA trend
  STRUCTURE_TIMEFRAME: '15m',  // 15-Minute for market structure
  ENTRY_TIMEFRAME: '1m',       // 1-Minute for precise entries
  
  // EMA periods
  EMA_200_PERIOD: 200,
  EMA_50_PERIOD: 50,
  
  // Market structure lookback
  STRUCTURE_LOOKBACK_CANDLES: 20,
  SWING_POINT_LOOKBACK: 5,
} as const;

// ============================================
// LIQUIDITY SWEEP DETECTION
// ============================================
export const LIQUIDITY_CONFIG = {
  // Lookback period for finding liquidity levels
  SWEEP_LOOKBACK_CANDLES: 20,
  
  // Minimum pierce distance (in ATR) to confirm sweep
  MIN_SWEEP_DISTANCE_ATR: 0.3,
  
  // Candle must close back inside band to confirm sweep
  REQUIRE_CLOSE_INSIDE: true,
  
  // Tolerance for "inside band" check (% of ATR)
  CLOSE_TOLERANCE_ATR: 0.1,
} as const;

// ============================================
// FAIR VALUE GAP (FVG) CONFIGURATION
// ============================================
export const FVG_CONFIG = {
  // Minimum gap size (in pips) to be considered valid
  MIN_GAP_PIPS: 1.5,
  
  // Entry at 50% (equilibrium) of FVG
  ENTRY_EQUILIBRIUM_LEVEL: 0.5,
  
  // FVG must be created by displacement candle
  MIN_DISPLACEMENT_ATR: 1.5,
  
  // Maximum FVG age (candles) for entry
  MAX_FVG_AGE_CANDLES: 10,
  
  // Stop Loss: X pips below sweep low
  SL_BUFFER_PIPS: 2,
} as const;

// ============================================
// MARKET STRUCTURE SHIFT (MSS) DETECTION
// ============================================
export const MSS_CONFIG = {
  // Lookback for swing points
  SWING_LOOKBACK: 5,
  
  // Minimum displacement candle body (% of ATR)
  MIN_DISPLACEMENT_BODY_ATR: 0.8,
  
  // MSS confirmation: break of recent swing
  REQUIRE_SWING_BREAK: true,
} as const;

// ============================================
// VALUE AREA ALERTS (formerly entry signals)
// ============================================
export const VALUE_AREA_PATTERNS = {
  // These are now ALERTS only, not entry triggers
  bb_lower_touch: { type: 'BUY_ALERT', description: 'Price at lower Bollinger Band' },
  bb_upper_touch: { type: 'SELL_ALERT', description: 'Price at upper Bollinger Band' },
  kc_lower_touch: { type: 'BUY_ALERT', description: 'Price at lower Keltner Channel' },
  kc_upper_touch: { type: 'SELL_ALERT', description: 'Price at upper Keltner Channel' },
} as const;

// ============================================
// PRUNED PATTERNS (statistically insufficient)
// Win rates below 52% cannot support 1:2.2 RRR
// ============================================
export const PRUNED_PATTERNS = [
  'stochrsi_overbought',  // 49.25% win rate
  'stochrsi_oversold',    // 50.34% win rate
  'macd_bullish_cross',   // 47.85% win rate
  'macd_bearish_cross',   // 47.39% win rate
  'golden_cross',         // 45.89% win rate
  'death_cross',          // 45.87% win rate
  'bullish_engulfing',    // 48.39% win rate
  'bearish_engulfing',    // 48.24% win rate
] as const;

// ============================================
// PERFORMANCE MONITORING
// ============================================
export const PERFORMANCE_CONFIG = {
  // Z-Score threshold for "streaky" behavior
  Z_SCORE_SIGNIFICANCE: 1.96,
  
  // Win rate warning threshold (last N trades)
  WIN_RATE_WARNING_THRESHOLD: 60,
  WIN_RATE_LOOKBACK_TRADES: 50,
  
  // Expectancy (EV) requirement
  MIN_EXPECTANCY: 1.0,
  
  // Slippage guard: max slippage as % of RRR target
  MAX_SLIPPAGE_PERCENT: 15,
  
  // Monte Carlo simulation
  MONTE_CARLO_SIMULATIONS: 10000,
  MONTE_CARLO_TRADE_HISTORY: 100,
  
  // Audit frequency
  AUDIT_FREQUENCY: 'daily',  // After NY session close
} as const;

// ============================================
// SUPPORTED CURRENCY PAIRS
// ============================================
export const SUPPORTED_PAIRS = [
  'EUR/USD',
  'GBP/USD',
  'USD/JPY',
  'USD/CHF',
  'AUD/USD',
  'USD/CAD',
  'EUR/JPY',
  'GBP/JPY',
  'AUD/JPY',
  'XAU/USD',
  'EUR/CHF',
  'EUR/GBP',
] as const;

export type SupportedPair = typeof SUPPORTED_PAIRS[number];

// ============================================
// PIP VALUES
// ============================================
export const PIP_VALUES: Record<SupportedPair, number> = {
  'EUR/USD': 0.0001,
  'GBP/USD': 0.0001,
  'USD/JPY': 0.01,
  'USD/CHF': 0.0001,
  'AUD/USD': 0.0001,
  'USD/CAD': 0.0001,
  'EUR/JPY': 0.01,
  'GBP/JPY': 0.01,
  'AUD/JPY': 0.01,
  'XAU/USD': 0.01,
  'EUR/CHF': 0.0001,
  'EUR/GBP': 0.0001,
};

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Check if current time is within a killzone
 */
export function isInKillzone(date: Date = new Date()): { inKillzone: boolean; killzone: KillzoneName | null } {
  const utcHour = date.getUTCHours();
  const utcMinute = date.getUTCMinutes();
  const totalMinutes = utcHour * 60 + utcMinute;

  for (const [name, zone] of Object.entries(KILLZONES)) {
    const startMinutes = zone.start.hour * 60 + zone.start.minute;
    const endMinutes = zone.end.hour * 60 + zone.end.minute;
    
    if (totalMinutes >= startMinutes && totalMinutes < endMinutes) {
      return { inKillzone: true, killzone: name as KillzoneName };
    }
  }

  return { inKillzone: false, killzone: null };
}

/**
 * Get pip value for a symbol
 */
export function getPipValue(symbol: string): number {
  return PIP_VALUES[symbol as SupportedPair] || 0.0001;
}

/**
 * Convert price difference to pips
 */
export function priceToPips(priceDiff: number, symbol: string): number {
  return Math.abs(priceDiff) / getPipValue(symbol);
}

/**
 * Convert pips to price
 */
export function pipToPrice(pips: number, symbol: string): number {
  return pips * getPipValue(symbol);
}

/**
 * Check if a pattern is pruned (statistically insufficient)
 */
export function isPatternPruned(patternName: string): boolean {
  return PRUNED_PATTERNS.includes(patternName as any);
}

/**
 * Calculate take profit based on RRR
 */
export function calculateTakeProfit(
  entryPrice: number,
  stopLoss: number,
  signalType: 'BUY' | 'SELL'
): number {
  const riskDistance = Math.abs(entryPrice - stopLoss);
  const rewardDistance = riskDistance * ICT_CONFIG.RRR_RATIO;
  
  return signalType === 'BUY' 
    ? entryPrice + rewardDistance 
    : entryPrice - rewardDistance;
}

/**
 * Get next killzone info
 */
export function getNextKillzone(date: Date = new Date()): { name: KillzoneName; startsIn: number } {
  const utcHour = date.getUTCHours();
  const utcMinute = date.getUTCMinutes();
  const totalMinutes = utcHour * 60 + utcMinute;

  const zones = Object.entries(KILLZONES).map(([name, zone]) => ({
    name: name as KillzoneName,
    startMinutes: zone.start.hour * 60 + zone.start.minute,
  }));

  // Find next killzone
  for (const zone of zones) {
    if (zone.startMinutes > totalMinutes) {
      return { name: zone.name, startsIn: zone.startMinutes - totalMinutes };
    }
  }

  // If all killzones passed today, return first one tomorrow
  const firstZone = zones[0];
  const minutesUntilMidnight = 1440 - totalMinutes;
  return { name: firstZone.name, startsIn: minutesUntilMidnight + firstZone.startMinutes };
}

export default ICT_CONFIG;
