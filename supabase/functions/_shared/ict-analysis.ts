/**
 * ICT/SMC Analysis Engine
 * 
 * Implements Institutional Order Flow analysis:
 * - HTF Trend Matrix (200 EMA on 1H, Market Structure on 15m)
 * - Liquidity Sweep Detection
 * - Fair Value Gap (FVG) Identification
 * - Market Structure Shift (MSS) Detection
 */

import {
  ICT_CONFIG,
  HTF_CONFIG,
  LIQUIDITY_CONFIG,
  FVG_CONFIG,
  MSS_CONFIG,
  getPipValue,
  priceToPips,
  pipToPrice,
} from './ict-config.ts';

// ============================================
// TYPES
// ============================================
export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface SwingPoint {
  index: number;
  price: number;
  type: 'HIGH' | 'LOW';
  timestamp: string;
}

export interface MarketStructure {
  trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  swingHighs: SwingPoint[];
  swingLows: SwingPoint[];
  lastSwingHigh: SwingPoint | null;
  lastSwingLow: SwingPoint | null;
  isHigherHighs: boolean;
  isHigherLows: boolean;
  isLowerHighs: boolean;
  isLowerLows: boolean;
}

export interface HTFBias {
  direction: 'BUY' | 'SELL' | null;
  hourlyTrend: 'ABOVE_200EMA' | 'BELOW_200EMA';
  fifteenMinStructure: MarketStructure;
  ema200: number;
  currentPrice: number;
  confidence: number;
  reasons: string[];
}

export interface LiquiditySweep {
  detected: boolean;
  type: 'BUY' | 'SELL';
  sweepPrice: number;
  sweepLow: number;
  sweepHigh: number;
  closedInside: boolean;
  candleIndex: number;
  timestamp: string;
  liquidityLevel: number;
}

export interface FairValueGap {
  detected: boolean;
  type: 'BULLISH' | 'BEARISH';
  topPrice: number;
  bottomPrice: number;
  equilibrium: number;
  gapSizePips: number;
  startIndex: number;
  isValid: boolean;
  age: number;  // candles since formation
}

export interface MarketStructureShift {
  detected: boolean;
  direction: 'BULLISH' | 'BEARISH';
  displacementCandle: Candle | null;
  displacementIndex: number;
  swingBroken: SwingPoint | null;
}

export interface ICTSignal {
  valid: boolean;
  direction: 'BUY' | 'SELL' | null;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  htfBias: HTFBias;
  liquiditySweep: LiquiditySweep | null;
  fvg: FairValueGap | null;
  mss: MarketStructureShift | null;
  confidence: number;
  reasons: string[];
  invalidReasons: string[];
}

// ============================================
// TECHNICAL INDICATOR FUNCTIONS
// ============================================

/**
 * Calculate Exponential Moving Average
 */
export function calculateEMA(closes: number[], period: number): number {
  if (closes.length < period) return closes[closes.length - 1] || 0;
  
  const multiplier = 2 / (period + 1);
  let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  for (let i = period; i < closes.length; i++) {
    ema = (closes[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

/**
 * Calculate Average True Range
 */
export function calculateATR(candles: Candle[], period: number = 14): number {
  if (candles.length < period + 1) return 0;
  
  const trueRanges: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const tr = Math.max(
      candles[i].high - candles[i].low,
      Math.abs(candles[i].high - candles[i - 1].close),
      Math.abs(candles[i].low - candles[i - 1].close)
    );
    trueRanges.push(tr);
  }
  
  return trueRanges.slice(-period).reduce((a, b) => a + b, 0) / period;
}

/**
 * Calculate Keltner Channel
 */
export function calculateKeltnerChannel(
  candles: Candle[],
  emaPeriod: number = 20,
  atrMultiplier: number = 2.0
): { upper: number; middle: number; lower: number } {
  const closes = candles.map(c => c.close);
  const ema = calculateEMA(closes, emaPeriod);
  const atr = calculateATR(candles, emaPeriod);
  
  return {
    upper: ema + (atr * atrMultiplier),
    middle: ema,
    lower: ema - (atr * atrMultiplier),
  };
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBollingerBands(
  closes: number[],
  period: number = 20,
  stdDevMultiplier: number = 2
): { upper: number; middle: number; lower: number } {
  if (closes.length < period) {
    const last = closes[closes.length - 1];
    return { upper: last, middle: last, lower: last };
  }
  
  const slice = closes.slice(-period);
  const sma = slice.reduce((a, b) => a + b, 0) / period;
  const variance = slice.reduce((a, b) => a + Math.pow(b - sma, 2), 0) / period;
  const stdDev = Math.sqrt(variance);
  
  return {
    upper: sma + stdDev * stdDevMultiplier,
    middle: sma,
    lower: sma - stdDev * stdDevMultiplier,
  };
}

// ============================================
// MARKET STRUCTURE ANALYSIS
// ============================================

/**
 * Find swing points in price data
 */
export function findSwingPoints(candles: Candle[], lookback: number = 5): { highs: SwingPoint[]; lows: SwingPoint[] } {
  const highs: SwingPoint[] = [];
  const lows: SwingPoint[] = [];
  
  for (let i = lookback; i < candles.length - lookback; i++) {
    let isSwingHigh = true;
    let isSwingLow = true;
    
    for (let j = 1; j <= lookback; j++) {
      if (candles[i].high <= candles[i - j].high || candles[i].high <= candles[i + j].high) {
        isSwingHigh = false;
      }
      if (candles[i].low >= candles[i - j].low || candles[i].low >= candles[i + j].low) {
        isSwingLow = false;
      }
    }
    
    if (isSwingHigh) {
      highs.push({
        index: i,
        price: candles[i].high,
        type: 'HIGH',
        timestamp: candles[i].timestamp,
      });
    }
    
    if (isSwingLow) {
      lows.push({
        index: i,
        price: candles[i].low,
        type: 'LOW',
        timestamp: candles[i].timestamp,
      });
    }
  }
  
  return { highs, lows };
}

/**
 * Analyze market structure from swing points
 */
export function analyzeMarketStructure(candles: Candle[]): MarketStructure {
  const { highs, lows } = findSwingPoints(candles, HTF_CONFIG.SWING_POINT_LOOKBACK);
  
  const lastSwingHigh = highs.length > 0 ? highs[highs.length - 1] : null;
  const lastSwingLow = lows.length > 0 ? lows[lows.length - 1] : null;
  
  // Check for higher highs/lows or lower highs/lows
  let isHigherHighs = false;
  let isHigherLows = false;
  let isLowerHighs = false;
  let isLowerLows = false;
  
  if (highs.length >= 2) {
    const recentHighs = highs.slice(-3);
    isHigherHighs = recentHighs.every((h, i) => i === 0 || h.price > recentHighs[i - 1].price);
    isLowerHighs = recentHighs.every((h, i) => i === 0 || h.price < recentHighs[i - 1].price);
  }
  
  if (lows.length >= 2) {
    const recentLows = lows.slice(-3);
    isHigherLows = recentLows.every((l, i) => i === 0 || l.price > recentLows[i - 1].price);
    isLowerLows = recentLows.every((l, i) => i === 0 || l.price < recentLows[i - 1].price);
  }
  
  // Determine trend
  let trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
  if (isHigherHighs && isHigherLows) {
    trend = 'BULLISH';
  } else if (isLowerHighs && isLowerLows) {
    trend = 'BEARISH';
  }
  
  return {
    trend,
    swingHighs: highs,
    swingLows: lows,
    lastSwingHigh,
    lastSwingLow,
    isHigherHighs,
    isHigherLows,
    isLowerHighs,
    isLowerLows,
  };
}

// ============================================
// HTF TREND MATRIX
// ============================================

/**
 * Calculate HTF (Higher Timeframe) Bias
 * 
 * Rules:
 * - If 1H Price > 200 EMA AND 15m structure is Bullish (HH/HL) -> ONLY BUY
 * - If 1H Price < 200 EMA AND 15m structure is Bearish (LH/LL) -> ONLY SELL
 * - Otherwise -> No trade
 */
export function calculateHTFBias(
  hourlyCandles: Candle[],
  fifteenMinCandles: Candle[]
): HTFBias {
  const reasons: string[] = [];
  
  // Calculate 200 EMA on 1H timeframe
  const hourlyCloses = hourlyCandles.map(c => c.close);
  const ema200 = calculateEMA(hourlyCloses, HTF_CONFIG.EMA_200_PERIOD);
  const currentPrice = hourlyCloses[hourlyCloses.length - 1];
  
  // Determine 1H trend relative to 200 EMA
  const hourlyTrend = currentPrice > ema200 ? 'ABOVE_200EMA' : 'BELOW_200EMA';
  reasons.push(`1H: Price ${hourlyTrend === 'ABOVE_200EMA' ? 'above' : 'below'} 200 EMA (${ema200.toFixed(5)})`);
  
  // Analyze 15m market structure
  const fifteenMinStructure = analyzeMarketStructure(fifteenMinCandles);
  reasons.push(`15m: ${fifteenMinStructure.trend} structure (HH:${fifteenMinStructure.isHigherHighs}, HL:${fifteenMinStructure.isHigherLows})`);
  
  // Determine bias
  let direction: 'BUY' | 'SELL' | null = null;
  let confidence = 0;
  
  if (hourlyTrend === 'ABOVE_200EMA' && fifteenMinStructure.trend === 'BULLISH') {
    direction = 'BUY';
    confidence = 75;
    reasons.push('✅ HTF Bias: BULLISH (1H > 200 EMA + 15m Bullish Structure)');
  } else if (hourlyTrend === 'BELOW_200EMA' && fifteenMinStructure.trend === 'BEARISH') {
    direction = 'SELL';
    confidence = 75;
    reasons.push('✅ HTF Bias: BEARISH (1H < 200 EMA + 15m Bearish Structure)');
  } else {
    reasons.push('❌ HTF Bias: NEUTRAL (Conflicting timeframes)');
  }
  
  return {
    direction,
    hourlyTrend,
    fifteenMinStructure,
    ema200,
    currentPrice,
    confidence,
    reasons,
  };
}

// ============================================
// LIQUIDITY SWEEP DETECTION
// ============================================

/**
 * Detect liquidity sweep
 * 
 * For BUY: Price must pierce the lowest low of last N candles 
 *          AND touch bb_lower/kc_lower
 *          AND close back above/inside the band
 * 
 * For SELL: Price must pierce the highest high of last N candles
 *           AND touch bb_upper/kc_upper
 *           AND close back below/inside the band
 */
export function detectLiquiditySweep(
  candles: Candle[],
  symbol: string,
  direction: 'BUY' | 'SELL'
): LiquiditySweep | null {
  if (candles.length < LIQUIDITY_CONFIG.SWEEP_LOOKBACK_CANDLES + 2) {
    return null;
  }
  
  const closes = candles.map(c => c.close);
  const atr = calculateATR(candles);
  const bb = calculateBollingerBands(closes);
  const kc = calculateKeltnerChannel(candles);
  
  // Get liquidity levels (swing lows for BUY, swing highs for SELL)
  const lookbackCandles = candles.slice(-LIQUIDITY_CONFIG.SWEEP_LOOKBACK_CANDLES - 1, -1);
  
  if (direction === 'BUY') {
    // Find lowest low in lookback period
    const liquidityLevel = Math.min(...lookbackCandles.map(c => c.low));
    
    // Check last few candles for a sweep
    for (let i = candles.length - 1; i >= candles.length - 3 && i >= 0; i--) {
      const candle = candles[i];
      
      // Price must pierce below the liquidity level
      const piercedLiquidity = candle.low < liquidityLevel;
      
      // And touch lower band (BB or KC)
      const touchedBBLower = candle.low <= bb.lower;
      const touchedKCLower = candle.low <= kc.lower;
      const touchedValueArea = touchedBBLower || touchedKCLower;
      
      // Must close back above the band (sweep confirmation)
      const tolerance = atr * LIQUIDITY_CONFIG.CLOSE_TOLERANCE_ATR;
      const closedInside = LIQUIDITY_CONFIG.REQUIRE_CLOSE_INSIDE
        ? candle.close >= bb.lower - tolerance || candle.close >= kc.lower - tolerance
        : true;
      
      if (piercedLiquidity && touchedValueArea && closedInside) {
        return {
          detected: true,
          type: 'BUY',
          sweepPrice: candle.low,
          sweepLow: candle.low,
          sweepHigh: candle.high,
          closedInside,
          candleIndex: i,
          timestamp: candle.timestamp,
          liquidityLevel,
        };
      }
    }
  } else {
    // Find highest high in lookback period
    const liquidityLevel = Math.max(...lookbackCandles.map(c => c.high));
    
    // Check last few candles for a sweep
    for (let i = candles.length - 1; i >= candles.length - 3 && i >= 0; i--) {
      const candle = candles[i];
      
      // Price must pierce above the liquidity level
      const piercedLiquidity = candle.high > liquidityLevel;
      
      // And touch upper band (BB or KC)
      const touchedBBUpper = candle.high >= bb.upper;
      const touchedKCUpper = candle.high >= kc.upper;
      const touchedValueArea = touchedBBUpper || touchedKCUpper;
      
      // Must close back below the band (sweep confirmation)
      const tolerance = atr * LIQUIDITY_CONFIG.CLOSE_TOLERANCE_ATR;
      const closedInside = LIQUIDITY_CONFIG.REQUIRE_CLOSE_INSIDE
        ? candle.close <= bb.upper + tolerance || candle.close <= kc.upper + tolerance
        : true;
      
      if (piercedLiquidity && touchedValueArea && closedInside) {
        return {
          detected: true,
          type: 'SELL',
          sweepPrice: candle.high,
          sweepLow: candle.low,
          sweepHigh: candle.high,
          closedInside,
          candleIndex: i,
          timestamp: candle.timestamp,
          liquidityLevel,
        };
      }
    }
  }
  
  return null;
}

// ============================================
// MARKET STRUCTURE SHIFT (MSS) DETECTION
// ============================================

/**
 * Detect Market Structure Shift
 * 
 * MSS = A strong displacement candle that breaks a recent swing point
 * in the OPPOSITE direction of the liquidity sweep
 */
export function detectMSS(
  candles: Candle[],
  sweepDirection: 'BUY' | 'SELL',
  sweepCandleIndex: number
): MarketStructureShift {
  const structure = analyzeMarketStructure(candles);
  const atr = calculateATR(candles);
  
  // Look for MSS after the sweep
  for (let i = sweepCandleIndex + 1; i < candles.length; i++) {
    const candle = candles[i];
    const bodySize = Math.abs(candle.close - candle.open);
    const minBodySize = atr * MSS_CONFIG.MIN_DISPLACEMENT_BODY_ATR;
    
    // Check for displacement candle (strong body)
    const isDisplacementCandle = bodySize >= minBodySize;
    
    if (!isDisplacementCandle) continue;
    
    if (sweepDirection === 'BUY') {
      // After BUY sweep, look for BULLISH MSS
      // Displacement candle must be bullish and break above recent swing high
      const isBullishDisplacement = candle.close > candle.open;
      
      if (isBullishDisplacement && structure.lastSwingHigh) {
        const brokeSwinghigh = candle.close > structure.lastSwingHigh.price;
        
        if (brokeSwinghigh || !MSS_CONFIG.REQUIRE_SWING_BREAK) {
          return {
            detected: true,
            direction: 'BULLISH',
            displacementCandle: candle,
            displacementIndex: i,
            swingBroken: brokeSwinghigh ? structure.lastSwingHigh : null,
          };
        }
      }
    } else {
      // After SELL sweep, look for BEARISH MSS
      // Displacement candle must be bearish and break below recent swing low
      const isBearishDisplacement = candle.close < candle.open;
      
      if (isBearishDisplacement && structure.lastSwingLow) {
        const brokeSwingLow = candle.close < structure.lastSwingLow.price;
        
        if (brokeSwingLow || !MSS_CONFIG.REQUIRE_SWING_BREAK) {
          return {
            detected: true,
            direction: 'BEARISH',
            displacementCandle: candle,
            displacementIndex: i,
            swingBroken: brokeSwingLow ? structure.lastSwingLow : null,
          };
        }
      }
    }
  }
  
  return {
    detected: false,
    direction: sweepDirection === 'BUY' ? 'BULLISH' : 'BEARISH',
    displacementCandle: null,
    displacementIndex: -1,
    swingBroken: null,
  };
}

// ============================================
// FAIR VALUE GAP (FVG) DETECTION
// ============================================

/**
 * Detect Fair Value Gap (FVG)
 * 
 * FVG = A 3-candle sequence where:
 * - BULLISH FVG: High of candle 1 < Low of candle 3 (gap up)
 * - BEARISH FVG: Low of candle 1 > High of candle 3 (gap down)
 * 
 * Entry is at 50% (equilibrium) of the FVG
 */
export function detectFVG(
  candles: Candle[],
  direction: 'BUY' | 'SELL',
  startFromIndex: number,
  symbol: string
): FairValueGap | null {
  const minGapPrice = pipToPrice(FVG_CONFIG.MIN_GAP_PIPS, symbol);
  
  // Scan for FVG starting from after MSS
  for (let i = Math.max(startFromIndex, 2); i < candles.length - 1; i++) {
    const candle1 = candles[i - 2];
    const candle2 = candles[i - 1]; // Displacement candle
    const candle3 = candles[i];
    
    if (direction === 'BUY') {
      // BULLISH FVG: Gap between candle1 high and candle3 low
      if (candle3.low > candle1.high) {
        const gapSize = candle3.low - candle1.high;
        
        if (gapSize >= minGapPrice) {
          const equilibrium = candle1.high + (gapSize * FVG_CONFIG.ENTRY_EQUILIBRIUM_LEVEL);
          const age = candles.length - 1 - i;
          
          return {
            detected: true,
            type: 'BULLISH',
            topPrice: candle3.low,
            bottomPrice: candle1.high,
            equilibrium,
            gapSizePips: priceToPips(gapSize, symbol),
            startIndex: i,
            isValid: age <= FVG_CONFIG.MAX_FVG_AGE_CANDLES,
            age,
          };
        }
      }
    } else {
      // BEARISH FVG: Gap between candle1 low and candle3 high
      if (candle3.high < candle1.low) {
        const gapSize = candle1.low - candle3.high;
        
        if (gapSize >= minGapPrice) {
          const equilibrium = candle1.low - (gapSize * FVG_CONFIG.ENTRY_EQUILIBRIUM_LEVEL);
          const age = candles.length - 1 - i;
          
          return {
            detected: true,
            type: 'BEARISH',
            topPrice: candle1.low,
            bottomPrice: candle3.high,
            equilibrium,
            gapSizePips: priceToPips(gapSize, symbol),
            startIndex: i,
            isValid: age <= FVG_CONFIG.MAX_FVG_AGE_CANDLES,
            age,
          };
        }
      }
    }
  }
  
  return null;
}

// ============================================
// FULL ICT SIGNAL GENERATION
// ============================================

/**
 * Generate ICT/SMC Signal
 * 
 * Signal Flow:
 * 1. HTF Bias (1H > 200 EMA + 15m Bullish/Bearish Structure)
 * 2. Liquidity Sweep (pierce + value area touch + close inside)
 * 3. Market Structure Shift (displacement candle)
 * 4. Fair Value Gap (50% equilibrium entry)
 * 5. Calculate SL (2 pips below sweep low) and TP (2.2x RRR)
 */
export function generateICTSignal(
  hourlyCandles: Candle[],
  fifteenMinCandles: Candle[],
  oneMinCandles: Candle[],
  symbol: string
): ICTSignal {
  const invalidReasons: string[] = [];
  const reasons: string[] = [];
  
  // Step 1: Calculate HTF Bias
  const htfBias = calculateHTFBias(hourlyCandles, fifteenMinCandles);
  reasons.push(...htfBias.reasons);
  
  if (!htfBias.direction) {
    invalidReasons.push('No HTF bias - conflicting timeframe signals');
    return {
      valid: false,
      direction: null,
      entryPrice: 0,
      stopLoss: 0,
      takeProfit: 0,
      htfBias,
      liquiditySweep: null,
      fvg: null,
      mss: null,
      confidence: 0,
      reasons,
      invalidReasons,
    };
  }
  
  // Step 2: Detect Liquidity Sweep
  const liquiditySweep = detectLiquiditySweep(oneMinCandles, symbol, htfBias.direction);
  
  if (!liquiditySweep) {
    invalidReasons.push(`No liquidity sweep detected for ${htfBias.direction} direction`);
    return {
      valid: false,
      direction: null,
      entryPrice: 0,
      stopLoss: 0,
      takeProfit: 0,
      htfBias,
      liquiditySweep: null,
      fvg: null,
      mss: null,
      confidence: 0,
      reasons,
      invalidReasons,
    };
  }
  
  reasons.push(`✅ Liquidity sweep detected at ${liquiditySweep.sweepPrice.toFixed(5)}`);
  
  // Step 3: Detect Market Structure Shift (MSS)
  const mss = detectMSS(oneMinCandles, htfBias.direction, liquiditySweep.candleIndex);
  
  if (!mss.detected) {
    invalidReasons.push('No market structure shift (MSS) after liquidity sweep');
    return {
      valid: false,
      direction: null,
      entryPrice: 0,
      stopLoss: 0,
      takeProfit: 0,
      htfBias,
      liquiditySweep,
      fvg: null,
      mss: null,
      confidence: 0,
      reasons,
      invalidReasons,
    };
  }
  
  reasons.push(`✅ MSS detected: ${mss.direction} displacement at candle ${mss.displacementIndex}`);
  
  // Step 4: Detect Fair Value Gap (FVG)
  const fvg = detectFVG(oneMinCandles, htfBias.direction, mss.displacementIndex, symbol);
  
  if (!fvg || !fvg.isValid) {
    invalidReasons.push(fvg ? `FVG too old (${fvg.age} candles)` : 'No Fair Value Gap detected');
    return {
      valid: false,
      direction: null,
      entryPrice: 0,
      stopLoss: 0,
      takeProfit: 0,
      htfBias,
      liquiditySweep,
      fvg,
      mss,
      confidence: 0,
      reasons,
      invalidReasons,
    };
  }
  
  reasons.push(`✅ FVG detected: ${fvg.gapSizePips.toFixed(1)} pips, entry at ${fvg.equilibrium.toFixed(5)}`);
  
  // Step 5: Calculate entry, SL, and TP
  const entryPrice = fvg.equilibrium;
  const slBufferPrice = pipToPrice(FVG_CONFIG.SL_BUFFER_PIPS, symbol);
  
  let stopLoss: number;
  let takeProfit: number;
  
  if (htfBias.direction === 'BUY') {
    stopLoss = liquiditySweep.sweepLow - slBufferPrice;
    const riskDistance = entryPrice - stopLoss;
    takeProfit = entryPrice + (riskDistance * ICT_CONFIG.RRR_RATIO);
  } else {
    stopLoss = liquiditySweep.sweepHigh + slBufferPrice;
    const riskDistance = stopLoss - entryPrice;
    takeProfit = entryPrice - (riskDistance * ICT_CONFIG.RRR_RATIO);
  }
  
  // Calculate confidence
  let confidence = htfBias.confidence;
  if (liquiditySweep.closedInside) confidence += 5;
  if (mss.swingBroken) confidence += 5;
  if (fvg.gapSizePips >= 3) confidence += 5;
  confidence = Math.min(95, confidence);
  
  reasons.push(`📊 Entry: ${entryPrice.toFixed(5)}, SL: ${stopLoss.toFixed(5)}, TP: ${takeProfit.toFixed(5)}`);
  reasons.push(`📈 Risk/Reward: 1:${ICT_CONFIG.RRR_RATIO}`);
  
  return {
    valid: true,
    direction: htfBias.direction,
    entryPrice,
    stopLoss,
    takeProfit,
    htfBias,
    liquiditySweep,
    fvg,
    mss,
    confidence,
    reasons,
    invalidReasons,
  };
}

export default {
  calculateHTFBias,
  detectLiquiditySweep,
  detectMSS,
  detectFVG,
  generateICTSignal,
  analyzeMarketStructure,
  calculateEMA,
  calculateATR,
  calculateBollingerBands,
  calculateKeltnerChannel,
};
