import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface TechnicalIndicators {
  rsi: number;
  macd: { value: number; signal: number; histogram: number };
  ema9: number;
  ema21: number;
  ema50: number;
  ema200: number;
  bollingerBands: { upper: number; middle: number; lower: number };
  stochastic: { k: number; d: number };
  atr: number;
  supportLevels: number[];
  resistanceLevels: number[];
}

interface SignalConfirmation {
  canBuy: boolean;
  canSell: boolean;
  buyReasons: string[];
  sellReasons: string[];
  conflicts: string[];
}

// Check if forex market is open (forex is closed on weekends)
function isForexMarketOpen(): { isOpen: boolean; reason: string } {
  const now = new Date();
  const utcDay = now.getUTCDay();
  const utcHour = now.getUTCHours();
  
  // Forex market opens Sunday 21:00 UTC and closes Friday 21:00 UTC
  if (utcDay === 6) {
    return { isOpen: false, reason: "Forex market is closed on Saturdays" };
  }
  if (utcDay === 0 && utcHour < 21) {
    return { isOpen: false, reason: "Forex market opens Sunday 21:00 UTC" };
  }
  if (utcDay === 5 && utcHour >= 21) {
    return { isOpen: false, reason: "Forex market closed Friday 21:00 UTC" };
  }
  
  return { isOpen: true, reason: "Market is open" };
}

// Check if price data is stale (hasn't changed in a while)
function isPriceStale(candles: Candle[]): { isStale: boolean; reason: string } {
  if (candles.length < 5) {
    return { isStale: false, reason: "Not enough data to determine staleness" };
  }
  
  const last5Closes = candles.slice(-5).map(c => c.close);
  const allSame = last5Closes.every(c => c === last5Closes[0]);
  
  if (allSame) {
    return { isStale: true, reason: "Price has not changed in last 5 candles - likely stale data" };
  }
  
  return { isStale: false, reason: "Price data is fresh" };
}

// Calculate RSI
function calculateRSI(closes: number[], period = 14): number {
  if (closes.length < period + 1) return 50;
  
  let gains = 0;
  let losses = 0;
  
  for (let i = closes.length - period; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Calculate EMA
function calculateEMA(data: number[], period: number): number {
  if (data.length < period) return data[data.length - 1] || 0;
  
  const multiplier = 2 / (period + 1);
  let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  for (let i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

// Calculate MACD with proper signal line
function calculateMACD(closes: number[]): { value: number; signal: number; histogram: number } {
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);
  const macdLine = ema12 - ema26;
  
  // Calculate MACD history for signal line
  const macdHistory: number[] = [];
  for (let i = 26; i < closes.length; i++) {
    const shortEma = calculateEMA(closes.slice(0, i + 1), 12);
    const longEma = calculateEMA(closes.slice(0, i + 1), 26);
    macdHistory.push(shortEma - longEma);
  }
  
  const signalLine = macdHistory.length >= 9 ? calculateEMA(macdHistory, 9) : macdLine;
  
  return {
    value: macdLine,
    signal: signalLine,
    histogram: macdLine - signalLine
  };
}

// Calculate Bollinger Bands
function calculateBollingerBands(closes: number[], period = 20): { upper: number; middle: number; lower: number } {
  if (closes.length < period) {
    const last = closes[closes.length - 1];
    return { upper: last, middle: last, lower: last };
  }
  
  const slice = closes.slice(-period);
  const sma = slice.reduce((a, b) => a + b, 0) / period;
  const variance = slice.reduce((a, b) => a + Math.pow(b - sma, 2), 0) / period;
  const stdDev = Math.sqrt(variance);
  
  return {
    upper: sma + stdDev * 2,
    middle: sma,
    lower: sma - stdDev * 2,
  };
}

// Calculate Stochastic with proper %D
function calculateStochastic(highs: number[], lows: number[], closes: number[], period = 14): { k: number; d: number } {
  if (closes.length < period) return { k: 50, d: 50 };
  
  const highSlice = highs.slice(-period);
  const lowSlice = lows.slice(-period);
  const currentClose = closes[closes.length - 1];
  
  const highestHigh = Math.max(...highSlice);
  const lowestLow = Math.min(...lowSlice);
  
  if (highestHigh === lowestLow) return { k: 50, d: 50 };
  
  const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
  
  // Calculate %D (3-period SMA of %K)
  const kValues: number[] = [];
  for (let i = period; i <= closes.length; i++) {
    const h = Math.max(...highs.slice(i - period, i));
    const l = Math.min(...lows.slice(i - period, i));
    const c = closes[i - 1];
    kValues.push(h === l ? 50 : ((c - l) / (h - l)) * 100);
  }
  
  const d = kValues.length >= 3 
    ? kValues.slice(-3).reduce((a, b) => a + b, 0) / 3 
    : k;
  
  return { k, d };
}

// Calculate ATR (Average True Range)
function calculateATR(highs: number[], lows: number[], closes: number[], period = 14): number {
  if (closes.length < period + 1) return 0;
  
  const trueRanges: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }
  
  return trueRanges.slice(-period).reduce((a, b) => a + b, 0) / period;
}

// Calculate Support and Resistance levels
function calculateSupportResistance(highs: number[], lows: number[]): { support: number[]; resistance: number[] } {
  const support: number[] = [];
  const resistance: number[] = [];
  
  const lookback = Math.min(100, highs.length);
  const recentHighs = highs.slice(-lookback);
  const recentLows = lows.slice(-lookback);
  
  for (let i = 2; i < lookback - 2; i++) {
    // Swing high
    if (recentHighs[i] > recentHighs[i-1] && recentHighs[i] > recentHighs[i-2] &&
        recentHighs[i] > recentHighs[i+1] && recentHighs[i] > recentHighs[i+2]) {
      resistance.push(recentHighs[i]);
    }
    // Swing low
    if (recentLows[i] < recentLows[i-1] && recentLows[i] < recentLows[i-2] &&
        recentLows[i] < recentLows[i+1] && recentLows[i] < recentLows[i+2]) {
      support.push(recentLows[i]);
    }
  }
  
  return {
    support: support.sort((a, b) => b - a).slice(0, 3),
    resistance: resistance.sort((a, b) => a - b).slice(0, 3)
  };
}

// Enhanced pattern detection using more historical data
function detectPatterns(candles: Candle[]): string[] {
  const patterns: string[] = [];
  if (candles.length < 50) return patterns;
  
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const opens = candles.map(c => c.open);
  
  const last = candles.length - 1;
  
  // Doji detection (last 3 candles)
  for (let i = last; i > last - 3 && i >= 0; i--) {
    const body = Math.abs(closes[i] - opens[i]);
    const range = highs[i] - lows[i];
    if (range > 0 && body / range < 0.1) {
      patterns.push('Doji - Indecision');
      break;
    }
  }
  
  // Engulfing patterns
  if (last >= 1) {
    const prevBody = Math.abs(closes[last-1] - opens[last-1]);
    const currBody = Math.abs(closes[last] - opens[last]);
    
    // Bullish engulfing
    if (closes[last-1] < opens[last-1] && closes[last] > opens[last] &&
        opens[last] <= closes[last-1] && closes[last] >= opens[last-1] && currBody > prevBody) {
      patterns.push('Bullish Engulfing - Reversal Signal');
    }
    
    // Bearish engulfing
    if (closes[last-1] > opens[last-1] && closes[last] < opens[last] &&
        opens[last] >= closes[last-1] && closes[last] <= opens[last-1] && currBody > prevBody) {
      patterns.push('Bearish Engulfing - Reversal Signal');
    }
  }
  
  // Double Top/Bottom detection (last 50 candles)
  const last50Highs = highs.slice(-50);
  const last50Lows = lows.slice(-50);
  const maxHigh = Math.max(...last50Highs);
  const minLow = Math.min(...last50Lows);
  const tolerance = (maxHigh - minLow) * 0.02;
  
  // Find double tops
  const highPeaks: number[] = [];
  for (let i = 2; i < last50Highs.length - 2; i++) {
    if (last50Highs[i] > last50Highs[i-1] && last50Highs[i] > last50Highs[i-2] &&
        last50Highs[i] > last50Highs[i+1] && last50Highs[i] > last50Highs[i+2]) {
      highPeaks.push(last50Highs[i]);
    }
  }
  
  if (highPeaks.length >= 2) {
    const [peak1, peak2] = highPeaks.slice(-2);
    if (Math.abs(peak1 - peak2) < tolerance) {
      patterns.push('Double Top Formation - Bearish Reversal');
    }
  }
  
  // Find double bottoms
  const lowTroughs: number[] = [];
  for (let i = 2; i < last50Lows.length - 2; i++) {
    if (last50Lows[i] < last50Lows[i-1] && last50Lows[i] < last50Lows[i-2] &&
        last50Lows[i] < last50Lows[i+1] && last50Lows[i] < last50Lows[i+2]) {
      lowTroughs.push(last50Lows[i]);
    }
  }
  
  if (lowTroughs.length >= 2) {
    const [trough1, trough2] = lowTroughs.slice(-2);
    if (Math.abs(trough1 - trough2) < tolerance) {
      patterns.push('Double Bottom Formation - Bullish Reversal');
    }
  }
  
  // Trend identification using EMAs
  const ema20 = calculateEMA(closes, 20);
  const ema50 = calculateEMA(closes, 50);
  const currentPrice = closes[last];
  
  if (currentPrice > ema20 && ema20 > ema50) {
    patterns.push('Strong Uptrend - Price above EMAs');
  } else if (currentPrice < ema20 && ema20 < ema50) {
    patterns.push('Strong Downtrend - Price below EMAs');
  } else if (currentPrice > ema20 && ema20 < ema50) {
    patterns.push('Potential Trend Reversal - Bullish crossover forming');
  } else if (currentPrice < ema20 && ema20 > ema50) {
    patterns.push('Potential Trend Reversal - Bearish crossover forming');
  }
  
  // Higher highs / Lower lows (last 20 candles)
  const last20 = candles.slice(-20);
  const recentHighs = last20.map(c => c.high);
  const recentLows = last20.map(c => c.low);
  
  let higherHighs = 0;
  let lowerLows = 0;
  
  for (let i = 1; i < recentHighs.length; i++) {
    if (recentHighs[i] > recentHighs[i-1]) higherHighs++;
    if (recentLows[i] < recentLows[i-1]) lowerLows++;
  }
  
  if (higherHighs > 12) patterns.push('Higher Highs Pattern - Bullish Momentum');
  if (lowerLows > 12) patterns.push('Lower Lows Pattern - Bearish Momentum');
  
  // Near support/resistance
  const { support, resistance } = calculateSupportResistance(highs, lows);
  if (support.length > 0 && Math.abs(currentPrice - support[0]) < tolerance * 2) {
    patterns.push('Near Support Level');
  }
  if (resistance.length > 0 && Math.abs(currentPrice - resistance[0]) < tolerance * 2) {
    patterns.push('Near Resistance Level');
  }
  
  return patterns;
}

// Calculate all technical indicators
function calculateIndicators(candles: Candle[]): TechnicalIndicators {
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  
  const { support, resistance } = calculateSupportResistance(highs, lows);
  
  return {
    rsi: calculateRSI(closes),
    macd: calculateMACD(closes),
    ema9: calculateEMA(closes, 9),
    ema21: calculateEMA(closes, 21),
    ema50: calculateEMA(closes, 50),
    ema200: calculateEMA(closes, 200),
    bollingerBands: calculateBollingerBands(closes),
    stochastic: calculateStochastic(highs, lows, closes),
    atr: calculateATR(highs, lows, closes),
    supportLevels: support,
    resistanceLevels: resistance
  };
}

// Pattern Tier System based on 25 years historical data
const PATTERN_WEIGHTS = {
  // Tier 1: Proven winners (>52% win rate)
  rsi_oversold: { weight: 1.5, tier: 1, expectedWinRate: 52.4 },
  rsi_overbought: { weight: 1.5, tier: 1, expectedWinRate: 52.16 },
  bb_lower_touch: { weight: 1.3, tier: 1, expectedWinRate: 52.06 },
  bb_upper_touch: { weight: 1.3, tier: 1, expectedWinRate: 51.67 },
  
  // Tier 2: Neutral (50-52%)
  stochastic_oversold: { weight: 1.0, tier: 2, expectedWinRate: 50 },
  stochastic_overbought: { weight: 1.0, tier: 2, expectedWinRate: 50 },
  
  // Tier 3: Weak (<50%)
  macd_bullish_cross: { weight: 0.5, tier: 3, expectedWinRate: 47.85 },
  macd_bearish_cross: { weight: 0.5, tier: 3, expectedWinRate: 47.39 },
  bullish_engulfing: { weight: 0.5, tier: 3, expectedWinRate: 48.39 },
  bearish_engulfing: { weight: 0.5, tier: 3, expectedWinRate: 48.24 },
  
  // Tier 4: Harmful (<47%)
  golden_cross: { weight: -0.5, tier: 4, expectedWinRate: 45.89 },
  death_cross: { weight: -0.5, tier: 4, expectedWinRate: 45.87 },
} as const;

interface PatternTierInfo {
  name: string;
  tier: number;
  weight: number;
  reason: string;
}

// Signal confirmation filters with tier-based weighting
function getSignalConfirmation(indicators: TechnicalIndicators, patterns: string[], currentPrice: number): SignalConfirmation {
  const buyReasons: string[] = [];
  const sellReasons: string[] = [];
  const conflicts: string[] = [];
  
  const buyPatterns: PatternTierInfo[] = [];
  const sellPatterns: PatternTierInfo[] = [];
  
  // RSI - Tier 1
  if (indicators.rsi < 30) {
    buyReasons.push('RSI oversold (<30) - Tier 1 HIGH PROBABILITY');
    buyPatterns.push({ name: 'rsi_oversold', tier: 1, weight: 1.5, reason: 'RSI oversold' });
  } else if (indicators.rsi < 40) {
    buyReasons.push('RSI approaching oversold (30-40)');
    buyPatterns.push({ name: 'rsi_approaching', tier: 2, weight: 0.5, reason: 'RSI approaching oversold' });
  } else if (indicators.rsi > 70) {
    sellReasons.push('RSI overbought (>70) - Tier 1 HIGH PROBABILITY');
    sellPatterns.push({ name: 'rsi_overbought', tier: 1, weight: 1.5, reason: 'RSI overbought' });
  } else if (indicators.rsi > 60) {
    sellReasons.push('RSI approaching overbought (60-70)');
    sellPatterns.push({ name: 'rsi_approaching', tier: 2, weight: 0.5, reason: 'RSI approaching overbought' });
  }
  
  // Bollinger Bands - Tier 1
  if (currentPrice < indicators.bollingerBands.lower) {
    buyReasons.push('Price below lower Bollinger Band - Tier 1 HIGH PROBABILITY');
    buyPatterns.push({ name: 'bb_lower_touch', tier: 1, weight: 1.3, reason: 'BB lower touch' });
  } else if (currentPrice > indicators.bollingerBands.upper) {
    sellReasons.push('Price above upper Bollinger Band - Tier 1 HIGH PROBABILITY');
    sellPatterns.push({ name: 'bb_upper_touch', tier: 1, weight: 1.3, reason: 'BB upper touch' });
  }
  
  // MACD - Tier 3 (weak, low weight)
  if (indicators.macd.histogram > 0 && indicators.macd.value > indicators.macd.signal) {
    buyReasons.push('MACD bullish - Tier 3 (weak edge)');
    buyPatterns.push({ name: 'macd_bullish_cross', tier: 3, weight: 0.5, reason: 'MACD bullish' });
  } else if (indicators.macd.histogram < 0 && indicators.macd.value < indicators.macd.signal) {
    sellReasons.push('MACD bearish - Tier 3 (weak edge)');
    sellPatterns.push({ name: 'macd_bearish_cross', tier: 3, weight: 0.5, reason: 'MACD bearish' });
  }
  
  // MACD near crossover - only flag if very close
  if (Math.abs(indicators.macd.histogram) < 0.00002) {
    conflicts.push('MACD near crossover');
  }
  
  // EMA alignment - Tier 4 (harmful based on historical data!)
  const ema21AboveEma50 = indicators.ema21 > indicators.ema50;
  const priceAboveEma21 = currentPrice > indicators.ema21;
  const priceAboveEma50 = currentPrice > indicators.ema50;
  
  if (priceAboveEma21 && priceAboveEma50 && ema21AboveEma50) {
    // Golden cross is historically harmful - add as negative signal
    buyReasons.push('Golden Cross - Tier 4 ⚠️ (historically <46% win rate)');
    buyPatterns.push({ name: 'golden_cross', tier: 4, weight: -0.5, reason: 'Golden cross - contrarian' });
  } else if (!priceAboveEma21 && !priceAboveEma50 && !ema21AboveEma50) {
    sellReasons.push('Death Cross - Tier 4 ⚠️ (historically <46% win rate)');
    sellPatterns.push({ name: 'death_cross', tier: 4, weight: -0.5, reason: 'Death cross - contrarian' });
  }
  
  // Stochastic - Tier 2
  if (indicators.stochastic.k < 20 && indicators.stochastic.d < 20) {
    buyReasons.push('Stochastic deeply oversold (<20)');
    buyPatterns.push({ name: 'stochastic_oversold', tier: 2, weight: 1.0, reason: 'Stochastic oversold' });
  } else if (indicators.stochastic.k > 80 && indicators.stochastic.d > 80) {
    sellReasons.push('Stochastic deeply overbought (>80)');
    sellPatterns.push({ name: 'stochastic_overbought', tier: 2, weight: 1.0, reason: 'Stochastic overbought' });
  }
  
  // Pattern analysis - Tier 3
  const bullishPatterns = patterns.filter(p => 
    p.includes('Bullish') || p.includes('Uptrend') || p.includes('Higher Highs') || p.includes('Support')
  );
  const bearishPatterns = patterns.filter(p => 
    p.includes('Bearish') || p.includes('Downtrend') || p.includes('Lower Lows') || p.includes('Resistance')
  );
  
  if (bullishPatterns.length > 0) {
    buyReasons.push(`Bullish patterns detected: ${bullishPatterns.length} - Tier 3`);
    buyPatterns.push({ name: 'bullish_patterns', tier: 3, weight: 0.5, reason: 'Bullish patterns' });
  }
  if (bearishPatterns.length > 0) {
    sellReasons.push(`Bearish patterns detected: ${bearishPatterns.length} - Tier 3`);
    sellPatterns.push({ name: 'bearish_patterns', tier: 3, weight: 0.5, reason: 'Bearish patterns' });
  }
  
  if (bullishPatterns.length >= 2 && bearishPatterns.length >= 2) {
    conflicts.push('Strong conflicting patterns detected');
  }
  
  // Calculate weighted scores
  const buyScore = buyPatterns.reduce((sum, p) => sum + p.weight, 0);
  const sellScore = sellPatterns.reduce((sum, p) => sum + p.weight, 0);
  
  // Count Tier 1 patterns (required for signal)
  const buyTier1Count = buyPatterns.filter(p => p.tier === 1).length;
  const sellTier1Count = sellPatterns.filter(p => p.tier === 1).length;
  
  // New requirements:
  // 1. MUST have at least 1 Tier 1 pattern
  // 2. Weighted score must exceed threshold (1.5)
  // 3. Must have net advantage
  const SCORE_THRESHOLD = 1.5;
  
  const canBuy = buyTier1Count >= 1 && buyScore >= SCORE_THRESHOLD && (buyScore - sellScore) >= 0.5 && conflicts.length <= 2;
  const canSell = sellTier1Count >= 1 && sellScore >= SCORE_THRESHOLD && (sellScore - buyScore) >= 0.5 && conflicts.length <= 2;
  
  console.log(`Tier analysis: BUY T1=${buyTier1Count} score=${buyScore.toFixed(2)}, SELL T1=${sellTier1Count} score=${sellScore.toFixed(2)}`);
  console.log(`Can trade: BUY=${canBuy}, SELL=${canSell}`);
  
  return { canBuy, canSell, buyReasons, sellReasons, conflicts };
}

// Calculate ATR-based stop loss and take profit
function calculateATRBasedLevels(
  currentPrice: number, 
  atr: number, 
  signalType: 'BUY' | 'SELL',
  timeframe: string
): { stopLoss: number; takeProfit1: number; takeProfit2: number } {
  // ATR multipliers based on timeframe
  const multipliers: Record<string, { sl: number; tp1: number; tp2: number }> = {
    '15m': { sl: 1.5, tp1: 1.5, tp2: 2.5 },
    '1h': { sl: 2.0, tp1: 2.0, tp2: 3.5 },
    '4h': { sl: 2.5, tp1: 3.0, tp2: 5.0 },
    '1d': { sl: 3.0, tp1: 4.0, tp2: 6.0 }
  };
  
  const mult = multipliers[timeframe] || multipliers['1h'];
  
  if (signalType === 'BUY') {
    return {
      stopLoss: currentPrice - (atr * mult.sl),
      takeProfit1: currentPrice + (atr * mult.tp1),
      takeProfit2: currentPrice + (atr * mult.tp2)
    };
  } else {
    return {
      stopLoss: currentPrice + (atr * mult.sl),
      takeProfit1: currentPrice - (atr * mult.tp1),
      takeProfit2: currentPrice - (atr * mult.tp2)
    };
  }
}

// Get timeframe-specific settings
function getTimeframeSettings(timeframe: string) {
  const settings: Record<string, { expiryHours: number; pipMultiplier: number; description: string }> = {
    '15m': { expiryHours: 1, pipMultiplier: 0.5, description: 'Short-term scalping' },
    '1h': { expiryHours: 4, pipMultiplier: 1, description: 'Intraday trading' },
    '4h': { expiryHours: 24, pipMultiplier: 2, description: 'Swing trading' },
    '1d': { expiryHours: 72, pipMultiplier: 4, description: 'Position trading' }
  };
  return settings[timeframe] || settings['1h'];
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { candles, currentPrice, timeframe = '1h', sentimentScore = 0 } = await req.json();
    
    if (!candles || !Array.isArray(candles) || candles.length < 50) {
      throw new Error("At least 50 candles are required for analysis");
    }

    console.log(`Generating prediction for ${candles.length} candles, timeframe: ${timeframe}, current price: ${currentPrice}`);

    // Check if market is open
    const marketStatus = isForexMarketOpen();
    if (!marketStatus.isOpen) {
      console.log("Market closed:", marketStatus.reason);
      return new Response(
        JSON.stringify({ 
          success: false, 
          error: marketStatus.reason,
          marketClosed: true
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Check for stale price data
    const staleCheck = isPriceStale(candles);
    if (staleCheck.isStale) {
      console.log("Stale data detected:", staleCheck.reason);
    }

    const indicators = calculateIndicators(candles);
    const patterns = detectPatterns(candles);
    const timeframeSettings = getTimeframeSettings(timeframe);
    const signalConfirmation = getSignalConfirmation(indicators, patterns, currentPrice);

    console.log("Technical indicators:", JSON.stringify(indicators));
    console.log("Patterns detected:", patterns);
    console.log("Signal confirmation:", JSON.stringify(signalConfirmation));

    // If no actionable signal can be generated, return early with explanation
    if (!signalConfirmation.canBuy && !signalConfirmation.canSell) {
      console.log("No actionable signal - market conditions unclear");
      return new Response(
        JSON.stringify({
          success: false,
          noSignal: true,
          message: "Market conditions unclear - no actionable signal",
          reason: signalConfirmation.conflicts.length > 0 
            ? `Conflicts: ${signalConfirmation.conflicts.join(', ')}`
            : "Insufficient confirmation for BUY or SELL",
          buyReasons: signalConfirmation.buyReasons,
          sellReasons: signalConfirmation.sellReasons,
          conflicts: signalConfirmation.conflicts
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Fetch last 15 predictions with outcomes for learning
    const { data: pastPredictions } = await supabase
      .from('predictions')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(15);

    // Fetch learnings from past predictions
    const { data: learnings } = await supabase
      .from('prediction_learnings')
      .select('lesson_extracted, success_factors, failure_reason')
      .order('created_at', { ascending: false })
      .limit(10);

    // Fetch historical pattern statistics (25 years of data)
    const { data: patternStats } = await supabase
      .from('pattern_statistics')
      .select('*');

    console.log(`Loaded ${patternStats?.length || 0} pattern statistics`);

    // Build historical pattern statistics context
    let patternStatsContext = '';
    if (patternStats && patternStats.length > 0) {
      // Map detected patterns to database pattern names
      const patternMapping: Record<string, string[]> = {
        'rsi_oversold': ['RSI oversold', 'OVERSOLD', 'rsi < 30'],
        'rsi_overbought': ['RSI overbought', 'OVERBOUGHT', 'rsi > 70'],
        'macd_bullish_cross': ['MACD bullish', 'Bullish crossover'],
        'macd_bearish_cross': ['MACD bearish', 'Bearish crossover'],
        'bb_lower_touch': ['Bollinger Lower', 'lower Bollinger', 'below lower'],
        'bb_upper_touch': ['Bollinger Upper', 'upper Bollinger', 'above upper'],
        'bullish_engulfing': ['Bullish Engulfing'],
        'bearish_engulfing': ['Bearish Engulfing'],
        'golden_cross': ['Strong Uptrend', 'Golden Cross', 'EMA alignment bullish'],
        'death_cross': ['Strong Downtrend', 'Death Cross', 'EMA alignment bearish'],
      };

      // Find which patterns are currently detected
      const detectedDbPatterns: string[] = [];
      patterns.forEach((pattern: string) => {
        Object.entries(patternMapping).forEach(([dbName, keywords]) => {
          if (keywords.some(kw => pattern.toLowerCase().includes(kw.toLowerCase()))) {
            if (!detectedDbPatterns.includes(dbName)) {
              detectedDbPatterns.push(dbName);
            }
          }
        });
      });

      // Check indicator conditions
      if (indicators.rsi < 30) detectedDbPatterns.push('rsi_oversold');
      if (indicators.rsi > 70) detectedDbPatterns.push('rsi_overbought');
      if (currentPrice < indicators.bollingerBands.lower) detectedDbPatterns.push('bb_lower_touch');
      if (currentPrice > indicators.bollingerBands.upper) detectedDbPatterns.push('bb_upper_touch');
      if (indicators.macd.histogram > 0 && indicators.macd.value > indicators.macd.signal) {
        detectedDbPatterns.push('macd_bullish_cross');
      }
      if (indicators.macd.histogram < 0 && indicators.macd.value < indicators.macd.signal) {
        detectedDbPatterns.push('macd_bearish_cross');
      }

      // Unique patterns
      const uniqueDetected = [...new Set(detectedDbPatterns)];

      // Get stats for detected patterns
      const matchedStats = patternStats.filter((s: any) => uniqueDetected.includes(s.pattern_name));
      const goodPatterns = matchedStats.filter((s: any) => s.win_rate_24h >= 51);
      const badPatterns = matchedStats.filter((s: any) => s.win_rate_24h < 48);

      patternStatsContext = `
HISTORICAL PATTERN STATISTICS (25 years of EUR/USD M1 data, 2001-2025):

PATTERN TIER SYSTEM (based on historical win rates):
🏆 TIER 1 - HIGH PROBABILITY (>52% win rate) - REQUIRED for signals:
   • RSI Oversold (<30): 52.4% win rate - STRONG BUY signal
   • RSI Overbought (>70): 52.2% win rate - STRONG SELL signal  
   • BB Lower Touch: 52.1% win rate - STRONG BUY signal
   • BB Upper Touch: 51.7% win rate - STRONG SELL signal

⚖️ TIER 2 - NEUTRAL (50-52% win rate) - Confirmation only:
   • Stochastic extremes: ~50% win rate

⚠️ TIER 3 - WEAK (<50% win rate) - Low weight:
   • MACD Crossovers: 47-48% win rate - weak signals
   • Engulfing patterns: 48% win rate - weak signals

❌ TIER 4 - HARMFUL (<47% win rate) - PENALIZE:
   • Golden Cross: 45.9% win rate - CONTRARIAN indicator!
   • Death Cross: 45.9% win rate - CONTRARIAN indicator!

${matchedStats.length > 0 ? `CURRENTLY DETECTED PATTERNS:
${matchedStats.map((s: any) => 
  `- ${s.pattern_name.replace(/_/g, ' ').toUpperCase()} (${s.signal_type}): ${s.win_rate_24h?.toFixed(1)}% win rate (n=${s.occurrences.toLocaleString()})`
).join('\n')}` : 'No patterns with historical data currently detected.'}

${goodPatterns.length > 0 ? `✅ TIER 1 PATTERNS DETECTED - Signal is valid:
${goodPatterns.map((s: any) => `- ${s.pattern_name}: ${s.win_rate_24h?.toFixed(1)}% win rate`).join('\n')}` : '⚠️ NO TIER 1 PATTERNS - Signal requires at least one!'}

${badPatterns.length > 0 ? `⚠️ TIER 3/4 PATTERNS (reduce confidence):
${badPatterns.map((s: any) => `- ${s.pattern_name}: ${s.win_rate_24h?.toFixed(1)}% win rate`).join('\n')}` : ''}

CRITICAL: Prioritize Tier 1 signals. Penalize confidence if only Tier 3/4 patterns present.`;
    }

    // Analyze past performance for learning context
    let learningContext = '';
    if (pastPredictions && pastPredictions.length > 0) {
      const wins = pastPredictions.filter(p => p.outcome === 'WIN').length;
      const losses = pastPredictions.filter(p => p.outcome === 'LOSS').length;
      const pending = pastPredictions.filter(p => !p.outcome || p.outcome === 'PENDING').length;
      const winRate = wins + losses > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : 'N/A';

      const failedTrades = pastPredictions.filter(p => p.outcome === 'LOSS');
      const failedAnalysis = failedTrades.slice(0, 5).map(t => 
        `${t.signal_type} at ${t.entry_price}, SL: ${t.stop_loss}, RSI was: ${t.technical_indicators?.rsi?.toFixed(1) || 'N/A'}`
      ).join('; ');

      const recentSignals = pastPredictions.slice(0, 5).map(p => p.signal_type);
      const buyCount = recentSignals.filter(s => s === 'BUY').length;
      const sellCount = recentSignals.filter(s => s === 'SELL').length;

      const lessonsLearned = learnings?.map(l => l.lesson_extracted).join('; ') || '';

      learningContext = `
HISTORICAL PERFORMANCE (Last 15 trades):
- Wins: ${wins}, Losses: ${losses}, Pending: ${pending}
- Win Rate: ${winRate}%
${winRate !== 'N/A' && parseFloat(winRate) < 50 ? '⚠️ WIN RATE IS LOW - Be more conservative with confidence levels' : ''}

${failedTrades.length > 0 ? `RECENT FAILED TRADES (AVOID similar setups):
${failedAnalysis}` : ''}

${lessonsLearned ? `LESSONS FROM PAST TRADES:
${lessonsLearned}` : ''}

${buyCount >= 4 ? '⚠️ CAUTION: Too many recent BUY signals. Market may be overbought. Consider SELL if conditions support it.' : ''}
${sellCount >= 4 ? '⚠️ CAUTION: Too many recent SELL signals. Market may be oversold. Consider BUY if conditions support it.' : ''}`;
    }

    // Build signal confirmation context
    const confirmationContext = `
SIGNAL CONFIRMATION ANALYSIS:
- Can generate BUY: ${signalConfirmation.canBuy ? 'YES' : 'NO'}
- Can generate SELL: ${signalConfirmation.canSell ? 'YES' : 'NO'}

BUY SIGNALS (${signalConfirmation.buyReasons.length}): ${signalConfirmation.buyReasons.join(', ') || 'None'}
SELL SIGNALS (${signalConfirmation.sellReasons.length}): ${signalConfirmation.sellReasons.join(', ') || 'None'}
CONFLICTS (${signalConfirmation.conflicts.length}): ${signalConfirmation.conflicts.join(', ') || 'None'}

SIGNAL GUIDANCE:
- Can generate BUY: ${signalConfirmation.canBuy ? 'YES - conditions favor BUY' : 'NO'}
- Can generate SELL: ${signalConfirmation.canSell ? 'YES - conditions favor SELL' : 'NO'}`;

    // Calculate ATR-based levels for reference
    const buyLevels = calculateATRBasedLevels(currentPrice, indicators.atr, 'BUY', timeframe);
    const sellLevels = calculateATRBasedLevels(currentPrice, indicators.atr, 'SELL', timeframe);

    // Call Lovable AI for prediction
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const analysisPrompt = `You are an EXPERT forex trading analyst specializing in EUR/USD. Generate a clear BUY or SELL signal based on the analysis.

TIMEFRAME: ${timeframe} (${timeframeSettings.description})
CURRENT PRICE: ${currentPrice}
SENTIMENT SCORE: ${sentimentScore} (range: -100 to 100)
${staleCheck.isStale ? `⚠️ WARNING: ${staleCheck.reason}` : ''}

TECHNICAL INDICATORS:
- RSI (14): ${indicators.rsi.toFixed(2)} ${indicators.rsi > 70 ? '(OVERBOUGHT - favor SELL)' : indicators.rsi < 30 ? '(OVERSOLD - favor BUY)' : '(NEUTRAL)'}
- MACD: ${indicators.macd.value.toFixed(5)} (Signal: ${indicators.macd.signal.toFixed(5)}, Histogram: ${indicators.macd.histogram.toFixed(5)})
- EMA 9: ${indicators.ema9.toFixed(5)}
- EMA 21: ${indicators.ema21.toFixed(5)}
- EMA 50: ${indicators.ema50.toFixed(5)}
- EMA 200: ${indicators.ema200.toFixed(5)}
- Bollinger: Upper ${indicators.bollingerBands.upper.toFixed(5)}, Middle ${indicators.bollingerBands.middle.toFixed(5)}, Lower ${indicators.bollingerBands.lower.toFixed(5)}
- Stochastic: %K ${indicators.stochastic.k.toFixed(2)}, %D ${indicators.stochastic.d.toFixed(2)}
- ATR (14): ${indicators.atr.toFixed(5)} (Use for stop loss calculation: SL = ${(indicators.atr * 2).toFixed(5)} from entry)
- Support: ${indicators.supportLevels.map(s => s.toFixed(5)).join(', ') || 'None'}
- Resistance: ${indicators.resistanceLevels.map(r => r.toFixed(5)).join(', ') || 'None'}

PATTERNS DETECTED:
${patterns.length > 0 ? patterns.map(p => `- ${p}`).join('\n') : '- No significant patterns'}

${patternStatsContext}

${confirmationContext}

${learningContext}

ATR-BASED LEVELS (MUST USE THESE):
If BUY: SL=${buyLevels.stopLoss.toFixed(5)}, TP1=${buyLevels.takeProfit1.toFixed(5)}, TP2=${buyLevels.takeProfit2.toFixed(5)}
If SELL: SL=${sellLevels.stopLoss.toFixed(5)}, TP1=${sellLevels.takeProfit1.toFixed(5)}, TP2=${sellLevels.takeProfit2.toFixed(5)}

TRADING GUIDELINES (TIER-BASED STRATEGY):
1. You MUST choose either BUY or SELL - no other options
2. REQUIRE at least one Tier 1 pattern (RSI extreme or BB touch) for high confidence
3. PREFER BUY when: RSI < 30 (Tier 1), price below lower BB (Tier 1), Stochastic < 20
4. PREFER SELL when: RSI > 70 (Tier 1), price above upper BB (Tier 1), Stochastic > 80
5. PENALIZE confidence if only Tier 3/4 patterns (MACD, engulfing, golden/death cross)
6. ALWAYS use the ATR-based SL/TP levels provided above
7. Confidence levels:
   - 75-100%: Multiple Tier 1 patterns confirming
   - 60-74%: At least one Tier 1 pattern with Tier 2 confirmation
   - Below 60%: Insufficient - signal rejected

RECENT PRICE ACTION:
- 10 candles ago: ${candles[candles.length - 10]?.close?.toFixed(5) || 'N/A'}
- 50 candles ago: ${candles[candles.length - 50]?.close?.toFixed(5) || 'N/A'}
- 100 candles ago: ${candles[candles.length - 100]?.close?.toFixed(5) || 'N/A'}`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { 
            role: "system", 
            content: `You are a skilled forex trading analyst. Generate clear, actionable BUY or SELL signals.

Your approach:
- Generate actionable BUY or SELL signals based on technical analysis
- You MUST choose either BUY or SELL - these are the only options
- Always use ATR-based stop loss levels provided - never set tighter stops
- Risk/Reward ratio should be at least 1:1.5
- Confidence of 55%+ is acceptable, but 70%+ is preferred

Explain your reasoning clearly and commit to a direction.`
          },
          { role: "user", content: analysisPrompt }
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "generate_trading_signal",
              description: "Generate a structured trading signal based on technical analysis",
              parameters: {
                type: "object",
                properties: {
                  signal_type: { type: "string", enum: ["BUY", "SELL"], description: "The trading signal - must be BUY or SELL" },
                  confidence: { type: "number", description: "Confidence level 0-100. Minimum 60% required. 70%+ for strong signals." },
                  entry_price: { type: "number", description: "Recommended entry price (current price)" },
                  take_profit_1: { type: "number", description: "First take profit target - use ATR-based level" },
                  take_profit_2: { type: "number", description: "Second take profit target - use ATR-based level" },
                  stop_loss: { type: "number", description: "Stop loss price - MUST use ATR-based level (2+ ATR from entry)" },
                  trend_direction: { type: "string", enum: ["BULLISH", "BEARISH", "NEUTRAL"], description: "Overall trend direction" },
                  trend_strength: { type: "number", description: "Trend strength 0-100" },
                  sentiment_score: { type: "number", description: "Market sentiment -100 to 100" },
                  reasoning: { type: "string", description: "Detailed reasoning including which indicators confirm the signal and any concerns" }
                },
                required: ["signal_type", "confidence", "entry_price", "stop_loss", "take_profit_1", "trend_direction", "trend_strength", "reasoning"]
              }
            }
          }
        ],
        tool_choice: { type: "function", function: { name: "generate_trading_signal" } }
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI Gateway error:", response.status, errorText);
      
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ success: false, error: "Rate limit exceeded. Please try again later." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ success: false, error: "AI credits exhausted. Please add more credits." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      throw new Error("AI analysis failed");
    }

    const aiResponse = await response.json();
    console.log("AI Response:", JSON.stringify(aiResponse));

    const toolCall = aiResponse.choices?.[0]?.message?.tool_calls?.[0];
    if (!toolCall || toolCall.function.name !== "generate_trading_signal") {
      throw new Error("Invalid AI response format");
    }

    let signal = JSON.parse(toolCall.function.arguments);
    console.log("Parsed signal:", signal);

    // Post-processing: enforce minimum confidence threshold (60% - raised from 55%)
    if (signal.confidence < 60) {
      console.log(`Confidence ${signal.confidence}% too low, returning no signal`);
      return new Response(
        JSON.stringify({
          success: false,
          noSignal: true,
          message: "Confidence too low for actionable signal",
          reason: `AI confidence was only ${signal.confidence.toFixed(0)}% (minimum 60% required for tier-based strategy)`,
          originalSignal: signal.signal_type
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Enforce signal confirmation rules
    if (signal.signal_type === 'BUY' && !signalConfirmation.canBuy) {
      console.log("BUY signal blocked by confirmation filters");
      return new Response(
        JSON.stringify({
          success: false,
          noSignal: true,
          message: "BUY signal blocked - insufficient confirmation",
          reason: `Conflicts: ${signalConfirmation.conflicts.join(', ')}`,
          buyReasons: signalConfirmation.buyReasons,
          sellReasons: signalConfirmation.sellReasons
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
    if (signal.signal_type === 'SELL' && !signalConfirmation.canSell) {
      console.log("SELL signal blocked by confirmation filters");
      return new Response(
        JSON.stringify({
          success: false,
          noSignal: true,
          message: "SELL signal blocked - insufficient confirmation",
          reason: `Conflicts: ${signalConfirmation.conflicts.join(', ')}`,
          buyReasons: signalConfirmation.buyReasons,
          sellReasons: signalConfirmation.sellReasons
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Ensure ATR-based levels are used
    if (signal.signal_type === 'BUY') {
      signal.stop_loss = buyLevels.stopLoss;
      signal.take_profit_1 = buyLevels.takeProfit1;
      signal.take_profit_2 = buyLevels.takeProfit2;
    } else if (signal.signal_type === 'SELL') {
      signal.stop_loss = sellLevels.stopLoss;
      signal.take_profit_1 = sellLevels.takeProfit1;
      signal.take_profit_2 = sellLevels.takeProfit2;
    }

    // Calculate expiry based on timeframe
    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + timeframeSettings.expiryHours);

    const predictionData = {
      signal_type: signal.signal_type,
      confidence: signal.confidence,
      entry_price: signal.entry_price || currentPrice,
      take_profit_1: signal.take_profit_1,
      take_profit_2: signal.take_profit_2,
      stop_loss: signal.stop_loss,
      current_price_at_prediction: currentPrice,
      trend_direction: signal.trend_direction,
      trend_strength: signal.trend_strength,
      reasoning: signal.reasoning,
      technical_indicators: indicators,
      patterns_detected: patterns,
      sentiment_score: signal.sentiment_score || sentimentScore,
      expires_at: expiresAt.toISOString()
    };

    const { data: prediction, error: insertError } = await supabase
      .from("predictions")
      .insert(predictionData)
      .select()
      .single();

    if (insertError) {
      console.error("Error storing prediction:", insertError);
      throw new Error("Failed to store prediction");
    }

    console.log("Prediction stored successfully:", prediction.id);

    return new Response(
      JSON.stringify({
        success: true,
        prediction: {
          ...prediction,
          timeframe,
          timeframeSettings
        },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error generating prediction:", error);
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error instanceof Error ? error.message : "Unknown error" 
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});