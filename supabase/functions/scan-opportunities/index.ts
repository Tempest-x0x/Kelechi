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

// Check if forex market is open
function isForexMarketOpen(): { isOpen: boolean; reason: string } {
  const now = new Date();
  const utcDay = now.getUTCDay();
  const utcHour = now.getUTCHours();
  
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

// Calculate MACD
function calculateMACD(closes: number[]): { value: number; signal: number; histogram: number } {
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);
  const macdLine = ema12 - ema26;
  
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

// Calculate Stochastic
function calculateStochastic(highs: number[], lows: number[], closes: number[], period = 14): { k: number; d: number } {
  if (closes.length < period) return { k: 50, d: 50 };
  
  const highSlice = highs.slice(-period);
  const lowSlice = lows.slice(-period);
  const currentClose = closes[closes.length - 1];
  
  const highestHigh = Math.max(...highSlice);
  const lowestLow = Math.min(...lowSlice);
  
  if (highestHigh === lowestLow) return { k: 50, d: 50 };
  
  const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
  
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

// Calculate ATR
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

// Calculate Support/Resistance
function calculateSupportResistance(highs: number[], lows: number[]): { support: number[]; resistance: number[] } {
  const support: number[] = [];
  const resistance: number[] = [];
  
  const lookback = Math.min(100, highs.length);
  const recentHighs = highs.slice(-lookback);
  const recentLows = lows.slice(-lookback);
  
  for (let i = 2; i < lookback - 2; i++) {
    if (recentHighs[i] > recentHighs[i-1] && recentHighs[i] > recentHighs[i-2] &&
        recentHighs[i] > recentHighs[i+1] && recentHighs[i] > recentHighs[i+2]) {
      resistance.push(recentHighs[i]);
    }
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

// Detect patterns
function detectPatterns(candles: Candle[]): string[] {
  const patterns: string[] = [];
  if (candles.length < 50) return patterns;
  
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const opens = candles.map(c => c.open);
  
  const last = candles.length - 1;
  
  // Doji
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
    
    if (closes[last-1] < opens[last-1] && closes[last] > opens[last] &&
        opens[last] <= closes[last-1] && closes[last] >= opens[last-1] && currBody > prevBody) {
      patterns.push('Bullish Engulfing');
    }
    
    if (closes[last-1] > opens[last-1] && closes[last] < opens[last] &&
        opens[last] >= closes[last-1] && closes[last] <= opens[last-1] && currBody > prevBody) {
      patterns.push('Bearish Engulfing');
    }
  }
  
  // Trend via EMAs
  const ema20 = calculateEMA(closes, 20);
  const ema50 = calculateEMA(closes, 50);
  const currentPrice = closes[last];
  
  if (currentPrice > ema20 && ema20 > ema50) {
    patterns.push('Strong Uptrend');
  } else if (currentPrice < ema20 && ema20 < ema50) {
    patterns.push('Strong Downtrend');
  }
  
  return patterns;
}

// Calculate all indicators
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
// Tier 1: Proven winners (>52% win rate) - Required for signals
// Tier 2: Neutral (50-52%) - Standard weight  
// Tier 3: Weak (<50%) - Reduced weight
// Tier 4: Actively harmful (<47%) - Penalize
const PATTERN_WEIGHTS = {
  // Tier 1: Proven winners
  rsi_oversold: { weight: 1.5, tier: 1, expectedWinRate: 52.4 },
  rsi_overbought: { weight: 1.5, tier: 1, expectedWinRate: 52.16 },
  bb_lower_touch: { weight: 1.3, tier: 1, expectedWinRate: 52.06 },
  bb_upper_touch: { weight: 1.3, tier: 1, expectedWinRate: 51.67 },
  
  // Tier 2: Neutral
  stochastic_oversold: { weight: 1.0, tier: 2, expectedWinRate: 50 },
  stochastic_overbought: { weight: 1.0, tier: 2, expectedWinRate: 50 },
  
  // Tier 3: Weak edge
  macd_bullish_cross: { weight: 0.5, tier: 3, expectedWinRate: 47.85 },
  macd_bearish_cross: { weight: 0.5, tier: 3, expectedWinRate: 47.39 },
  bullish_engulfing: { weight: 0.5, tier: 3, expectedWinRate: 48.39 },
  bearish_engulfing: { weight: 0.5, tier: 3, expectedWinRate: 48.24 },
  
  // Tier 4: Actively harmful - penalize
  golden_cross: { weight: -0.5, tier: 4, expectedWinRate: 45.89 },
  death_cross: { weight: -0.5, tier: 4, expectedWinRate: 45.87 },
} as const;

type PatternName = keyof typeof PATTERN_WEIGHTS;

interface PatternDetection {
  name: PatternName;
  type: 'BUY' | 'SELL';
  tier: number;
  weight: number;
  reason: string;
}

// Analyze opportunity with pattern stats and tier-based weighting
function analyzeOpportunity(
  indicators: TechnicalIndicators,
  patterns: string[],
  currentPrice: number,
  patternStats: any[]
): { signal: 'BUY' | 'SELL' | null; confidence: number; reasons: string[]; patternData: any[] } {
  const buyPatterns: PatternDetection[] = [];
  const sellPatterns: PatternDetection[] = [];
  const matchedPatternStats: any[] = [];
  
  // Detect patterns and classify by tier
  
  // RSI - Tier 1
  if (indicators.rsi < 30) {
    buyPatterns.push({ 
      name: 'rsi_oversold', 
      type: 'BUY', 
      tier: 1, 
      weight: PATTERN_WEIGHTS.rsi_oversold.weight,
      reason: 'RSI oversold (<30) - Tier 1 signal'
    });
    const stat = patternStats.find(p => p.pattern_name === 'rsi_oversold');
    if (stat) matchedPatternStats.push(stat);
  } else if (indicators.rsi < 40) {
    buyPatterns.push({ 
      name: 'rsi_oversold', 
      type: 'BUY', 
      tier: 2, 
      weight: 0.5,
      reason: 'RSI approaching oversold (30-40)'
    });
  }
  
  if (indicators.rsi > 70) {
    sellPatterns.push({ 
      name: 'rsi_overbought', 
      type: 'SELL', 
      tier: 1, 
      weight: PATTERN_WEIGHTS.rsi_overbought.weight,
      reason: 'RSI overbought (>70) - Tier 1 signal'
    });
    const stat = patternStats.find(p => p.pattern_name === 'rsi_overbought');
    if (stat) matchedPatternStats.push(stat);
  } else if (indicators.rsi > 60) {
    sellPatterns.push({ 
      name: 'rsi_overbought', 
      type: 'SELL', 
      tier: 2, 
      weight: 0.5,
      reason: 'RSI approaching overbought (60-70)'
    });
  }
  
  // Bollinger Bands - Tier 1
  if (currentPrice < indicators.bollingerBands.lower) {
    buyPatterns.push({ 
      name: 'bb_lower_touch', 
      type: 'BUY', 
      tier: 1, 
      weight: PATTERN_WEIGHTS.bb_lower_touch.weight,
      reason: 'Price below lower Bollinger Band - Tier 1 signal'
    });
    const stat = patternStats.find(p => p.pattern_name === 'bb_lower_touch');
    if (stat) matchedPatternStats.push(stat);
  }
  
  if (currentPrice > indicators.bollingerBands.upper) {
    sellPatterns.push({ 
      name: 'bb_upper_touch', 
      type: 'SELL', 
      tier: 1, 
      weight: PATTERN_WEIGHTS.bb_upper_touch.weight,
      reason: 'Price above upper Bollinger Band - Tier 1 signal'
    });
    const stat = patternStats.find(p => p.pattern_name === 'bb_upper_touch');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // Stochastic - Tier 2
  if (indicators.stochastic.k < 20 && indicators.stochastic.d < 20) {
    buyPatterns.push({ 
      name: 'stochastic_oversold', 
      type: 'BUY', 
      tier: 2, 
      weight: PATTERN_WEIGHTS.stochastic_oversold.weight,
      reason: 'Stochastic deeply oversold (<20)'
    });
  }
  
  if (indicators.stochastic.k > 80 && indicators.stochastic.d > 80) {
    sellPatterns.push({ 
      name: 'stochastic_overbought', 
      type: 'SELL', 
      tier: 2, 
      weight: PATTERN_WEIGHTS.stochastic_overbought.weight,
      reason: 'Stochastic deeply overbought (>80)'
    });
  }
  
  // MACD - Tier 3 (weak signals, low weight)
  if (indicators.macd.histogram > 0 && indicators.macd.value > indicators.macd.signal) {
    buyPatterns.push({ 
      name: 'macd_bullish_cross', 
      type: 'BUY', 
      tier: 3, 
      weight: PATTERN_WEIGHTS.macd_bullish_cross.weight,
      reason: 'MACD bullish crossover - Tier 3 (weak)'
    });
    const stat = patternStats.find(p => p.pattern_name === 'macd_bullish_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  
  if (indicators.macd.histogram < 0 && indicators.macd.value < indicators.macd.signal) {
    sellPatterns.push({ 
      name: 'macd_bearish_cross', 
      type: 'SELL', 
      tier: 3, 
      weight: PATTERN_WEIGHTS.macd_bearish_cross.weight,
      reason: 'MACD bearish crossover - Tier 3 (weak)'
    });
    const stat = patternStats.find(p => p.pattern_name === 'macd_bearish_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // EMA alignment - Tier 4 (actively harmful based on historical data)
  const priceAboveEma21 = currentPrice > indicators.ema21;
  const priceAboveEma50 = currentPrice > indicators.ema50;
  const ema21AboveEma50 = indicators.ema21 > indicators.ema50;
  
  if (priceAboveEma21 && priceAboveEma50 && ema21AboveEma50) {
    // This is golden cross - historically harmful, penalize
    buyPatterns.push({ 
      name: 'golden_cross', 
      type: 'BUY', 
      tier: 4, 
      weight: PATTERN_WEIGHTS.golden_cross.weight,
      reason: 'Golden Cross alignment - Tier 4 (historically <46% win rate)'
    });
    const stat = patternStats.find(p => p.pattern_name === 'golden_cross');
    if (stat) matchedPatternStats.push(stat);
  } else if (!priceAboveEma21 && !priceAboveEma50 && !ema21AboveEma50) {
    // Death cross - historically harmful, penalize
    sellPatterns.push({ 
      name: 'death_cross', 
      type: 'SELL', 
      tier: 4, 
      weight: PATTERN_WEIGHTS.death_cross.weight,
      reason: 'Death Cross alignment - Tier 4 (historically <46% win rate)'
    });
    const stat = patternStats.find(p => p.pattern_name === 'death_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // Candlestick patterns - Tier 3
  patterns.forEach(p => {
    if (p.includes('Bullish') && p.includes('Engulfing')) {
      buyPatterns.push({ 
        name: 'bullish_engulfing', 
        type: 'BUY', 
        tier: 3, 
        weight: PATTERN_WEIGHTS.bullish_engulfing.weight,
        reason: 'Bullish Engulfing - Tier 3 (48% win rate)'
      });
      const stat = patternStats.find(s => s.pattern_name === 'bullish_engulfing');
      if (stat && !matchedPatternStats.includes(stat)) matchedPatternStats.push(stat);
    }
    if (p.includes('Bearish') && p.includes('Engulfing')) {
      sellPatterns.push({ 
        name: 'bearish_engulfing', 
        type: 'SELL', 
        tier: 3, 
        weight: PATTERN_WEIGHTS.bearish_engulfing.weight,
        reason: 'Bearish Engulfing - Tier 3 (48% win rate)'
      });
      const stat = patternStats.find(s => s.pattern_name === 'bearish_engulfing');
      if (stat && !matchedPatternStats.includes(stat)) matchedPatternStats.push(stat);
    }
  });
  
  // Calculate weighted scores
  const buyScore = buyPatterns.reduce((sum, p) => sum + p.weight, 0);
  const sellScore = sellPatterns.reduce((sum, p) => sum + p.weight, 0);
  
  // Count patterns by tier
  const buyTier1Count = buyPatterns.filter(p => p.tier === 1).length;
  const sellTier1Count = sellPatterns.filter(p => p.tier === 1).length;
  const buyTier4Count = buyPatterns.filter(p => p.tier === 4).length;
  const sellTier4Count = sellPatterns.filter(p => p.tier === 4).length;
  
  console.log(`Pattern analysis: BUY score=${buyScore.toFixed(2)} (${buyTier1Count} T1, ${buyTier4Count} T4), SELL score=${sellScore.toFixed(2)} (${sellTier1Count} T1, ${sellTier4Count} T4)`);
  
  // Decision logic:
  // 1. REQUIRE at least one Tier 1 pattern
  // 2. Weighted score must be positive and exceed threshold (1.5)
  // 3. Must have net advantage over opposing side
  
  const SCORE_THRESHOLD = 1.5;
  let signal: 'BUY' | 'SELL' | null = null;
  let confidence = 50;
  let reasons: string[] = [];
  
  const netBuyAdvantage = buyScore - sellScore;
  const netSellAdvantage = sellScore - buyScore;
  
  if (buyTier1Count >= 1 && buyScore >= SCORE_THRESHOLD && netBuyAdvantage >= 0.5) {
    signal = 'BUY';
    reasons = buyPatterns.map(p => p.reason);
    
    // Confidence calculation based on tiers
    confidence = 50 
      + (buyTier1Count * 12)  // +12 per Tier 1 pattern
      + (buyPatterns.filter(p => p.tier === 2).length * 5)   // +5 per Tier 2
      + (buyPatterns.filter(p => p.tier === 3).length * 2)   // +2 per Tier 3
      - (buyTier4Count * 10); // -10 per Tier 4 (penalty)
    
    // Combination bonuses
    const hasRsiOversold = buyPatterns.some(p => p.name === 'rsi_oversold' && p.tier === 1);
    const hasBbLower = buyPatterns.some(p => p.name === 'bb_lower_touch');
    const hasStochOversold = buyPatterns.some(p => p.name === 'stochastic_oversold');
    
    if (hasRsiOversold && hasBbLower) {
      confidence += 15;
      reasons.push('✨ RSI + BB combo bonus (+15%)');
    }
    if (hasRsiOversold && hasStochOversold) {
      confidence += 10;
      reasons.push('✨ RSI + Stochastic combo bonus (+10%)');
    }
    
  } else if (sellTier1Count >= 1 && sellScore >= SCORE_THRESHOLD && netSellAdvantage >= 0.5) {
    signal = 'SELL';
    reasons = sellPatterns.map(p => p.reason);
    
    confidence = 50 
      + (sellTier1Count * 12)
      + (sellPatterns.filter(p => p.tier === 2).length * 5)
      + (sellPatterns.filter(p => p.tier === 3).length * 2)
      - (sellTier4Count * 10);
    
    // Combination bonuses
    const hasRsiOverbought = sellPatterns.some(p => p.name === 'rsi_overbought' && p.tier === 1);
    const hasBbUpper = sellPatterns.some(p => p.name === 'bb_upper_touch');
    const hasStochOverbought = sellPatterns.some(p => p.name === 'stochastic_overbought');
    
    if (hasRsiOverbought && hasBbUpper) {
      confidence += 15;
      reasons.push('✨ RSI + BB combo bonus (+15%)');
    }
    if (hasRsiOverbought && hasStochOverbought) {
      confidence += 10;
      reasons.push('✨ RSI + Stochastic combo bonus (+10%)');
    }
  }
  
  // If no Tier 1 pattern, log why we're rejecting
  if (!signal) {
    console.log(`No signal: BUY T1=${buyTier1Count}, SELL T1=${sellTier1Count}, threshold=${SCORE_THRESHOLD}`);
  }
  
  // Clamp confidence
  confidence = Math.min(95, Math.max(0, confidence));
  
  return { signal, confidence, reasons, patternData: matchedPatternStats };
}

// Calculate ATR-based levels with ENFORCED 1:3 R:R
// 1:3 R:R ensures profitability at just 25% win rate - prop firm compatible
function calculateLevels(
  currentPrice: number, 
  atr: number, 
  signalType: 'BUY' | 'SELL',
  confidence: number
): { stopLoss: number; takeProfit1: number; takeProfit2: number; riskRewardRatio: string } {
  // ENFORCED 1:3 MINIMUM R:R - Suitable for prop firm challenges
  // SL = 1.0x ATR, TP1 = 3.0x ATR (1:3 R:R)
  // TP2 = 4.5x ATR (1:4.5 R:R for runners)
  
  // Base multipliers for 1:3 R:R
  let slMult = 1.0;
  let tp1Mult = 3.0;
  let tp2Mult = 4.5;
  
  // Higher confidence = slightly tighter stop for even better R:R
  if (confidence >= 80) {
    slMult = 0.8;  // Tighter stop = 1:3.75 R:R
    tp1Mult = 3.0;
    tp2Mult = 4.5;
  } else if (confidence >= 70) {
    slMult = 1.0;  // Standard 1:3 R:R
    tp1Mult = 3.0;
    tp2Mult = 4.5;
  } else {
    // Lower confidence: slightly wider stop but maintain 1:3 minimum
    slMult = 1.2;
    tp1Mult = 3.6;  // 1:3 R:R maintained
    tp2Mult = 5.0;
  }
  
  const riskRewardRatio = `1:${(tp1Mult / slMult).toFixed(1)}`;
  
  console.log(`Calculating levels: confidence=${confidence}, R:R=${riskRewardRatio}, SL=${slMult}x ATR, TP1=${tp1Mult}x ATR`);
  
  if (signalType === 'BUY') {
    return {
      stopLoss: currentPrice - (atr * slMult),
      takeProfit1: currentPrice + (atr * tp1Mult),
      takeProfit2: currentPrice + (atr * tp2Mult),
      riskRewardRatio
    };
  } else {
    return {
      stopLoss: currentPrice + (atr * slMult),
      takeProfit1: currentPrice - (atr * tp1Mult),
      takeProfit2: currentPrice - (atr * tp2Mult),
      riskRewardRatio
    };
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    console.log("Starting opportunity scan...");
    
    // Check market status
    const marketStatus = isForexMarketOpen();
    if (!marketStatus.isOpen) {
      console.log("Market closed:", marketStatus.reason);
      return new Response(
        JSON.stringify({ success: true, message: marketStatus.reason, scanned: false }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Initialize Supabase
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Expire old opportunities
    await supabase
      .from('trading_opportunities')
      .update({ status: 'EXPIRED' })
      .eq('status', 'ACTIVE')
      .lt('expires_at', new Date().toISOString());

    // Fetch price data from cache
    const { data: priceData, error: priceError } = await supabase
      .from('price_history')
      .select('*')
      .eq('symbol', 'EUR/USD')
      .eq('timeframe', '1h')
      .order('timestamp', { ascending: true })
      .limit(200);

    if (priceError || !priceData || priceData.length < 50) {
      console.log("Not enough price data:", priceError?.message || `Only ${priceData?.length || 0} candles`);
      return new Response(
        JSON.stringify({ success: true, message: "Not enough price data for analysis", scanned: false }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`Analyzing ${priceData.length} candles...`);

    // Transform to candle format
    const candles: Candle[] = priceData.map(p => ({
      timestamp: p.timestamp,
      open: Number(p.open),
      high: Number(p.high),
      low: Number(p.low),
      close: Number(p.close),
      volume: p.volume ? Number(p.volume) : undefined
    }));

    const currentPrice = candles[candles.length - 1].close;
    
    // Calculate indicators and detect patterns
    const indicators = calculateIndicators(candles);
    const patterns = detectPatterns(candles);
    
    console.log("Indicators:", JSON.stringify({
      rsi: indicators.rsi.toFixed(2),
      macd: indicators.macd.histogram.toFixed(5),
      stochastic: indicators.stochastic.k.toFixed(2)
    }));
    console.log("Patterns:", patterns);

    // Fetch pattern statistics
    const { data: patternStats } = await supabase
      .from('pattern_statistics')
      .select('*');

    // Analyze for opportunity
    const analysis = analyzeOpportunity(indicators, patterns, currentPrice, patternStats || []);
    
    console.log("Analysis result:", {
      signal: analysis.signal,
      confidence: analysis.confidence,
      reasons: analysis.reasons.length
    });

    // Only create opportunity if conditions are met (tier-based strategy)
    // - Signal is BUY or SELL (requires Tier 1 pattern)
    // - Confidence >= 60% (now enforced by tier system)
    // - At least 2 confirming reasons
    if (!analysis.signal || analysis.confidence < 60 || analysis.reasons.length < 2) {
      console.log("No high-probability opportunity detected");
      return new Response(
        JSON.stringify({ 
          success: true, 
          message: "No high-probability opportunity detected", 
          scanned: true,
          analysis: {
            bestSignal: analysis.signal,
            confidence: analysis.confidence,
            reasons: analysis.reasons.length
          }
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Check for conflicting active signals (opposite direction)
    const oppositeSignal = analysis.signal === 'BUY' ? 'SELL' : 'BUY';
    const { data: conflictingOpps } = await supabase
      .from('trading_opportunities')
      .select('id, signal_type, confidence, created_at, entry_price')
      .eq('status', 'ACTIVE')
      .eq('signal_type', oppositeSignal);

    let isSignalReversal = false;
    let previousSignal: { signal_type: string; confidence: number; created_at: string } | null = null;

    if (conflictingOpps && conflictingOpps.length > 0) {
      const mostRecentConflict = conflictingOpps[0];
      const conflictAge = Date.now() - new Date(mostRecentConflict.created_at).getTime();
      const oneHourMs = 60 * 60 * 1000;

      // Cooldown: require 1 hour OR 10%+ higher confidence to reverse
      if (conflictAge < oneHourMs && analysis.confidence < mostRecentConflict.confidence + 10) {
        console.log(`Cooldown active: ${oppositeSignal} signal from ${mostRecentConflict.created_at} is less than 1 hour old`);
        return new Response(
          JSON.stringify({ 
            success: true, 
            message: `Cooldown active: Recent ${oppositeSignal} signal (${mostRecentConflict.confidence.toFixed(0)}%) is less than 1 hour old. Need 10%+ higher confidence to reverse.`, 
            scanned: true,
            cooldownRemaining: Math.ceil((oneHourMs - conflictAge) / 60000) + ' minutes'
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Expire conflicting signals
      console.log(`Expiring ${conflictingOpps.length} conflicting ${oppositeSignal} signal(s) due to reversal`);
      await supabase
        .from('trading_opportunities')
        .update({ status: 'EXPIRED', outcome: 'EXPIRED' })
        .eq('status', 'ACTIVE')
        .eq('signal_type', oppositeSignal);

      isSignalReversal = true;
      previousSignal = {
        signal_type: mostRecentConflict.signal_type,
        confidence: mostRecentConflict.confidence,
        created_at: mostRecentConflict.created_at
      };
    }

    // Enhanced duplicate check - look at recent opportunities (4 hours) regardless of status
    // and require significant price movement before creating new opportunity
    const { data: recentOpps } = await supabase
      .from('trading_opportunities')
      .select('id, signal_type, entry_price, created_at, status')
      .eq('signal_type', analysis.signal)
      .gte('created_at', new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()); // Last 4 hours

    if (recentOpps && recentOpps.length > 0) {
      // Check if price has moved significantly (at least 15 pips)
      const mostRecent = recentOpps[0];
      const priceDiff = Math.abs(currentPrice - mostRecent.entry_price);
      const pipsDiff = priceDiff * 10000; // Convert to pips for EUR/USD
      
      if (pipsDiff < 15) {
        console.log(`Similar opportunity exists (${pipsDiff.toFixed(1)} pips difference, need 15+)`);
        return new Response(
          JSON.stringify({ 
            success: true, 
            message: `Similar ${analysis.signal} opportunity exists from ${mostRecent.created_at} (only ${pipsDiff.toFixed(1)} pips apart)`, 
            scanned: true,
            existingOpportunity: mostRecent.id,
            priceDifference: pipsDiff.toFixed(1)
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      console.log(`Price moved ${pipsDiff.toFixed(1)} pips since last ${analysis.signal} signal - creating new opportunity`);
    }

    // Calculate entry levels with dynamic R:R based on confidence
    const levels = calculateLevels(currentPrice, indicators.atr, analysis.signal, analysis.confidence);

    // Build reasoning
    const reasoning = `${analysis.signal} opportunity detected with ${analysis.confidence.toFixed(0)}% confidence.\n\n` +
      `Confirming factors:\n${analysis.reasons.map(r => `• ${r}`).join('\n')}\n\n` +
      `Technical snapshot:\n` +
      `• RSI: ${indicators.rsi.toFixed(1)}\n` +
      `• MACD Histogram: ${indicators.macd.histogram > 0 ? '+' : ''}${indicators.macd.histogram.toFixed(5)}\n` +
      `• Stochastic %K: ${indicators.stochastic.k.toFixed(1)}\n` +
      `• ATR: ${indicators.atr.toFixed(5)}`;

    // Insert opportunity
    const expiresAt = new Date(Date.now() + 4 * 60 * 60 * 1000); // 4 hours

    const { data: newOpp, error: insertError } = await supabase
      .from('trading_opportunities')
      .insert({
        signal_type: analysis.signal,
        confidence: analysis.confidence,
        entry_price: currentPrice,
        current_price: currentPrice,
        stop_loss: levels.stopLoss,
        take_profit_1: levels.takeProfit1,
        take_profit_2: levels.takeProfit2,
        patterns_detected: patterns,
        technical_indicators: indicators,
        pattern_stats: analysis.patternData,
        reasoning,
        status: 'ACTIVE',
        expires_at: expiresAt.toISOString()
      })
      .select()
      .single();

    if (insertError) {
      console.error("Failed to insert opportunity:", insertError);
      throw new Error("Failed to save opportunity");
    }

    console.log("Created new opportunity:", newOpp.id);

    // Send Telegram notification for new opportunity
    try {
      const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
      const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
      
      await fetch(`${supabaseUrl}/functions/v1/send-telegram-notification`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${supabaseAnonKey}`,
        },
        body: JSON.stringify({
          signal_type: newOpp.signal_type,
          confidence: newOpp.confidence,
          entry_price: newOpp.entry_price,
          stop_loss: newOpp.stop_loss,
          take_profit_1: newOpp.take_profit_1,
          take_profit_2: newOpp.take_profit_2,
          reasoning: newOpp.reasoning,
          is_reversal: isSignalReversal,
          previous_signal: previousSignal,
        }),
      });
      console.log("Telegram notification sent for opportunity:", newOpp.id, isSignalReversal ? "(REVERSAL)" : "");
    } catch (notifyError) {
      console.error("Failed to send Telegram notification:", notifyError);
      // Don't fail the whole request if notification fails
    }

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: "New opportunity detected!", 
        scanned: true,
        opportunity: newOpp
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Scan error:", error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
