/**
 * ICT/SMC Institutional Order Flow Scanner
 * 
 * Signal Flow:
 * 1. Killzone Check (London 07:00-10:00 UTC, NY 12:00-15:00 UTC)
 * 2. HTF Bias (1H > 200 EMA + 15m Bullish/Bearish Structure)
 * 3. Liquidity Sweep (pierce + value area touch + close inside)
 * 4. Market Structure Shift (displacement candle)
 * 5. Fair Value Gap (50% equilibrium entry)
 * 
 * Target Metrics: 70% Win Rate | 1:2.2 RRR | 1-2 Trades per Pair/Day
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// ============================================
// ICT CONFIGURATION (from ict-config.ts)
// ============================================
const ICT_CONFIG = {
  RRR_RATIO: 2.2,
  TARGET_WIN_RATE: 70,
  MAX_TRADES_PER_PAIR_PER_DAY: 2,
  MIN_CONFIDENCE_THRESHOLD: 65,
  POSITION_EXPIRY_HOURS: 4,
};

const KILLZONES = {
  LONDON: { start: { hour: 7, minute: 0 }, end: { hour: 10, minute: 0 } },
  NEW_YORK: { start: { hour: 12, minute: 0 }, end: { hour: 15, minute: 0 } },
};

const HTF_CONFIG = {
  EMA_200_PERIOD: 200,
  SWING_POINT_LOOKBACK: 5,
  STRUCTURE_LOOKBACK_CANDLES: 20,
};

const LIQUIDITY_CONFIG = {
  SWEEP_LOOKBACK_CANDLES: 20,
  CLOSE_TOLERANCE_ATR: 0.1,
};

const FVG_CONFIG = {
  MIN_GAP_PIPS: 1.5,
  ENTRY_EQUILIBRIUM_LEVEL: 0.5,
  MAX_FVG_AGE_CANDLES: 10,
  SL_BUFFER_PIPS: 2,
};

const MSS_CONFIG = {
  MIN_DISPLACEMENT_BODY_ATR: 0.8,
};

const DEFAULT_PIP_VALUES: Record<string, number> = {
  "EUR/USD": 0.0001, "GBP/USD": 0.0001, "USD/JPY": 0.01, "USD/CHF": 0.0001,
  "AUD/USD": 0.0001, "USD/CAD": 0.0001, "EUR/JPY": 0.01, "GBP/JPY": 0.01,
  "AUD/JPY": 0.01, "XAU/USD": 0.01, "EUR/CHF": 0.0001, "EUR/GBP": 0.0001,
};

// ============================================
// TYPES
// ============================================
interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface SwingPoint {
  index: number;
  price: number;
  type: 'HIGH' | 'LOW';
  timestamp: string;
}

interface MarketStructure {
  trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  isHigherHighs: boolean;
  isHigherLows: boolean;
  isLowerHighs: boolean;
  isLowerLows: boolean;
  lastSwingHigh: SwingPoint | null;
  lastSwingLow: SwingPoint | null;
}

interface HTFBias {
  direction: 'BUY' | 'SELL' | null;
  hourlyTrend: 'ABOVE_200EMA' | 'BELOW_200EMA';
  fifteenMinStructure: MarketStructure;
  ema200: number;
  reasons: string[];
}

interface LiquiditySweep {
  detected: boolean;
  type: 'BUY' | 'SELL';
  sweepPrice: number;
  sweepLow: number;
  sweepHigh: number;
  candleIndex: number;
  liquidityLevel: number;
}

interface FairValueGap {
  detected: boolean;
  type: 'BULLISH' | 'BEARISH';
  equilibrium: number;
  gapSizePips: number;
  isValid: boolean;
}

interface MarketStructureShift {
  detected: boolean;
  direction: 'BULLISH' | 'BEARISH';
  displacementIndex: number;
  swingBroken: SwingPoint | null;
}

interface ICTSignal {
  valid: boolean;
  direction: 'BUY' | 'SELL' | null;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  reasons: string[];
  invalidReasons: string[];
}

// ============================================
// HELPER FUNCTIONS
// ============================================
let dynamicPipValues: Record<string, number> = {};

function getPipValue(symbol: string): number {
  return dynamicPipValues[symbol] || DEFAULT_PIP_VALUES[symbol] || 0.0001;
}

function priceToPips(priceDiff: number, symbol: string): number {
  return Math.abs(priceDiff) / getPipValue(symbol);
}

function pipToPrice(pips: number, symbol: string): number {
  return pips * getPipValue(symbol);
}

function isInKillzone(date: Date = new Date()): { inKillzone: boolean; killzone: string | null } {
  const utcHour = date.getUTCHours();
  const utcMinute = date.getUTCMinutes();
  const totalMinutes = utcHour * 60 + utcMinute;

  for (const [name, zone] of Object.entries(KILLZONES)) {
    const startMinutes = zone.start.hour * 60 + zone.start.minute;
    const endMinutes = zone.end.hour * 60 + zone.end.minute;
    
    if (totalMinutes >= startMinutes && totalMinutes < endMinutes) {
      return { inKillzone: true, killzone: name };
    }
  }

  return { inKillzone: false, killzone: null };
}

function isForexMarketOpen(): { isOpen: boolean; reason: string } {
  const now = new Date();
  const utcDay = now.getUTCDay();
  const utcHour = now.getUTCHours();
  
  if (utcDay === 6) return { isOpen: false, reason: "Forex market is closed on Saturdays" };
  if (utcDay === 0 && utcHour < 21) return { isOpen: false, reason: "Forex market opens Sunday 21:00 UTC" };
  if (utcDay === 5 && utcHour >= 21) return { isOpen: false, reason: "Forex market closed Friday 21:00 UTC" };
  
  return { isOpen: true, reason: "Market is open" };
}

// ============================================
// TECHNICAL INDICATORS
// ============================================
function calculateEMA(closes: number[], period: number): number {
  if (closes.length < period) return closes[closes.length - 1] || 0;
  
  const multiplier = 2 / (period + 1);
  let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  for (let i = period; i < closes.length; i++) {
    ema = (closes[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

function calculateATR(candles: Candle[], period: number = 14): number {
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

function calculateBollingerBands(closes: number[], period: number = 20): { upper: number; middle: number; lower: number } {
  if (closes.length < period) {
    const last = closes[closes.length - 1];
    return { upper: last, middle: last, lower: last };
  }
  
  const slice = closes.slice(-period);
  const sma = slice.reduce((a, b) => a + b, 0) / period;
  const variance = slice.reduce((a, b) => a + Math.pow(b - sma, 2), 0) / period;
  const stdDev = Math.sqrt(variance);
  
  return { upper: sma + stdDev * 2, middle: sma, lower: sma - stdDev * 2 };
}

function calculateKeltnerChannel(candles: Candle[], emaPeriod: number = 20, atrMultiplier: number = 2.0): { upper: number; middle: number; lower: number } {
  const closes = candles.map(c => c.close);
  const ema = calculateEMA(closes, emaPeriod);
  const atr = calculateATR(candles, emaPeriod);
  
  return { upper: ema + (atr * atrMultiplier), middle: ema, lower: ema - (atr * atrMultiplier) };
}

// ============================================
// MARKET STRUCTURE ANALYSIS
// ============================================
function findSwingPoints(candles: Candle[], lookback: number = 5): { highs: SwingPoint[]; lows: SwingPoint[] } {
  const highs: SwingPoint[] = [];
  const lows: SwingPoint[] = [];
  
  for (let i = lookback; i < candles.length - lookback; i++) {
    let isSwingHigh = true;
    let isSwingLow = true;
    
    for (let j = 1; j <= lookback; j++) {
      if (candles[i].high <= candles[i - j].high || candles[i].high <= candles[i + j].high) isSwingHigh = false;
      if (candles[i].low >= candles[i - j].low || candles[i].low >= candles[i + j].low) isSwingLow = false;
    }
    
    if (isSwingHigh) highs.push({ index: i, price: candles[i].high, type: 'HIGH', timestamp: candles[i].timestamp });
    if (isSwingLow) lows.push({ index: i, price: candles[i].low, type: 'LOW', timestamp: candles[i].timestamp });
  }
  
  return { highs, lows };
}

function analyzeMarketStructure(candles: Candle[]): MarketStructure {
  const { highs, lows } = findSwingPoints(candles, HTF_CONFIG.SWING_POINT_LOOKBACK);
  
  const lastSwingHigh = highs.length > 0 ? highs[highs.length - 1] : null;
  const lastSwingLow = lows.length > 0 ? lows[lows.length - 1] : null;
  
  let isHigherHighs = false, isHigherLows = false, isLowerHighs = false, isLowerLows = false;
  
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
  
  let trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
  if (isHigherHighs && isHigherLows) trend = 'BULLISH';
  else if (isLowerHighs && isLowerLows) trend = 'BEARISH';
  
  return { trend, isHigherHighs, isHigherLows, isLowerHighs, isLowerLows, lastSwingHigh, lastSwingLow };
}

// ============================================
// HTF TREND MATRIX
// ============================================
function calculateHTFBias(hourlyCandles: Candle[], fifteenMinCandles: Candle[]): HTFBias {
  const reasons: string[] = [];
  
  const hourlyCloses = hourlyCandles.map(c => c.close);
  const ema200 = calculateEMA(hourlyCloses, HTF_CONFIG.EMA_200_PERIOD);
  const currentPrice = hourlyCloses[hourlyCloses.length - 1];
  
  const hourlyTrend = currentPrice > ema200 ? 'ABOVE_200EMA' : 'BELOW_200EMA';
  reasons.push(`1H: Price ${hourlyTrend === 'ABOVE_200EMA' ? 'above' : 'below'} 200 EMA (${ema200.toFixed(5)})`);
  
  const fifteenMinStructure = analyzeMarketStructure(fifteenMinCandles);
  reasons.push(`15m: ${fifteenMinStructure.trend} structure (HH:${fifteenMinStructure.isHigherHighs}, HL:${fifteenMinStructure.isHigherLows})`);
  
  let direction: 'BUY' | 'SELL' | null = null;
  
  if (hourlyTrend === 'ABOVE_200EMA' && fifteenMinStructure.trend === 'BULLISH') {
    direction = 'BUY';
    reasons.push('✅ HTF Bias: BULLISH (1H > 200 EMA + 15m Bullish Structure)');
  } else if (hourlyTrend === 'BELOW_200EMA' && fifteenMinStructure.trend === 'BEARISH') {
    direction = 'SELL';
    reasons.push('✅ HTF Bias: BEARISH (1H < 200 EMA + 15m Bearish Structure)');
  } else {
    reasons.push('❌ HTF Bias: NEUTRAL (Conflicting timeframes)');
  }
  
  return { direction, hourlyTrend, fifteenMinStructure, ema200, reasons };
}

// ============================================
// LIQUIDITY SWEEP DETECTION
// ============================================
function detectLiquiditySweep(candles: Candle[], symbol: string, direction: 'BUY' | 'SELL'): LiquiditySweep | null {
  if (candles.length < LIQUIDITY_CONFIG.SWEEP_LOOKBACK_CANDLES + 2) return null;
  
  const closes = candles.map(c => c.close);
  const atr = calculateATR(candles);
  const bb = calculateBollingerBands(closes);
  const kc = calculateKeltnerChannel(candles);
  
  const lookbackCandles = candles.slice(-LIQUIDITY_CONFIG.SWEEP_LOOKBACK_CANDLES - 1, -1);
  
  if (direction === 'BUY') {
    const liquidityLevel = Math.min(...lookbackCandles.map(c => c.low));
    
    for (let i = candles.length - 1; i >= candles.length - 3 && i >= 0; i--) {
      const candle = candles[i];
      const piercedLiquidity = candle.low < liquidityLevel;
      const touchedValueArea = candle.low <= bb.lower || candle.low <= kc.lower;
      const tolerance = atr * LIQUIDITY_CONFIG.CLOSE_TOLERANCE_ATR;
      const closedInside = candle.close >= bb.lower - tolerance || candle.close >= kc.lower - tolerance;
      
      if (piercedLiquidity && touchedValueArea && closedInside) {
        return { detected: true, type: 'BUY', sweepPrice: candle.low, sweepLow: candle.low, sweepHigh: candle.high, candleIndex: i, liquidityLevel };
      }
    }
  } else {
    const liquidityLevel = Math.max(...lookbackCandles.map(c => c.high));
    
    for (let i = candles.length - 1; i >= candles.length - 3 && i >= 0; i--) {
      const candle = candles[i];
      const piercedLiquidity = candle.high > liquidityLevel;
      const touchedValueArea = candle.high >= bb.upper || candle.high >= kc.upper;
      const tolerance = atr * LIQUIDITY_CONFIG.CLOSE_TOLERANCE_ATR;
      const closedInside = candle.close <= bb.upper + tolerance || candle.close <= kc.upper + tolerance;
      
      if (piercedLiquidity && touchedValueArea && closedInside) {
        return { detected: true, type: 'SELL', sweepPrice: candle.high, sweepLow: candle.low, sweepHigh: candle.high, candleIndex: i, liquidityLevel };
      }
    }
  }
  
  return null;
}

// ============================================
// MARKET STRUCTURE SHIFT DETECTION
// ============================================
function detectMSS(candles: Candle[], sweepDirection: 'BUY' | 'SELL', sweepCandleIndex: number): MarketStructureShift {
  const structure = analyzeMarketStructure(candles);
  const atr = calculateATR(candles);
  
  for (let i = sweepCandleIndex + 1; i < candles.length; i++) {
    const candle = candles[i];
    const bodySize = Math.abs(candle.close - candle.open);
    const minBodySize = atr * MSS_CONFIG.MIN_DISPLACEMENT_BODY_ATR;
    
    if (bodySize < minBodySize) continue;
    
    if (sweepDirection === 'BUY') {
      const isBullishDisplacement = candle.close > candle.open;
      if (isBullishDisplacement && structure.lastSwingHigh) {
        const brokeSwinghigh = candle.close > structure.lastSwingHigh.price;
        if (brokeSwinghigh) {
          return { detected: true, direction: 'BULLISH', displacementIndex: i, swingBroken: structure.lastSwingHigh };
        }
      }
    } else {
      const isBearishDisplacement = candle.close < candle.open;
      if (isBearishDisplacement && structure.lastSwingLow) {
        const brokeSwingLow = candle.close < structure.lastSwingLow.price;
        if (brokeSwingLow) {
          return { detected: true, direction: 'BEARISH', displacementIndex: i, swingBroken: structure.lastSwingLow };
        }
      }
    }
  }
  
  return { detected: false, direction: sweepDirection === 'BUY' ? 'BULLISH' : 'BEARISH', displacementIndex: -1, swingBroken: null };
}

// ============================================
// FAIR VALUE GAP DETECTION
// ============================================
function detectFVG(candles: Candle[], direction: 'BUY' | 'SELL', startFromIndex: number, symbol: string): FairValueGap | null {
  const minGapPrice = pipToPrice(FVG_CONFIG.MIN_GAP_PIPS, symbol);
  
  for (let i = Math.max(startFromIndex, 2); i < candles.length - 1; i++) {
    const candle1 = candles[i - 2];
    const candle3 = candles[i];
    
    if (direction === 'BUY') {
      if (candle3.low > candle1.high) {
        const gapSize = candle3.low - candle1.high;
        if (gapSize >= minGapPrice) {
          const equilibrium = candle1.high + (gapSize * FVG_CONFIG.ENTRY_EQUILIBRIUM_LEVEL);
          const age = candles.length - 1 - i;
          return { detected: true, type: 'BULLISH', equilibrium, gapSizePips: priceToPips(gapSize, symbol), isValid: age <= FVG_CONFIG.MAX_FVG_AGE_CANDLES };
        }
      }
    } else {
      if (candle3.high < candle1.low) {
        const gapSize = candle1.low - candle3.high;
        if (gapSize >= minGapPrice) {
          const equilibrium = candle1.low - (gapSize * FVG_CONFIG.ENTRY_EQUILIBRIUM_LEVEL);
          const age = candles.length - 1 - i;
          return { detected: true, type: 'BEARISH', equilibrium, gapSizePips: priceToPips(gapSize, symbol), isValid: age <= FVG_CONFIG.MAX_FVG_AGE_CANDLES };
        }
      }
    }
  }
  
  return null;
}

// ============================================
// FULL ICT SIGNAL GENERATION
// ============================================
function generateICTSignal(hourlyCandles: Candle[], fifteenMinCandles: Candle[], oneMinCandles: Candle[], symbol: string): ICTSignal {
  const invalidReasons: string[] = [];
  const reasons: string[] = [];
  
  // Step 1: Calculate HTF Bias
  const htfBias = calculateHTFBias(hourlyCandles, fifteenMinCandles);
  reasons.push(...htfBias.reasons);
  
  if (!htfBias.direction) {
    invalidReasons.push('No HTF bias - conflicting timeframe signals');
    return { valid: false, direction: null, entryPrice: 0, stopLoss: 0, takeProfit: 0, confidence: 0, reasons, invalidReasons };
  }
  
  // Step 2: Detect Liquidity Sweep
  const liquiditySweep = detectLiquiditySweep(oneMinCandles, symbol, htfBias.direction);
  
  if (!liquiditySweep) {
    invalidReasons.push(`No liquidity sweep detected for ${htfBias.direction} direction`);
    return { valid: false, direction: null, entryPrice: 0, stopLoss: 0, takeProfit: 0, confidence: 0, reasons, invalidReasons };
  }
  
  reasons.push(`✅ Liquidity sweep detected at ${liquiditySweep.sweepPrice.toFixed(5)}`);
  
  // Step 3: Detect Market Structure Shift (MSS)
  const mss = detectMSS(oneMinCandles, htfBias.direction, liquiditySweep.candleIndex);
  
  if (!mss.detected) {
    invalidReasons.push('No market structure shift (MSS) after liquidity sweep');
    return { valid: false, direction: null, entryPrice: 0, stopLoss: 0, takeProfit: 0, confidence: 0, reasons, invalidReasons };
  }
  
  reasons.push(`✅ MSS detected: ${mss.direction} displacement`);
  
  // Step 4: Detect Fair Value Gap (FVG)
  const fvg = detectFVG(oneMinCandles, htfBias.direction, mss.displacementIndex, symbol);
  
  if (!fvg || !fvg.isValid) {
    invalidReasons.push(fvg ? `FVG too old` : 'No Fair Value Gap detected');
    return { valid: false, direction: null, entryPrice: 0, stopLoss: 0, takeProfit: 0, confidence: 0, reasons, invalidReasons };
  }
  
  reasons.push(`✅ FVG detected: ${fvg.gapSizePips.toFixed(1)} pips, entry at ${fvg.equilibrium.toFixed(5)}`);
  
  // Step 5: Calculate entry, SL, and TP
  const entryPrice = fvg.equilibrium;
  const slBufferPrice = pipToPrice(FVG_CONFIG.SL_BUFFER_PIPS, symbol);
  
  let stopLoss: number, takeProfit: number;
  
  if (htfBias.direction === 'BUY') {
    stopLoss = liquiditySweep.sweepLow - slBufferPrice;
    const riskDistance = entryPrice - stopLoss;
    takeProfit = entryPrice + (riskDistance * ICT_CONFIG.RRR_RATIO);
  } else {
    stopLoss = liquiditySweep.sweepHigh + slBufferPrice;
    const riskDistance = stopLoss - entryPrice;
    takeProfit = entryPrice - (riskDistance * ICT_CONFIG.RRR_RATIO);
  }
  
  const confidence = 75 + (mss.swingBroken ? 5 : 0) + (fvg.gapSizePips >= 3 ? 5 : 0);
  
  reasons.push(`📊 Entry: ${entryPrice.toFixed(5)}, SL: ${stopLoss.toFixed(5)}, TP: ${takeProfit.toFixed(5)}`);
  reasons.push(`📈 Risk/Reward: 1:${ICT_CONFIG.RRR_RATIO}`);
  
  return { valid: true, direction: htfBias.direction, entryPrice, stopLoss, takeProfit, confidence: Math.min(95, confidence), reasons, invalidReasons };
}

// ============================================
// SCAN SYMBOL
// ============================================
async function scanSymbolICT(supabase: any, symbol: string): Promise<{ success: boolean; opportunity?: any; message: string; reasons: string[] }> {
  console.log(`\n========== ICT Scanning ${symbol} ==========`);
  
  // Fetch 1-hour candles
  const { data: hourlyData, error: hourlyError } = await supabase
    .from('price_history')
    .select('*')
    .eq('symbol', symbol)
    .eq('timeframe', '1h')
    .order('timestamp', { ascending: true })
    .limit(300);

  if (hourlyError || !hourlyData || hourlyData.length < 200) {
    return { success: false, message: `Not enough 1H data for ${symbol}`, reasons: [] };
  }

  // Fetch 15-minute candles
  const { data: fifteenMinData, error: fifteenMinError } = await supabase
    .from('price_history')
    .select('*')
    .eq('symbol', symbol)
    .eq('timeframe', '15m')
    .order('timestamp', { ascending: true })
    .limit(100);

  if (fifteenMinError || !fifteenMinData || fifteenMinData.length < 50) {
    return { success: false, message: `Not enough 15m data for ${symbol}`, reasons: [] };
  }

  // Fetch 1-minute candles
  const { data: oneMinData, error: oneMinError } = await supabase
    .from('price_history')
    .select('*')
    .eq('symbol', symbol)
    .eq('timeframe', '1m')
    .order('timestamp', { ascending: true })
    .limit(100);

  if (oneMinError || !oneMinData || oneMinData.length < 50) {
    return { success: false, message: `Not enough 1m data for ${symbol}`, reasons: [] };
  }

  // Transform data
  const transformCandles = (data: any[]): Candle[] => data.map((p: any) => ({
    timestamp: p.timestamp,
    open: Number(p.open),
    high: Number(p.high),
    low: Number(p.low),
    close: Number(p.close),
    volume: p.volume ? Number(p.volume) : undefined
  }));

  const hourlyCandles = transformCandles(hourlyData);
  const fifteenMinCandles = transformCandles(fifteenMinData);
  const oneMinCandles = transformCandles(oneMinData);

  // Generate ICT signal
  const signal = generateICTSignal(hourlyCandles, fifteenMinCandles, oneMinCandles, symbol);

  if (!signal.valid) {
    console.log(`[${symbol}] No valid ICT signal: ${signal.invalidReasons.join(', ')}`);
    return { success: true, message: `No ICT signal for ${symbol}`, reasons: signal.invalidReasons };
  }

  // Check for duplicate opportunities
  const { data: existingOpps } = await supabase
    .from('trading_opportunities')
    .select('id, created_at')
    .eq('symbol', symbol)
    .eq('signal_type', signal.direction)
    .eq('status', 'ACTIVE')
    .gte('created_at', new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString());

  if (existingOpps && existingOpps.length > 0) {
    return { success: true, message: `Active ${signal.direction} opportunity already exists for ${symbol}`, reasons: signal.reasons };
  }

  // Check daily trade limit
  const todayStart = new Date();
  todayStart.setUTCHours(0, 0, 0, 0);
  
  const { data: todayTrades } = await supabase
    .from('trading_opportunities')
    .select('id')
    .eq('symbol', symbol)
    .gte('created_at', todayStart.toISOString());

  if (todayTrades && todayTrades.length >= ICT_CONFIG.MAX_TRADES_PER_PAIR_PER_DAY) {
    return { success: true, message: `Daily trade limit reached for ${symbol}`, reasons: signal.reasons };
  }

  // Build reasoning
  const reasoning = `ICT ${signal.direction} opportunity detected on ${symbol} with ${signal.confidence.toFixed(0)}% confidence.\n\n` +
    `Signal Flow:\n${signal.reasons.map(r => `• ${r}`).join('\n')}\n\n` +
    `Entry Type: Limit Order at FVG 50% Equilibrium\n` +
    `Risk/Reward: 1:${ICT_CONFIG.RRR_RATIO}`;

  // Insert opportunity
  const expiresAt = new Date(Date.now() + ICT_CONFIG.POSITION_EXPIRY_HOURS * 60 * 60 * 1000);

  const { data: newOpp, error: insertError } = await supabase
    .from('trading_opportunities')
    .insert({
      symbol,
      signal_type: signal.direction,
      confidence: signal.confidence,
      entry_price: signal.entryPrice,
      current_price: oneMinCandles[oneMinCandles.length - 1].close,
      stop_loss: signal.stopLoss,
      take_profit_1: signal.takeProfit,
      take_profit_2: signal.takeProfit * 1.2,
      patterns_detected: ['ICT_HTF_BIAS', 'LIQUIDITY_SWEEP', 'MSS', 'FVG'],
      technical_indicators: { ema200: fifteenMinCandles[fifteenMinCandles.length - 1].close },
      reasoning,
      status: 'ACTIVE',
      expires_at: expiresAt.toISOString()
    })
    .select()
    .single();

  if (insertError) {
    console.error(`[${symbol}] Failed to insert opportunity:`, insertError);
    return { success: false, message: `Failed to save opportunity for ${symbol}`, reasons: signal.reasons };
  }

  console.log(`[${symbol}] Created ICT opportunity:`, newOpp.id);

  // Send Telegram notification
  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    
    await fetch(`${supabaseUrl}/functions/v1/send-telegram-notification`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${supabaseAnonKey}` },
      body: JSON.stringify({
        symbol,
        signal_type: newOpp.signal_type,
        confidence: newOpp.confidence,
        entry_price: newOpp.entry_price,
        stop_loss: newOpp.stop_loss,
        take_profit_1: newOpp.take_profit_1,
        reasoning: newOpp.reasoning,
      }),
    });
  } catch (notifyError) {
    console.error(`[${symbol}] Failed to send Telegram notification:`, notifyError);
  }

  return { success: true, opportunity: newOpp, message: `New ICT ${signal.direction} opportunity for ${symbol}!`, reasons: signal.reasons };
}

// ============================================
// MAIN SERVE HANDLER
// ============================================
serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    console.log("Starting ICT/SMC multi-currency opportunity scan...");
    
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    const body = await req.json().catch(() => ({}));
    
    // Check market status
    const marketStatus = isForexMarketOpen();
    if (!marketStatus.isOpen) {
      console.log("Market closed:", marketStatus.reason);
      return new Response(
        JSON.stringify({ success: true, message: marketStatus.reason, scanned: false }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Check killzone
    const killzoneStatus = isInKillzone();
    if (!killzoneStatus.inKillzone) {
      console.log("Outside killzone - no new trades");
      return new Response(
        JSON.stringify({ 
          success: true, 
          message: "Outside trading killzone (London 07:00-10:00, NY 12:00-15:00 UTC)", 
          scanned: false,
          nextKillzone: killzoneStatus.killzone 
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`In ${killzoneStatus.killzone} killzone - scanning for opportunities`);

    // Get active pairs
    let requestedSymbols: string[];
    if (body?.symbols) {
      requestedSymbols = body.symbols;
    } else if (body?.symbol) {
      requestedSymbols = [body.symbol];
    } else {
      const { data: activePairs } = await supabase
        .from('supported_currency_pairs')
        .select('symbol, pip_value')
        .eq('is_active', true);
      
      if (!activePairs || activePairs.length === 0) {
        return new Response(
          JSON.stringify({ success: true, message: "No active currency pairs configured", scanned: false }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      // Update pip values
      dynamicPipValues = {};
      for (const pair of activePairs) {
        dynamicPipValues[pair.symbol] = Number(pair.pip_value);
      }
      
      requestedSymbols = activePairs.map(p => p.symbol);
    }

    // Expire old opportunities
    await supabase
      .from('trading_opportunities')
      .update({ status: 'EXPIRED' })
      .eq('status', 'ACTIVE')
      .lt('expires_at', new Date().toISOString());

    // Scan each symbol
    const results: { symbol: string; opportunity?: any; message: string; reasons: string[] }[] = [];
    const newOpportunities: any[] = [];
    
    for (const symbol of requestedSymbols) {
      const result = await scanSymbolICT(supabase, symbol);
      results.push({ symbol, ...result });
      if (result.opportunity) {
        newOpportunities.push(result.opportunity);
      }
    }

    console.log(`\n========== ICT Scan Complete ==========`);
    console.log(`Scanned ${requestedSymbols.length} pairs, found ${newOpportunities.length} opportunities`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: newOpportunities.length > 0 
          ? `Found ${newOpportunities.length} new ICT opportunity(ies)!` 
          : "No high-confluence ICT opportunities detected",
        scanned: true,
        killzone: killzoneStatus.killzone,
        symbolsScanned: requestedSymbols.length,
        opportunitiesFound: newOpportunities.length,
        opportunities: newOpportunities,
        results
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("ICT Scan error:", error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
