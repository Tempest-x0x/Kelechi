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

// Analyze opportunity with pattern stats
function analyzeOpportunity(
  indicators: TechnicalIndicators,
  patterns: string[],
  currentPrice: number,
  patternStats: any[]
): { signal: 'BUY' | 'SELL' | null; confidence: number; reasons: string[]; patternData: any[] } {
  const buyReasons: string[] = [];
  const sellReasons: string[] = [];
  const matchedPatternStats: any[] = [];
  
  // RSI
  if (indicators.rsi < 30) {
    buyReasons.push('RSI oversold (<30)');
    const stat = patternStats.find(p => p.pattern_name === 'rsi_oversold');
    if (stat) matchedPatternStats.push(stat);
  } else if (indicators.rsi < 40) {
    buyReasons.push('RSI approaching oversold');
  }
  
  if (indicators.rsi > 70) {
    sellReasons.push('RSI overbought (>70)');
    const stat = patternStats.find(p => p.pattern_name === 'rsi_overbought');
    if (stat) matchedPatternStats.push(stat);
  } else if (indicators.rsi > 60) {
    sellReasons.push('RSI approaching overbought');
  }
  
  // MACD
  if (indicators.macd.histogram > 0 && indicators.macd.value > indicators.macd.signal) {
    buyReasons.push('MACD bullish crossover');
    const stat = patternStats.find(p => p.pattern_name === 'macd_bullish_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  if (indicators.macd.histogram < 0 && indicators.macd.value < indicators.macd.signal) {
    sellReasons.push('MACD bearish crossover');
    const stat = patternStats.find(p => p.pattern_name === 'macd_bearish_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // Bollinger Bands
  if (currentPrice < indicators.bollingerBands.lower) {
    buyReasons.push('Price below lower Bollinger Band');
    const stat = patternStats.find(p => p.pattern_name === 'bb_lower_touch');
    if (stat) matchedPatternStats.push(stat);
  }
  if (currentPrice > indicators.bollingerBands.upper) {
    sellReasons.push('Price above upper Bollinger Band');
    const stat = patternStats.find(p => p.pattern_name === 'bb_upper_touch');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // Stochastic
  if (indicators.stochastic.k < 20 && indicators.stochastic.d < 20) {
    buyReasons.push('Stochastic deeply oversold');
  }
  if (indicators.stochastic.k > 80 && indicators.stochastic.d > 80) {
    sellReasons.push('Stochastic deeply overbought');
  }
  
  // EMA alignment
  const priceAboveEma21 = currentPrice > indicators.ema21;
  const priceAboveEma50 = currentPrice > indicators.ema50;
  const ema21AboveEma50 = indicators.ema21 > indicators.ema50;
  
  if (priceAboveEma21 && priceAboveEma50 && ema21AboveEma50) {
    buyReasons.push('Strong bullish EMA alignment');
    const stat = patternStats.find(p => p.pattern_name === 'golden_cross');
    if (stat) matchedPatternStats.push(stat);
  } else if (!priceAboveEma21 && !priceAboveEma50 && !ema21AboveEma50) {
    sellReasons.push('Strong bearish EMA alignment');
    const stat = patternStats.find(p => p.pattern_name === 'death_cross');
    if (stat) matchedPatternStats.push(stat);
  }
  
  // Pattern analysis
  patterns.forEach(p => {
    if (p.includes('Bullish')) {
      buyReasons.push(p);
      const stat = patternStats.find(s => s.pattern_name === 'bullish_engulfing');
      if (stat && !matchedPatternStats.includes(stat)) matchedPatternStats.push(stat);
    }
    if (p.includes('Bearish')) {
      sellReasons.push(p);
      const stat = patternStats.find(s => s.pattern_name === 'bearish_engulfing');
      if (stat && !matchedPatternStats.includes(stat)) matchedPatternStats.push(stat);
    }
    if (p.includes('Uptrend')) buyReasons.push(p);
    if (p.includes('Downtrend')) sellReasons.push(p);
  });
  
  // Decision logic: need 2+ reasons AND net advantage
  const netBuyAdvantage = buyReasons.length - sellReasons.length;
  const netSellAdvantage = sellReasons.length - buyReasons.length;
  
  // Calculate confidence based on reasons and pattern stats
  let baseConfidence = 50;
  let signal: 'BUY' | 'SELL' | null = null;
  let reasons: string[] = [];
  
  if (buyReasons.length >= 2 && netBuyAdvantage >= 1) {
    signal = 'BUY';
    reasons = buyReasons;
    baseConfidence = 55 + (buyReasons.length * 5) + (netBuyAdvantage * 3);
    
    // Boost or reduce based on pattern stats
    const goodStats = matchedPatternStats.filter(s => s.signal_type === 'BUY' && s.win_rate_24h >= 51);
    const badStats = matchedPatternStats.filter(s => s.signal_type === 'BUY' && s.win_rate_24h < 48);
    baseConfidence += goodStats.length * 3;
    baseConfidence -= badStats.length * 5;
  } else if (sellReasons.length >= 2 && netSellAdvantage >= 1) {
    signal = 'SELL';
    reasons = sellReasons;
    baseConfidence = 55 + (sellReasons.length * 5) + (netSellAdvantage * 3);
    
    const goodStats = matchedPatternStats.filter(s => s.signal_type === 'SELL' && s.win_rate_24h >= 51);
    const badStats = matchedPatternStats.filter(s => s.signal_type === 'SELL' && s.win_rate_24h < 48);
    baseConfidence += goodStats.length * 3;
    baseConfidence -= badStats.length * 5;
  }
  
  // Clamp confidence
  baseConfidence = Math.min(95, Math.max(0, baseConfidence));
  
  return { signal, confidence: baseConfidence, reasons, patternData: matchedPatternStats };
}

// Calculate ATR-based levels
function calculateLevels(
  currentPrice: number, 
  atr: number, 
  signalType: 'BUY' | 'SELL'
): { stopLoss: number; takeProfit1: number; takeProfit2: number } {
  const slMult = 2.0;
  const tp1Mult = 2.0;
  const tp2Mult = 3.5;
  
  if (signalType === 'BUY') {
    return {
      stopLoss: currentPrice - (atr * slMult),
      takeProfit1: currentPrice + (atr * tp1Mult),
      takeProfit2: currentPrice + (atr * tp2Mult)
    };
  } else {
    return {
      stopLoss: currentPrice + (atr * slMult),
      takeProfit1: currentPrice - (atr * tp1Mult),
      takeProfit2: currentPrice - (atr * tp2Mult)
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

    // Only create opportunity if conditions are met
    // - Signal is BUY or SELL (not null)
    // - Confidence >= 65%
    // - At least 2 confirming indicators
    if (!analysis.signal || analysis.confidence < 65 || analysis.reasons.length < 2) {
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

    // Check for duplicate active opportunity
    const { data: existingOpps } = await supabase
      .from('trading_opportunities')
      .select('id, signal_type, created_at')
      .eq('status', 'ACTIVE')
      .eq('signal_type', analysis.signal)
      .gte('created_at', new Date(Date.now() - 30 * 60 * 1000).toISOString()); // Last 30 mins

    if (existingOpps && existingOpps.length > 0) {
      console.log("Similar opportunity already exists from recent scan");
      return new Response(
        JSON.stringify({ 
          success: true, 
          message: "Similar opportunity already active", 
          scanned: true,
          existingOpportunity: existingOpps[0].id
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Calculate entry levels
    const levels = calculateLevels(currentPrice, indicators.atr, analysis.signal);

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
