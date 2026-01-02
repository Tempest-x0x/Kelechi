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
}

interface BacktestTrade {
  entryTime: string;
  exitTime: string;
  signalType: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  stopLoss: number;
  takeProfit: number;
  outcome: 'WIN' | 'LOSS';
  pips: number;
  confidence: number;
  patterns: string[];
}

interface BacktestResult {
  period: { start: string; end: string };
  totalSignals: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPips: number;
  avgPipsPerTrade: number;
  profitFactor: number;
  maxDrawdownPips: number;
  patternPerformance: Record<string, { wins: number; losses: number; winRate: number }>;
  trades: BacktestTrade[];
}

// Calculate RSI
function calculateRSI(closes: number[], period = 14): number {
  if (closes.length < period + 1) return 50;
  let gains = 0, losses = 0;
  for (let i = closes.length - period; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }
  const avgGain = gains / period;
  const avgLoss = losses / period;
  if (avgLoss === 0) return 100;
  return 100 - (100 / (1 + avgGain / avgLoss));
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
  return { value: macdLine, signal: signalLine, histogram: macdLine - signalLine };
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
  return { upper: sma + stdDev * 2, middle: sma, lower: sma - stdDev * 2 };
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
  return { k, d: k };
}

// Calculate ATR
function calculateATR(highs: number[], lows: number[], closes: number[], period = 14): number {
  if (closes.length < period + 1) return 0.001;
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

// Detect patterns
function detectPatterns(candles: Candle[], index: number): string[] {
  const patterns: string[] = [];
  if (index < 50) return patterns;
  
  const slice = candles.slice(Math.max(0, index - 50), index + 1);
  const closes = slice.map(c => c.close);
  const opens = slice.map(c => c.open);
  const highs = slice.map(c => c.high);
  const lows = slice.map(c => c.low);
  const last = slice.length - 1;

  // Doji
  const body = Math.abs(closes[last] - opens[last]);
  const range = highs[last] - lows[last];
  if (range > 0 && body / range < 0.1) patterns.push('Doji');

  // Engulfing
  if (last >= 1) {
    const prevBody = Math.abs(closes[last-1] - opens[last-1]);
    const currBody = Math.abs(closes[last] - opens[last]);
    if (closes[last-1] < opens[last-1] && closes[last] > opens[last] && currBody > prevBody) {
      patterns.push('Bullish Engulfing');
    }
    if (closes[last-1] > opens[last-1] && closes[last] < opens[last] && currBody > prevBody) {
      patterns.push('Bearish Engulfing');
    }
  }

  // Trend
  const ema20 = calculateEMA(closes, 20);
  const ema50 = calculateEMA(closes, 50);
  if (closes[last] > ema20 && ema20 > ema50) patterns.push('Uptrend');
  else if (closes[last] < ema20 && ema20 < ema50) patterns.push('Downtrend');

  return patterns;
}

// Analyze for signal
function analyzeSignal(
  candles: Candle[],
  index: number,
  minConfidence: number
): { signal: 'BUY' | 'SELL' | null; confidence: number; patterns: string[] } {
  if (index < 50) return { signal: null, confidence: 0, patterns: [] };

  const slice = candles.slice(Math.max(0, index - 200), index + 1);
  const closes = slice.map(c => c.close);
  const highs = slice.map(c => c.high);
  const lows = slice.map(c => c.low);
  const currentPrice = closes[closes.length - 1];

  const rsi = calculateRSI(closes);
  const macd = calculateMACD(closes);
  const bb = calculateBollingerBands(closes);
  const stoch = calculateStochastic(highs, lows, closes);
  const patterns = detectPatterns(candles, index);

  const buyReasons: string[] = [];
  const sellReasons: string[] = [];

  if (rsi < 30) buyReasons.push('RSI oversold');
  if (rsi > 70) sellReasons.push('RSI overbought');
  if (macd.histogram > 0 && macd.value > macd.signal) buyReasons.push('MACD bullish');
  if (macd.histogram < 0 && macd.value < macd.signal) sellReasons.push('MACD bearish');
  if (currentPrice < bb.lower) buyReasons.push('BB lower touch');
  if (currentPrice > bb.upper) sellReasons.push('BB upper touch');
  if (stoch.k < 20) buyReasons.push('Stochastic oversold');
  if (stoch.k > 80) sellReasons.push('Stochastic overbought');

  patterns.forEach(p => {
    if (p.includes('Bullish') || p === 'Uptrend') buyReasons.push(p);
    if (p.includes('Bearish') || p === 'Downtrend') sellReasons.push(p);
  });

  const netBuy = buyReasons.length - sellReasons.length;
  const netSell = sellReasons.length - buyReasons.length;

  let signal: 'BUY' | 'SELL' | null = null;
  let confidence = 50;

  if (buyReasons.length >= 2 && netBuy >= 1) {
    signal = 'BUY';
    confidence = 55 + buyReasons.length * 5 + netBuy * 3;
  } else if (sellReasons.length >= 2 && netSell >= 1) {
    signal = 'SELL';
    confidence = 55 + sellReasons.length * 5 + netSell * 3;
  }

  confidence = Math.min(95, Math.max(0, confidence));

  if (confidence < minConfidence) {
    return { signal: null, confidence, patterns };
  }

  return { signal, confidence, patterns };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { 
      startDate, 
      endDate, 
      minConfidence = 65,
      maxTrades = 500 
    } = body;

    console.log(`Starting backtest from ${startDate} to ${endDate}`);

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Fetch historical data with pagination to handle large datasets
    const allPriceData: any[] = [];
    let lastTimestamp: string | null = null;
    const PAGE_SIZE = 50000;
    
    console.log(`Fetching price data from ${startDate || 'beginning'} to ${endDate || 'end'}`);
    
    while (true) {
      let query = supabase
        .from('price_history')
        .select('*')
        .eq('symbol', 'EUR/USD')
        .eq('timeframe', '1h')
        .order('timestamp', { ascending: true })
        .limit(PAGE_SIZE);

      if (startDate) query = query.gte('timestamp', startDate);
      if (endDate) query = query.lte('timestamp', endDate);
      if (lastTimestamp) query = query.gt('timestamp', lastTimestamp);

      const { data: pageData, error: pageError } = await query;

      if (pageError) {
        throw new Error(`Failed to fetch price data: ${pageError.message}`);
      }

      if (!pageData || pageData.length === 0) break;

      allPriceData.push(...pageData);
      lastTimestamp = pageData[pageData.length - 1].timestamp;
      
      console.log(`Loaded ${allPriceData.length} candles so far...`);

      // If we got less than page size, we've reached the end
      if (pageData.length < PAGE_SIZE) break;
      
      // Safety limit to prevent infinite loops
      if (allPriceData.length > 500000) {
        console.log('Reached 500k candle limit');
        break;
      }
    }

    const priceData = allPriceData;
    console.log(`Total loaded: ${priceData.length} candles`);

    if (priceData.length < 100) {
      return new Response(
        JSON.stringify({ success: false, error: "Not enough historical data for backtest" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const candles: Candle[] = priceData.map(p => ({
      timestamp: p.timestamp,
      open: Number(p.open),
      high: Number(p.high),
      low: Number(p.low),
      close: Number(p.close)
    }));

    const trades: BacktestTrade[] = [];
    const patternPerformance: Record<string, { wins: number; losses: number }> = {};
    
    let i = 100; // Start after enough history for indicators
    let lastTradeTime = 0;

    while (i < candles.length - 24 && trades.length < maxTrades) {
      // Skip if we just made a trade (cooldown of 4 candles)
      if (i - lastTradeTime < 4) {
        i++;
        continue;
      }

      const { signal, confidence, patterns } = analyzeSignal(candles, i, minConfidence);

      if (signal) {
        const entryCandle = candles[i];
        const entryPrice = entryCandle.close;
        
        // Calculate ATR for SL/TP
        const slice = candles.slice(Math.max(0, i - 100), i + 1);
        const closes = slice.map(c => c.close);
        const highs = slice.map(c => c.high);
        const lows = slice.map(c => c.low);
        const atr = calculateATR(highs, lows, closes);

        const slMult = 2.0;
        const tpMult = 2.0;

        let stopLoss: number;
        let takeProfit: number;

        if (signal === 'BUY') {
          stopLoss = entryPrice - atr * slMult;
          takeProfit = entryPrice + atr * tpMult;
        } else {
          stopLoss = entryPrice + atr * slMult;
          takeProfit = entryPrice - atr * tpMult;
        }

        // Simulate trade through next candles
        let exitPrice = entryPrice;
        let exitTime = entryCandle.timestamp;
        let outcome: 'WIN' | 'LOSS' = 'LOSS';

        for (let j = i + 1; j < Math.min(i + 24, candles.length); j++) {
          const c = candles[j];
          
          if (signal === 'BUY') {
            if (c.low <= stopLoss) {
              exitPrice = stopLoss;
              exitTime = c.timestamp;
              outcome = 'LOSS';
              break;
            }
            if (c.high >= takeProfit) {
              exitPrice = takeProfit;
              exitTime = c.timestamp;
              outcome = 'WIN';
              break;
            }
          } else {
            if (c.high >= stopLoss) {
              exitPrice = stopLoss;
              exitTime = c.timestamp;
              outcome = 'LOSS';
              break;
            }
            if (c.low <= takeProfit) {
              exitPrice = takeProfit;
              exitTime = c.timestamp;
              outcome = 'WIN';
              break;
            }
          }
          
          // If we reach the end without hitting SL/TP, close at last price
          if (j === Math.min(i + 23, candles.length - 1)) {
            exitPrice = c.close;
            exitTime = c.timestamp;
            // Determine outcome based on P&L
            if (signal === 'BUY') {
              outcome = c.close > entryPrice ? 'WIN' : 'LOSS';
            } else {
              outcome = c.close < entryPrice ? 'WIN' : 'LOSS';
            }
          }
        }

        const pips = signal === 'BUY' 
          ? (exitPrice - entryPrice) * 10000 
          : (entryPrice - exitPrice) * 10000;

        trades.push({
          entryTime: entryCandle.timestamp,
          exitTime,
          signalType: signal,
          entryPrice,
          exitPrice,
          stopLoss,
          takeProfit,
          outcome,
          pips,
          confidence,
          patterns
        });

        // Track pattern performance
        patterns.forEach(p => {
          if (!patternPerformance[p]) {
            patternPerformance[p] = { wins: 0, losses: 0 };
          }
          if (outcome === 'WIN') patternPerformance[p].wins++;
          else patternPerformance[p].losses++;
        });

        lastTradeTime = i;
      }

      i++;
    }

    // Calculate results
    const wins = trades.filter(t => t.outcome === 'WIN').length;
    const losses = trades.filter(t => t.outcome === 'LOSS').length;
    const totalPips = trades.reduce((sum, t) => sum + t.pips, 0);
    const grossProfit = trades.filter(t => t.pips > 0).reduce((sum, t) => sum + t.pips, 0);
    const grossLoss = Math.abs(trades.filter(t => t.pips < 0).reduce((sum, t) => sum + t.pips, 0));

    // Calculate max drawdown
    let peak = 0;
    let maxDrawdown = 0;
    let runningPnl = 0;
    trades.forEach(t => {
      runningPnl += t.pips;
      if (runningPnl > peak) peak = runningPnl;
      const drawdown = peak - runningPnl;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    });

    const result: BacktestResult = {
      period: {
        start: candles[0].timestamp,
        end: candles[candles.length - 1].timestamp
      },
      totalSignals: trades.length,
      wins,
      losses,
      winRate: trades.length > 0 ? (wins / trades.length) * 100 : 0,
      totalPips,
      avgPipsPerTrade: trades.length > 0 ? totalPips / trades.length : 0,
      profitFactor: grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0,
      maxDrawdownPips: maxDrawdown,
      patternPerformance: Object.fromEntries(
        Object.entries(patternPerformance).map(([k, v]) => [
          k,
          { ...v, winRate: v.wins + v.losses > 0 ? (v.wins / (v.wins + v.losses)) * 100 : 0 }
        ])
      ),
      trades: trades.slice(-100) // Return last 100 trades for display
    };

    console.log(`Backtest complete: ${wins}W/${losses}L, ${result.winRate.toFixed(1)}% win rate`);

    return new Response(
      JSON.stringify({ success: true, result }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Backtest error:", error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
