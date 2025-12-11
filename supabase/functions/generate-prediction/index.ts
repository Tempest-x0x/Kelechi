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

    const indicators = calculateIndicators(candles);
    const patterns = detectPatterns(candles);
    const timeframeSettings = getTimeframeSettings(timeframe);

    console.log("Technical indicators:", JSON.stringify(indicators));
    console.log("Patterns detected:", patterns);

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

    // Analyze past performance for learning context
    let learningContext = '';
    if (pastPredictions && pastPredictions.length > 0) {
      const wins = pastPredictions.filter(p => p.outcome === 'WIN').length;
      const losses = pastPredictions.filter(p => p.outcome === 'LOSS').length;
      const pending = pastPredictions.filter(p => !p.outcome || p.outcome === 'PENDING').length;
      const winRate = wins + losses > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : 'N/A';

      const failedTrades = pastPredictions.filter(p => p.outcome === 'LOSS');
      const failedAnalysis = failedTrades.slice(0, 5).map(t => 
        `${t.signal_type} at ${t.entry_price}, SL: ${t.stop_loss}`
      ).join('; ');

      const recentSignals = pastPredictions.slice(0, 5).map(p => p.signal_type);
      const buyCount = recentSignals.filter(s => s === 'BUY').length;
      const sellCount = recentSignals.filter(s => s === 'SELL').length;

      learningContext = `
HISTORICAL PERFORMANCE (Last 15 trades):
- Wins: ${wins}, Losses: ${losses}, Pending: ${pending}
- Win Rate: ${winRate}%

${failedTrades.length > 0 ? `RECENT FAILED TRADES (analyze and avoid similar):
${failedAnalysis}` : ''}

${buyCount >= 4 ? 'CAUTION: Many recent BUY signals. Consider if market is overbought.' : ''}
${sellCount >= 4 ? 'CAUTION: Many recent SELL signals. Consider if market is oversold.' : ''}`;
    }

    // Call Lovable AI for prediction
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const analysisPrompt = `You are an expert forex trading analyst specializing in EUR/USD. Analyze the data and provide a trading signal.

TIMEFRAME: ${timeframe} (${timeframeSettings.description})
CURRENT PRICE: ${currentPrice}
SENTIMENT SCORE: ${sentimentScore} (range: -100 to 100)

TECHNICAL INDICATORS:
- RSI (14): ${indicators.rsi.toFixed(2)} ${indicators.rsi > 70 ? '(OVERBOUGHT)' : indicators.rsi < 30 ? '(OVERSOLD)' : ''}
- MACD: ${indicators.macd.value.toFixed(5)} (Signal: ${indicators.macd.signal.toFixed(5)}, Histogram: ${indicators.macd.histogram.toFixed(5)})
- EMA 9: ${indicators.ema9.toFixed(5)}
- EMA 21: ${indicators.ema21.toFixed(5)}
- EMA 50: ${indicators.ema50.toFixed(5)}
- EMA 200: ${indicators.ema200.toFixed(5)}
- Bollinger: Upper ${indicators.bollingerBands.upper.toFixed(5)}, Middle ${indicators.bollingerBands.middle.toFixed(5)}, Lower ${indicators.bollingerBands.lower.toFixed(5)}
- Stochastic: %K ${indicators.stochastic.k.toFixed(2)}, %D ${indicators.stochastic.d.toFixed(2)}
- ATR (14): ${indicators.atr.toFixed(5)}
- Support: ${indicators.supportLevels.map(s => s.toFixed(5)).join(', ') || 'None'}
- Resistance: ${indicators.resistanceLevels.map(r => r.toFixed(5)).join(', ') || 'None'}

PATTERNS DETECTED:
${patterns.length > 0 ? patterns.map(p => `- ${p}`).join('\n') : '- No significant patterns'}

${learningContext}

TIMEFRAME GUIDANCE:
${timeframe === '15m' ? 'Use tight stops (5-15 pips), quick TP targets. Focus on momentum.' : ''}
${timeframe === '1h' ? 'Standard stops (15-30 pips), balanced risk/reward.' : ''}
${timeframe === '4h' ? 'Wider stops (30-50 pips), focus on swing levels.' : ''}
${timeframe === '1d' ? 'Wide stops (50-100 pips), major support/resistance focus.' : ''}

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
          { role: "system", content: "You are an expert forex trading analyst. Provide structured trading signals with specific price targets and risk management." },
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
                  signal_type: { type: "string", enum: ["BUY", "SELL", "HOLD"], description: "The trading signal" },
                  confidence: { type: "number", description: "Confidence level 0-100" },
                  entry_price: { type: "number", description: "Recommended entry price" },
                  take_profit_1: { type: "number", description: "First take profit target" },
                  take_profit_2: { type: "number", description: "Second take profit target" },
                  stop_loss: { type: "number", description: "Stop loss price" },
                  trend_direction: { type: "string", enum: ["BULLISH", "BEARISH", "NEUTRAL"], description: "Overall trend direction" },
                  trend_strength: { type: "number", description: "Trend strength 0-100" },
                  sentiment_score: { type: "number", description: "Market sentiment -100 to 100" },
                  reasoning: { type: "string", description: "Detailed reasoning for the signal" }
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

    const signal = JSON.parse(toolCall.function.arguments);
    console.log("Parsed signal:", signal);

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
