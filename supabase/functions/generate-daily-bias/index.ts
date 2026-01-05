import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { type, date } = await req.json();
    console.log(`Generating ${type} bias for ${date}`);

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Fetch latest price data
    const { data: priceData, error: priceError } = await supabase
      .from("price_history")
      .select("*")
      .order("timestamp", { ascending: false })
      .limit(200);

    if (priceError) {
      console.error("Error fetching price data:", priceError);
      throw priceError;
    }

    // Fetch recent predictions for context
    const { data: recentPredictions, error: predError } = await supabase
      .from("predictions")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(10);

    if (predError) {
      console.error("Error fetching predictions:", predError);
    }

    // Calculate key levels from price data
    const prices = priceData?.map((p) => ({
      open: p.open,
      high: p.high,
      low: p.low,
      close: p.close,
      timestamp: p.timestamp,
    })) || [];

    const latestPrice = prices[0]?.close || 0;
    const highs = prices.slice(0, 50).map((p) => p.high);
    const lows = prices.slice(0, 50).map((p) => p.low);
    
    const recentHigh = Math.max(...highs);
    const recentLow = Math.min(...lows);
    
    // Calculate simple moving averages
    const closes = prices.map((p) => p.close);
    const sma20 = closes.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
    const sma50 = closes.slice(0, 50).reduce((a, b) => a + b, 0) / 50;

    // Determine preliminary bias
    const isBullish = latestPrice > sma20 && sma20 > sma50;
    const isBearish = latestPrice < sma20 && sma20 < sma50;
    const preliminaryBias = isBullish ? "BULLISH" : isBearish ? "BEARISH" : "NEUTRAL";

    // Calculate today's key levels
    const todayPrices = prices.slice(0, 24); // Assuming hourly data
    const todayHigh = todayPrices.length > 0 ? Math.max(...todayPrices.map((p) => p.high)) : recentHigh;
    const todayLow = todayPrices.length > 0 ? Math.min(...todayPrices.map((p) => p.low)) : recentLow;

    const analysisContext = {
      date,
      type,
      currentPrice: latestPrice,
      todayRange: { high: todayHigh, low: todayLow },
      keyLevels: {
        resistance: recentHigh,
        support: recentLow,
        sma20,
        sma50,
      },
      preliminaryBias,
      recentPredictions: recentPredictions?.slice(0, 5).map((p) => ({
        signal: p.signal_type,
        confidence: p.confidence,
        outcome: p.outcome,
        entry: p.entry_price,
      })),
    };

    // Generate AI content
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const systemPrompt = type === "morning" 
      ? `You are a professional forex analyst creating a morning bias post for X (Twitter).
Generate a concise, actionable EUR/USD daily bias post that includes:
- Clear directional bias (BULLISH/BEARISH/NEUTRAL)
- Confidence level (percentage)
- Key invalidation level (where bias flips)
- 2 target levels
- One thing to watch for

Keep it under 280 characters for the main content, with key levels on separate lines.
Use clean formatting with bullet points (•).`
      : `You are a professional forex analyst creating an evening recap post for X (Twitter).
Generate a concise EUR/USD end-of-day recap that includes:
- Whether the bias was correct
- Actual high and low of the day
- How it compared to targets
- Brief outlook for tomorrow

Keep it under 280 characters. Be honest about results.`;

    const aiResponse = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: `Generate a ${type} post for EUR/USD based on this data:

${JSON.stringify(analysisContext, null, 2)}

${type === "morning" 
  ? `Create the morning bias post. Current preliminary bias: ${preliminaryBias}.
Use these levels:
- Current price: ${latestPrice.toFixed(4)}
- Resistance: ${recentHigh.toFixed(4)}
- Support: ${recentLow.toFixed(4)}
- 20 SMA: ${sma20.toFixed(4)}
- 50 SMA: ${sma50.toFixed(4)}`
  : `Create the evening recap post.
Today's range:
- High: ${todayHigh.toFixed(4)}
- Low: ${todayLow.toFixed(4)}
- Close: ${latestPrice.toFixed(4)}`
}`,
          },
        ],
      }),
    });

    if (!aiResponse.ok) {
      const errorText = await aiResponse.text();
      console.error("AI API error:", aiResponse.status, errorText);
      throw new Error(`AI API error: ${aiResponse.status}`);
    }

    const aiData = await aiResponse.json();
    const aiContent = aiData.choices?.[0]?.message?.content || "";

    // Fallback content if AI fails
    const fallbackMorning = `EUR/USD Daily Bias - ${date}

Direction: ${preliminaryBias}
Confidence: ${preliminaryBias === "NEUTRAL" ? "50" : "65"}%

Key Levels:
• Invalidation: ${preliminaryBias === "BEARISH" ? recentHigh.toFixed(4) : recentLow.toFixed(4)}
• Target 1: ${preliminaryBias === "BEARISH" ? recentLow.toFixed(4) : recentHigh.toFixed(4)}
• Target 2: ${preliminaryBias === "BEARISH" ? (recentLow - 0.0030).toFixed(4) : (recentHigh + 0.0030).toFixed(4)}

Watch for: Price action at ${sma20.toFixed(4)} (20 SMA)

Updates at market close.`;

    const fallbackEvening = `EUR/USD Recap - ${date}

Bias: ${preliminaryBias} - See results below
High: ${todayHigh.toFixed(4)}
Low: ${todayLow.toFixed(4)}
Close: ${latestPrice.toFixed(4)}

Tomorrow: Watching for continuation of trend.

Follow for daily EUR/USD analysis.`;

    const post = aiContent || (type === "morning" ? fallbackMorning : fallbackEvening);

    console.log(`${type} bias generated successfully`);

    // Return price data for chart (last 48 hours)
    const chartData = prices.slice(0, 48).reverse().map((p) => ({
      timestamp: p.timestamp,
      open: p.open,
      high: p.high,
      low: p.low,
      close: p.close,
    }));

    return new Response(
      JSON.stringify({
        post,
        direction: preliminaryBias,
        levels: {
          current: latestPrice,
          resistance: recentHigh,
          support: recentLow,
          sma20,
          sma50,
        },
        priceData: chartData,
        keyLevels: {
          high: todayHigh,
          low: todayLow,
          resistance: recentHigh,
          support: recentLow,
        },
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error in generate-daily-bias:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return new Response(
      JSON.stringify({ error: errorMessage }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
