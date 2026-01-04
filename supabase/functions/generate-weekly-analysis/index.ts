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
    const { startDate, endDate } = await req.json();
    console.log(`Generating weekly analysis for ${startDate} to ${endDate}`);

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Fetch predictions for the date range
    const { data: predictions, error: predError } = await supabase
      .from("predictions")
      .select("*")
      .gte("created_at", `${startDate}T00:00:00Z`)
      .lte("created_at", `${endDate}T23:59:59Z`)
      .order("created_at", { ascending: true });

    if (predError) {
      console.error("Error fetching predictions:", predError);
      throw predError;
    }

    // Fetch learnings for context
    const { data: learnings, error: learnError } = await supabase
      .from("prediction_learnings")
      .select("*")
      .gte("created_at", `${startDate}T00:00:00Z`)
      .lte("created_at", `${endDate}T23:59:59Z`)
      .order("created_at", { ascending: true });

    if (learnError) {
      console.error("Error fetching learnings:", learnError);
    }

    // Fetch latest price data for next week outlook
    const { data: priceData, error: priceError } = await supabase
      .from("price_history")
      .select("*")
      .order("timestamp", { ascending: false })
      .limit(100);

    if (priceError) {
      console.error("Error fetching price data:", priceError);
    }

    // Calculate statistics
    const totalPredictions = predictions?.length || 0;
    const wins = predictions?.filter((p) => p.outcome === "WIN").length || 0;
    const losses = predictions?.filter((p) => p.outcome === "LOSS").length || 0;
    const pending = predictions?.filter((p) => !p.outcome).length || 0;
    const winRate = totalPredictions > 0 ? Math.round((wins / (wins + losses || 1)) * 100) : 0;

    // Group by signal type
    const buySignals = predictions?.filter((p) => p.signal_type === "BUY") || [];
    const sellSignals = predictions?.filter((p) => p.signal_type === "SELL") || [];
    const buyWins = buySignals.filter((p) => p.outcome === "WIN").length;
    const sellWins = sellSignals.filter((p) => p.outcome === "WIN").length;

    // Extract key levels from predictions
    const entryPrices = predictions?.map((p) => p.entry_price).filter(Boolean) || [];
    const highestEntry = entryPrices.length > 0 ? Math.max(...entryPrices) : null;
    const lowestEntry = entryPrices.length > 0 ? Math.min(...entryPrices) : null;

    // Get latest price for context
    const latestPrice = priceData?.[0]?.close || null;

    // Build context for AI
    const analysisContext = {
      dateRange: { start: startDate, end: endDate },
      stats: { totalPredictions, wins, losses, pending, winRate },
      byDirection: {
        buys: { total: buySignals.length, wins: buyWins },
        sells: { total: sellSignals.length, wins: sellWins },
      },
      keyLevels: { highest: highestEntry, lowest: lowestEntry, current: latestPrice },
      predictions: predictions?.slice(0, 20).map((p) => ({
        date: p.created_at,
        signal: p.signal_type,
        confidence: p.confidence,
        entry: p.entry_price,
        outcome: p.outcome,
        reasoning: p.reasoning,
      })),
      learnings: learnings?.slice(0, 10).map((l) => ({
        lesson: l.lesson_extracted,
        factors: l.success_factors,
        failure: l.failure_reason,
      })),
    };

    // Generate AI analysis
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    // Add timeout to prevent hanging
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    let aiContent = "";
    try {
      const aiResponse = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
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
              content: `You are a professional forex analyst generating weekly post-mortem content for X (Twitter). 
Your analysis should be:
- Concise and actionable
- Focus on EUR/USD
- Include specific price levels
- Be honest about what worked and failed
- Provide clear scenarios for the week ahead

Generate THREE separate outputs:
1. WEEKLY_REVIEW: A tweet-length (280 chars max) summary of last week
2. WEEK_AHEAD: A longer post (up to 800 chars) with next week's outlook
3. DETAILED: A full internal analysis for reference

Format each section clearly with the headers:
---WEEKLY_REVIEW---
---WEEK_AHEAD---
---DETAILED---`,
            },
            {
              role: "user",
              content: `Generate weekly EUR/USD post-mortem analysis based on this data:

${JSON.stringify(analysisContext, null, 2)}

Create content for X posts that:
1. Reviews performance (${wins}W/${losses}L, ${winRate}% win rate)
2. Highlights key lessons learned
3. Provides next week's directional bias
4. Lists key levels to watch
5. Includes both bullish and bearish scenarios`,
            },
          ],
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle rate limits
      if (aiResponse.status === 429 || aiResponse.status === 402) {
        console.warn("AI rate limited, using fallback content");
      } else if (!aiResponse.ok) {
        const errorText = await aiResponse.text();
        console.error("AI API error:", aiResponse.status, errorText);
      } else {
        // Read as text first for safer handling
        const responseText = await aiResponse.text();
        console.log("AI response length:", responseText.length);

        if (responseText && responseText.length > 0) {
          try {
            const aiData = JSON.parse(responseText);
            aiContent = aiData.choices?.[0]?.message?.content || "";
          } catch (parseError) {
            console.error("Failed to parse AI response:", parseError);
            console.error("Raw response (first 500 chars):", responseText.substring(0, 500));
          }
        }
      }
    } catch (fetchError) {
      clearTimeout(timeoutId);
      console.error("AI fetch error:", fetchError);
      // Continue with fallback content
    }

    // Parse the AI response into sections
    const weeklyReviewMatch = aiContent.match(/---WEEKLY_REVIEW---\s*([\s\S]*?)(?=---WEEK_AHEAD---|$)/);
    const weekAheadMatch = aiContent.match(/---WEEK_AHEAD---\s*([\s\S]*?)(?=---DETAILED---|$)/);
    const detailedMatch = aiContent.match(/---DETAILED---\s*([\s\S]*?)$/);

    const weeklyReview = weeklyReviewMatch?.[1]?.trim() || `EUR/USD Week Review (${startDate} - ${endDate})

${wins}W/${losses}L (${winRate}% WR)

${wins > losses ? "Trend-following signals performed well." : "Mixed results. Refining approach."}

Follow for daily EUR/USD bias.`;

    const weekAheadOutlook = weekAheadMatch?.[1]?.trim() || `EUR/USD — Week Ahead Outlook

HTF Bias: ${sellSignals.length > buySignals.length ? "BEARISH" : "BULLISH"}
Key Levels: ${highestEntry?.toFixed(4) || "TBD"} / ${lowestEntry?.toFixed(4) || "TBD"}

${sellSignals.length > buySignals.length ? "Bearish" : "Bullish"} Scenario: Continuation of trend
${sellSignals.length > buySignals.length ? "Bullish" : "Bearish"} Scenario: Break of key level

I publish daily bias + outcomes. Follow if you trade EUR/USD.`;

    const detailedAnalysis = detailedMatch?.[1]?.trim() || `DETAILED WEEKLY ANALYSIS

Period: ${startDate} to ${endDate}

PERFORMANCE SUMMARY
- Total Predictions: ${totalPredictions}
- Wins: ${wins}
- Losses: ${losses}
- Pending: ${pending}
- Win Rate: ${winRate}%

BY DIRECTION
- BUY signals: ${buySignals.length} (${buyWins} wins)
- SELL signals: ${sellSignals.length} (${sellWins} wins)

KEY LEVELS TRADED
- Highest Entry: ${highestEntry?.toFixed(4) || "N/A"}
- Lowest Entry: ${lowestEntry?.toFixed(4) || "N/A"}
- Current Price: ${latestPrice?.toFixed(4) || "N/A"}

LEARNINGS
${learnings?.map((l) => `- ${l.lesson_extracted || "N/A"}`).join("\n") || "No learnings recorded this week."}`;

    console.log("Weekly analysis generated successfully");

    return new Response(
      JSON.stringify({
        weeklyReview,
        weekAheadOutlook,
        detailedAnalysis,
        stats: analysisContext.stats,
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error in generate-weekly-analysis:", error);
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
