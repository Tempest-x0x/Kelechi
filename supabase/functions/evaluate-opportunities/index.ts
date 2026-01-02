import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface Opportunity {
  id: string;
  signal_type: string;
  entry_price: number;
  stop_loss: number | null;
  take_profit_1: number | null;
  take_profit_2: number | null;
  patterns_detected: string[] | null;
  technical_indicators: Record<string, any> | null;
  reasoning: string | null;
  confidence: number;
  created_at: string;
  expires_at: string;
}

interface PricePoint {
  timestamp: string;
  high: number;
  low: number;
  close: number;
}

// Determine outcome by checking if SL or TP was hit
function evaluateOutcome(
  opportunity: Opportunity,
  priceHistory: PricePoint[]
): { outcome: 'WIN' | 'LOSS' | 'EXPIRED'; outcomePrice: number; outcomeAt: string } {
  const { signal_type, entry_price, stop_loss, take_profit_1 } = opportunity;
  
  for (const point of priceHistory) {
    if (signal_type === 'BUY') {
      // Check if stop loss was hit first
      if (stop_loss && point.low <= stop_loss) {
        return { outcome: 'LOSS', outcomePrice: stop_loss, outcomeAt: point.timestamp };
      }
      // Check if take profit was hit
      if (take_profit_1 && point.high >= take_profit_1) {
        return { outcome: 'WIN', outcomePrice: take_profit_1, outcomeAt: point.timestamp };
      }
    } else if (signal_type === 'SELL') {
      // Check if stop loss was hit first
      if (stop_loss && point.high >= stop_loss) {
        return { outcome: 'LOSS', outcomePrice: stop_loss, outcomeAt: point.timestamp };
      }
      // Check if take profit was hit
      if (take_profit_1 && point.low <= take_profit_1) {
        return { outcome: 'WIN', outcomePrice: take_profit_1, outcomeAt: point.timestamp };
      }
    }
  }
  
  // If neither SL nor TP was hit, it expired
  const lastPrice = priceHistory[priceHistory.length - 1]?.close || entry_price;
  return { 
    outcome: 'EXPIRED', 
    outcomePrice: lastPrice, 
    outcomeAt: new Date().toISOString() 
  };
}

// Generate AI learning from the outcome
async function generateLearning(
  opportunity: Opportunity,
  outcome: 'WIN' | 'LOSS' | 'EXPIRED',
  outcomePrice: number,
  lovableApiKey: string
): Promise<{ lesson: string; successFactors: string | null; failureReason: string | null }> {
  const prompt = `Analyze this forex trading opportunity and its outcome to extract learning insights.

OPPORTUNITY DETAILS:
- Signal: ${opportunity.signal_type} at ${opportunity.entry_price.toFixed(5)}
- Stop Loss: ${opportunity.stop_loss?.toFixed(5) || 'Not set'}
- Take Profit: ${opportunity.take_profit_1?.toFixed(5) || 'Not set'}
- Confidence: ${opportunity.confidence.toFixed(0)}%
- Patterns Detected: ${JSON.stringify(opportunity.patterns_detected || [])}
- Technical Indicators: RSI=${opportunity.technical_indicators?.rsi?.toFixed(1) || 'N/A'}, MACD Histogram=${opportunity.technical_indicators?.macd?.histogram?.toFixed(5) || 'N/A'}

OUTCOME:
- Result: ${outcome}
- Outcome Price: ${outcomePrice.toFixed(5)}
- Original Reasoning: ${opportunity.reasoning || 'None provided'}

Based on this ${outcome === 'WIN' ? 'successful' : 'unsuccessful'} trade, provide:
1. A key lesson learned (1-2 sentences)
2. ${outcome === 'WIN' ? 'What factors contributed to success' : 'What went wrong and why'}
3. How to improve future signals with similar setups

Keep each response under 100 words. Be specific and actionable.`;

  try {
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${lovableApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are a forex trading analyst. Provide concise, actionable insights from trade outcomes." },
          { role: "user", content: prompt }
        ],
      }),
    });

    if (!response.ok) {
      console.error("AI API error:", response.status);
      return {
        lesson: `${outcome} trade - ${opportunity.signal_type} signal at ${opportunity.confidence.toFixed(0)}% confidence.`,
        successFactors: outcome === 'WIN' ? 'Technical indicators aligned correctly' : null,
        failureReason: outcome !== 'WIN' ? 'Market moved against the position' : null,
      };
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';
    
    // Parse the AI response
    const lines = content.split('\n').filter((l: string) => l.trim());
    const lesson = lines[0] || `${outcome} trade analyzed`;
    
    return {
      lesson: lesson.slice(0, 500),
      successFactors: outcome === 'WIN' ? (lines.slice(1).join(' ').slice(0, 500) || 'Indicators aligned correctly') : null,
      failureReason: outcome !== 'WIN' ? (lines.slice(1).join(' ').slice(0, 500) || 'Market conditions changed') : null,
    };
  } catch (error) {
    console.error("Learning generation error:", error);
    return {
      lesson: `${outcome}: ${opportunity.signal_type} at ${opportunity.entry_price.toFixed(5)}`,
      successFactors: outcome === 'WIN' ? 'Signal executed as planned' : null,
      failureReason: outcome !== 'WIN' ? 'Trade did not reach target' : null,
    };
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    console.log("Starting opportunity evaluation...");

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const lovableApiKey = Deno.env.get("LOVABLE_API_KEY");
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Find expired opportunities that haven't been evaluated
    const { data: pendingOpps, error: fetchError } = await supabase
      .from('trading_opportunities')
      .select('*')
      .is('outcome', null)
      .lt('expires_at', new Date().toISOString())
      .limit(10);

    if (fetchError) {
      console.error("Fetch error:", fetchError);
      throw new Error("Failed to fetch pending opportunities");
    }

    if (!pendingOpps || pendingOpps.length === 0) {
      console.log("No pending opportunities to evaluate");
      return new Response(
        JSON.stringify({ success: true, message: "No pending opportunities", evaluated: 0 }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`Found ${pendingOpps.length} opportunities to evaluate`);

    const results: any[] = [];

    for (const opp of pendingOpps) {
      console.log(`Evaluating opportunity ${opp.id}...`);

      // Fetch price history since the opportunity was created
      const { data: priceHistory, error: priceError } = await supabase
        .from('price_history')
        .select('timestamp, high, low, close')
        .eq('symbol', 'EUR/USD')
        .eq('timeframe', '1h')
        .gte('timestamp', opp.created_at)
        .order('timestamp', { ascending: true });

      if (priceError || !priceHistory || priceHistory.length === 0) {
        console.log(`No price history for opportunity ${opp.id}`);
        continue;
      }

      // Evaluate outcome
      const { outcome, outcomePrice, outcomeAt } = evaluateOutcome(
        opp as Opportunity,
        priceHistory.map(p => ({
          timestamp: p.timestamp,
          high: Number(p.high),
          low: Number(p.low),
          close: Number(p.close)
        }))
      );

      console.log(`Opportunity ${opp.id}: ${outcome} at ${outcomePrice}`);

      // Generate learning using AI
      let learning = { lesson: '', successFactors: null as string | null, failureReason: null as string | null };
      
      if (lovableApiKey) {
        learning = await generateLearning(opp as Opportunity, outcome, outcomePrice, lovableApiKey);
      } else {
        learning = {
          lesson: `${outcome}: ${opp.signal_type} signal ${outcome === 'WIN' ? 'reached target' : 'did not perform as expected'}`,
          successFactors: outcome === 'WIN' ? 'Technical alignment was correct' : null,
          failureReason: outcome !== 'WIN' ? 'Market conditions changed' : null,
        };
      }

      // Store learning
      const { data: newLearning, error: learningError } = await supabase
        .from('prediction_learnings')
        .insert({
          opportunity_id: opp.id,
          lesson_extracted: learning.lesson,
          success_factors: learning.successFactors,
          failure_reason: learning.failureReason,
          pattern_context: {
            patterns: opp.patterns_detected || [],
            indicators: opp.technical_indicators || {},
            signal_type: opp.signal_type,
            confidence: opp.confidence
          },
          market_conditions: {
            entry_price: opp.entry_price,
            outcome_price: outcomePrice,
            trend_direction: opp.signal_type,
            trend_strength: opp.confidence,
            sentiment_score: 0
          }
        })
        .select()
        .single();

      if (learningError) {
        console.error("Failed to store learning:", learningError);
      }

      // Update opportunity
      const { error: updateError } = await supabase
        .from('trading_opportunities')
        .update({
          outcome,
          status: outcome === 'WIN' ? 'COMPLETED' : 'CLOSED',
          evaluated_at: new Date().toISOString(),
          ai_learning_id: newLearning?.id || null
        })
        .eq('id', opp.id);

      if (updateError) {
        console.error("Failed to update opportunity:", updateError);
      }

      results.push({
        id: opp.id,
        outcome,
        outcomePrice,
        learning: learning.lesson
      });
    }

    console.log(`Evaluated ${results.length} opportunities`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: `Evaluated ${results.length} opportunities`,
        evaluated: results.length,
        results 
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Evaluation error:", error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
