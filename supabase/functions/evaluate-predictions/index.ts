import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { currentPrice } = await req.json();

    if (!currentPrice || typeof currentPrice !== 'number') {
      throw new Error('Current price is required');
    }

    console.log(`Evaluating predictions against current price: ${currentPrice}`);

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Fetch pending predictions (no outcome yet)
    const { data: pendingPredictions, error: fetchError } = await supabase
      .from('predictions')
      .select('*')
      .is('outcome', null)
      .order('created_at', { ascending: false });

    if (fetchError) {
      throw new Error(`Failed to fetch predictions: ${fetchError.message}`);
    }

    console.log(`Found ${pendingPredictions?.length || 0} pending predictions to evaluate`);

    const evaluatedPredictions: any[] = [];
    const learnings: any[] = [];
    const now = new Date();

    for (const prediction of pendingPredictions || []) {
      const expiresAt = new Date(prediction.expires_at);
      const isExpired = now > expiresAt;
      
      let outcome: 'WIN' | 'LOSS' | null = null;
      let failureReason: string | null = null;
      let successFactors: string | null = null;

      if (prediction.signal_type === 'BUY') {
        // Check if TP1 or TP2 was hit (WIN)
        if (currentPrice >= prediction.take_profit_1) {
          outcome = 'WIN';
          successFactors = currentPrice >= (prediction.take_profit_2 || prediction.take_profit_1) 
            ? 'Hit TP2 - Strong momentum continuation' 
            : 'Hit TP1 - Primary target reached';
        } 
        // Check if SL was hit (LOSS)
        else if (currentPrice <= prediction.stop_loss) {
          outcome = 'LOSS';
          failureReason = 'Stop loss hit - Price reversed against position';
        }
        // Check if expired without hitting targets
        else if (isExpired) {
          // Determine if it was a partial win (moved in right direction) or loss
          if (currentPrice > prediction.entry_price) {
            outcome = 'WIN';
            successFactors = 'Expired in profit - Partial target achieved';
          } else {
            outcome = 'LOSS';
            failureReason = 'Expired below entry - Momentum failed to materialize';
          }
        }
      } else if (prediction.signal_type === 'SELL') {
        // Check if TP1 or TP2 was hit (WIN)
        if (currentPrice <= prediction.take_profit_1) {
          outcome = 'WIN';
          successFactors = currentPrice <= (prediction.take_profit_2 || prediction.take_profit_1)
            ? 'Hit TP2 - Strong bearish continuation'
            : 'Hit TP1 - Primary target reached';
        }
        // Check if SL was hit (LOSS)
        else if (currentPrice >= prediction.stop_loss) {
          outcome = 'LOSS';
          failureReason = 'Stop loss hit - Price reversed against position';
        }
        // Check if expired without hitting targets
        else if (isExpired) {
          if (currentPrice < prediction.entry_price) {
            outcome = 'WIN';
            successFactors = 'Expired in profit - Partial target achieved';
          } else {
            outcome = 'LOSS';
            failureReason = 'Expired above entry - Bearish momentum failed';
          }
        }
      } else if (prediction.signal_type === 'HOLD') {
        // HOLD signals are marked as WIN if price stayed relatively stable
        if (isExpired) {
          const priceChange = Math.abs(currentPrice - prediction.entry_price) / prediction.entry_price;
          if (priceChange < 0.005) { // Less than 0.5% movement
            outcome = 'WIN';
            successFactors = 'Correctly identified consolidation period';
          } else {
            outcome = 'LOSS';
            failureReason = 'Missed a significant price movement during hold period';
          }
        }
      }

      // Update prediction if outcome determined
      if (outcome) {
        const { error: updateError } = await supabase
          .from('predictions')
          .update({
            outcome,
            outcome_price: currentPrice,
            outcome_at: now.toISOString()
          })
          .eq('id', prediction.id);

        if (updateError) {
          console.error(`Failed to update prediction ${prediction.id}:`, updateError);
          continue;
        }

        evaluatedPredictions.push({
          id: prediction.id,
          signal_type: prediction.signal_type,
          outcome,
          entry_price: prediction.entry_price,
          outcome_price: currentPrice
        });

        // Extract learning from this prediction
        const patternContext = {
          patterns: prediction.patterns_detected,
          indicators: prediction.technical_indicators,
          signal_type: prediction.signal_type,
          confidence: prediction.confidence
        };

        const marketConditions = {
          entry_price: prediction.entry_price,
          outcome_price: currentPrice,
          trend_direction: prediction.trend_direction,
          trend_strength: prediction.trend_strength,
          sentiment_score: prediction.sentiment_score
        };

        let lessonExtracted = '';
        if (outcome === 'LOSS') {
          // Analyze why the trade failed
          const indicators = prediction.technical_indicators as any;
          
          if (prediction.signal_type === 'BUY' && indicators?.rsi > 70) {
            lessonExtracted = 'Avoid BUY signals when RSI is in overbought territory (>70)';
          } else if (prediction.signal_type === 'SELL' && indicators?.rsi < 30) {
            lessonExtracted = 'Avoid SELL signals when RSI is in oversold territory (<30)';
          } else if (prediction.signal_type === 'BUY' && currentPrice < indicators?.ema50) {
            lessonExtracted = 'BUY signal failed when price was below EMA50 - wait for price to reclaim EMA';
          } else if (prediction.signal_type === 'SELL' && currentPrice > indicators?.ema50) {
            lessonExtracted = 'SELL signal failed when price was above EMA50 - wait for breakdown';
          } else {
            lessonExtracted = `${prediction.signal_type} signal at ${prediction.entry_price} failed. Review pattern reliability.`;
          }
        } else {
          lessonExtracted = `${prediction.signal_type} signal succeeded. Pattern: ${(prediction.patterns_detected as string[])?.join(', ') || 'N/A'}`;
        }

        // Store learning
        const { error: learningError } = await supabase
          .from('prediction_learnings')
          .insert({
            prediction_id: prediction.id,
            pattern_context: patternContext,
            market_conditions: marketConditions,
            lesson_extracted: lessonExtracted,
            failure_reason: failureReason,
            success_factors: successFactors
          });

        if (learningError) {
          console.error(`Failed to store learning for ${prediction.id}:`, learningError);
        } else {
          learnings.push({
            prediction_id: prediction.id,
            lesson: lessonExtracted
          });
        }
      }
    }

    console.log(`Evaluated ${evaluatedPredictions.length} predictions, extracted ${learnings.length} learnings`);

    return new Response(JSON.stringify({
      success: true,
      evaluated: evaluatedPredictions.length,
      results: evaluatedPredictions,
      learnings
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error evaluating predictions:', error);
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});
