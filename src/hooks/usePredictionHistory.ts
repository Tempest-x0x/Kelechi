import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Prediction } from '@/types/trading';

export function usePredictionHistory() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setIsLoading(true);
    try {
      const { data, error: queryError } = await supabase
        .from('predictions')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

      if (queryError) throw queryError;

      // Transform database rows to Prediction type
      const transformed = (data || []).map((row: any) => ({
        id: row.id,
        created_at: row.created_at,
        signal_type: row.signal_type as 'BUY' | 'SELL',
        confidence: Number(row.confidence),
        entry_price: Number(row.entry_price),
        take_profit_1: row.take_profit_1 ? Number(row.take_profit_1) : undefined,
        take_profit_2: row.take_profit_2 ? Number(row.take_profit_2) : undefined,
        stop_loss: row.stop_loss ? Number(row.stop_loss) : undefined,
        current_price_at_prediction: Number(row.current_price_at_prediction),
        trend_direction: row.trend_direction as 'BULLISH' | 'BEARISH' | 'NEUTRAL',
        trend_strength: Number(row.trend_strength),
        reasoning: row.reasoning,
        technical_indicators: row.technical_indicators,
        patterns_detected: row.patterns_detected,
        sentiment_score: row.sentiment_score ? Number(row.sentiment_score) : undefined,
        outcome: row.outcome as 'WIN' | 'LOSS' | 'PENDING' | 'EXPIRED',
        outcome_price: row.outcome_price ? Number(row.outcome_price) : undefined,
        outcome_at: row.outcome_at,
        expires_at: row.expires_at,
      }));

      setPredictions(transformed);
    } catch (err) {
      console.error('Error fetching prediction history:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch history');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Subscribe to realtime updates
  useEffect(() => {
    const channel = supabase
      .channel('predictions-changes')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'predictions' },
        (payload) => {
          const newPrediction = payload.new as any;
          const transformed: Prediction = {
            id: newPrediction.id,
            created_at: newPrediction.created_at,
            signal_type: newPrediction.signal_type as 'BUY' | 'SELL',
            confidence: Number(newPrediction.confidence),
            entry_price: Number(newPrediction.entry_price),
            take_profit_1: newPrediction.take_profit_1 ? Number(newPrediction.take_profit_1) : undefined,
            take_profit_2: newPrediction.take_profit_2 ? Number(newPrediction.take_profit_2) : undefined,
            stop_loss: newPrediction.stop_loss ? Number(newPrediction.stop_loss) : undefined,
            current_price_at_prediction: Number(newPrediction.current_price_at_prediction),
            trend_direction: newPrediction.trend_direction,
            trend_strength: Number(newPrediction.trend_strength),
            reasoning: newPrediction.reasoning,
            technical_indicators: newPrediction.technical_indicators,
            patterns_detected: newPrediction.patterns_detected,
            sentiment_score: newPrediction.sentiment_score ? Number(newPrediction.sentiment_score) : undefined,
            outcome: newPrediction.outcome,
            outcome_price: newPrediction.outcome_price ? Number(newPrediction.outcome_price) : undefined,
            outcome_at: newPrediction.outcome_at,
            expires_at: newPrediction.expires_at,
          };
          setPredictions((prev) => [transformed, ...prev.slice(0, 49)]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  return { predictions, isLoading, error, refetch: fetchHistory };
}
