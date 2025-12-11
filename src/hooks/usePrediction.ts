import { useState, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Prediction, Candle, Timeframe } from '@/types/trading';

export function usePrediction() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generatePrediction = useCallback(async (
    candles: Candle[], 
    timeframe: Timeframe,
    sentimentScore: number | null
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const currentPrice = candles[0]?.close;
      const { data: result, error: fnError } = await supabase.functions.invoke('generate-prediction', {
        body: { candles, currentPrice, timeframe, sentimentScore },
      });

      if (fnError) throw fnError;

      if (!result.success) {
        throw new Error(result.error || 'Failed to generate prediction');
      }

      setPrediction(result.prediction);
      return result.prediction;
    } catch (err) {
      console.error('Error generating prediction:', err);
      const message = err instanceof Error ? err.message : 'Failed to generate prediction';
      setError(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { prediction, isLoading, error, generatePrediction };
}
