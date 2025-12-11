import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { ForexData, Timeframe } from '@/types/trading';

export function useForexData(timeframe: Timeframe = '1h') {
  const [data, setData] = useState<ForexData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const { data: result, error: fnError } = await supabase.functions.invoke('fetch-forex-data', {
        body: { timeframe, outputsize: 300 },
      });

      if (fnError) throw fnError;

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch forex data');
      }

      setData({
        symbol: result.symbol,
        currentPrice: result.currentPrice,
        candles: result.candles,
        meta: result.meta,
      });
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Error fetching forex data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, [timeframe]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData, lastUpdated };
}
