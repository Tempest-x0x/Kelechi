import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';

export interface CurrencyPair {
  id: string;
  symbol: string;
  display_name: string;
  pip_value: number;
  is_active: boolean;
  has_pattern_stats: boolean;
  created_at: string;
  updated_at: string;
}

export function useCurrencyPairs() {
  const [pairs, setPairs] = useState<CurrencyPair[]>([]);
  const [activePairs, setActivePairs] = useState<CurrencyPair[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPairs = useCallback(async () => {
    try {
      const { data, error: fetchError } = await supabase
        .from('supported_currency_pairs')
        .select('*')
        .order('symbol', { ascending: true });

      if (fetchError) throw fetchError;

      const allPairs = (data || []) as unknown as CurrencyPair[];
      setPairs(allPairs);
      setActivePairs(allPairs.filter(p => p.is_active));
      setError(null);
    } catch (err) {
      console.error('Failed to fetch currency pairs:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch currency pairs');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const togglePairActive = useCallback(async (id: string, isActive: boolean) => {
    const { error: updateError } = await supabase
      .from('supported_currency_pairs')
      .update({ is_active: isActive })
      .eq('id', id);

    if (updateError) {
      throw updateError;
    }

    await fetchPairs();
  }, [fetchPairs]);

  const addPair = useCallback(async (pair: Omit<CurrencyPair, 'id' | 'created_at' | 'updated_at'>) => {
    const { error: insertError } = await supabase
      .from('supported_currency_pairs')
      .insert(pair);

    if (insertError) {
      throw insertError;
    }

    await fetchPairs();
  }, [fetchPairs]);

  const updatePairStats = useCallback(async (id: string, hasPatternStats: boolean) => {
    const { error: updateError } = await supabase
      .from('supported_currency_pairs')
      .update({ has_pattern_stats: hasPatternStats })
      .eq('id', id);

    if (updateError) {
      throw updateError;
    }

    await fetchPairs();
  }, [fetchPairs]);

  useEffect(() => {
    fetchPairs();
  }, [fetchPairs]);

  return {
    pairs,
    activePairs,
    isLoading,
    error,
    refetch: fetchPairs,
    togglePairActive,
    addPair,
    updatePairStats
  };
}
