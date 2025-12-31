import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';

export interface TradingOpportunity {
  id: string;
  created_at: string;
  expires_at: string;
  signal_type: 'BUY' | 'SELL';
  confidence: number;
  entry_price: number;
  current_price: number;
  stop_loss: number | null;
  take_profit_1: number | null;
  take_profit_2: number | null;
  patterns_detected: string[] | null;
  technical_indicators: any;
  pattern_stats: any[] | null;
  reasoning: string | null;
  status: 'ACTIVE' | 'EXPIRED' | 'TRIGGERED';
  triggered_at: string | null;
  outcome: 'WIN' | 'LOSS' | 'PENDING' | 'EXPIRED' | null;
}

interface ScanResult {
  success: boolean;
  message: string;
  scanned: boolean;
  opportunity?: TradingOpportunity;
}

export function useOpportunities() {
  const [opportunities, setOpportunities] = useState<TradingOpportunity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastScanned, setLastScanned] = useState<Date | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  const fetchOpportunities = useCallback(async () => {
    try {
      const { data, error: fetchError } = await supabase
        .from('trading_opportunities')
        .select('*')
        .eq('status', 'ACTIVE')
        .order('created_at', { ascending: false });

      if (fetchError) throw fetchError;

      // Type assertion since we know the structure
      setOpportunities((data || []) as unknown as TradingOpportunity[]);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch opportunities:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch opportunities');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const triggerScan = useCallback(async (): Promise<ScanResult> => {
    setIsScanning(true);
    try {
      const { data, error: scanError } = await supabase.functions.invoke('scan-opportunities');

      if (scanError) throw scanError;

      setLastScanned(new Date());
      
      // Refetch opportunities after scan
      await fetchOpportunities();

      return data as ScanResult;
    } catch (err) {
      console.error('Scan failed:', err);
      throw err;
    } finally {
      setIsScanning(false);
    }
  }, [fetchOpportunities]);

  // Initial fetch
  useEffect(() => {
    fetchOpportunities();
  }, [fetchOpportunities]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      fetchOpportunities();
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, [fetchOpportunities]);

  // Subscribe to realtime updates
  useEffect(() => {
    const channel = supabase
      .channel('opportunities-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'trading_opportunities'
        },
        () => {
          fetchOpportunities();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [fetchOpportunities]);

  return {
    opportunities,
    isLoading,
    error,
    lastScanned,
    isScanning,
    triggerScan,
    refetch: fetchOpportunities
  };
}
