import { useState } from 'react';
import { supabase } from '@/integrations/supabase/client';

export interface BacktestTrade {
  entryTime: string;
  exitTime: string;
  signalType: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  stopLoss: number;
  takeProfit: number;
  outcome: 'WIN' | 'LOSS';
  pips: number;
  confidence: number;
  patterns: string[];
}

export interface BacktestResult {
  period: { start: string; end: string };
  totalSignals: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPips: number;
  avgPipsPerTrade: number;
  profitFactor: number;
  maxDrawdownPips: number;
  patternPerformance: Record<string, { wins: number; losses: number; winRate: number }>;
  trades: BacktestTrade[];
}

export interface BacktestParams {
  startDate?: string;
  endDate?: string;
  minConfidence?: number;
  maxTrades?: number;
}

export function useBacktest() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = async (params: BacktestParams = {}) => {
    try {
      setIsLoading(true);
      setError(null);

      const { data, error: invokeError } = await supabase.functions.invoke('backtest-strategy', {
        body: params
      });

      if (invokeError) throw invokeError;
      
      if (!data.success) {
        throw new Error(data.error || 'Backtest failed');
      }

      setResult(data.result);
      return data.result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Backtest failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const clearResult = () => {
    setResult(null);
    setError(null);
  };

  return {
    result,
    isLoading,
    error,
    runBacktest,
    clearResult
  };
}
