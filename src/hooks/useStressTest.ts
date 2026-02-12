/**
 * Hook for Monte Carlo Stress Test
 * 
 * Provides access to the stress test functionality with loading states.
 */

import { useState, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';

export interface MonteCarloResult {
  probabilityOfRuin: number;
  drawdown95CI: { lower: number; upper: number };
  finalEquity95CI: { lower: number; upper: number };
  medianFinalEquity: number;
  worstCaseEquity: number;
  bestCaseEquity: number;
  simulationsRun: number;
  tradesAnalyzed: number;
  avgReturn: number;
  volatility: number;
  interpretation: {
    probabilityOfRuin: string;
    maxDrawdown: string;
    finalEquity: string;
    expectedOutcome: string;
  };
  tradeStats: {
    totalTrades: number;
    winRate: string;
    avgReturnPips: string;
    volatilityPips: string;
  };
  riskAssessment: {
    level: 'CRITICAL' | 'WARNING' | 'OK';
    message: string;
    action: string;
  }[];
}

export interface StressTestParams {
  tradeCount?: number;
  simulations?: number;
  startingCapital?: number;
  symbol?: string;
}

export function useStressTest() {
  const [result, setResult] = useState<MonteCarloResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runStressTest = useCallback(async (params: StressTestParams = {}) => {
    setIsLoading(true);
    setError(null);

    try {
      const { data, error: invokeError } = await supabase.functions.invoke('stress-test', {
        body: params
      });

      if (invokeError) throw invokeError;
      
      if (!data.success) {
        throw new Error(data.error || 'Stress test failed');
      }

      setResult(data.result);
      return data.result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Stress test failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    result,
    isLoading,
    error,
    runStressTest,
    clearResult
  };
}
