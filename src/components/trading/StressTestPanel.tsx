/**
 * Monte Carlo Stress Test Panel
 * 
 * Displays results from Monte Carlo simulation:
 * - Probability of Ruin
 * - 95% Confidence Interval for drawdown
 * - Final equity distribution
 * - Risk assessment
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useStressTest, MonteCarloResult } from '@/hooks/useStressTest';
import {
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Loader2,
  PlayCircle,
  BarChart2,
} from 'lucide-react';

export function StressTestPanel() {
  const { result, isLoading, error, runStressTest } = useStressTest();
  const [hasRun, setHasRun] = useState(false);

  const handleRunTest = async () => {
    try {
      await runStressTest({
        tradeCount: 100,
        simulations: 10000,
        startingCapital: 10000,
      });
      setHasRun(true);
    } catch (err) {
      console.error('Stress test failed:', err);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'CRITICAL': return 'destructive';
      case 'WARNING': return 'secondary';
      default: return 'default';
    }
  };

  const getRuinColor = (probability: number) => {
    if (probability < 1) return 'text-green-500';
    if (probability < 5) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart2 className="h-5 w-5 text-primary" />
            Monte Carlo Stress Test
          </CardTitle>
          <Button
            size="sm"
            onClick={handleRunTest}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4 mr-2" />
                Run Test
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="flex items-center gap-2 text-sm text-red-500 bg-red-500/10 p-2 rounded mb-4">
            <AlertTriangle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        {!hasRun && !isLoading && (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">
              Run a Monte Carlo simulation to stress test your trading system.
            </p>
            <p className="text-xs mt-2">
              Simulates 10,000 different equity paths using your last 100 trades.
            </p>
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-4">
              {/* Probability of Ruin */}
              <div className="bg-muted/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Probability of Ruin</span>
                  {result.probabilityOfRuin < 5 ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                  )}
                </div>
                <div className={`text-2xl font-bold ${getRuinColor(result.probabilityOfRuin)}`}>
                  {result.probabilityOfRuin.toFixed(2)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Chance of account hitting zero
                </div>
              </div>

              {/* Max Drawdown 95% CI */}
              <div className="bg-muted/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Max Drawdown (95% CI)</span>
                  <TrendingDown className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="text-2xl font-bold">
                  {result.drawdown95CI.lower.toFixed(1)}% - {result.drawdown95CI.upper.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Expected range of max drawdown
                </div>
              </div>
            </div>

            {/* Final Equity Distribution */}
            <div className="bg-muted/50 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Final Equity Distribution</span>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-lg font-semibold text-red-400">
                    ${result.worstCaseEquity.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground">Worst Case</div>
                </div>
                <div>
                  <div className="text-lg font-semibold text-primary">
                    ${result.medianFinalEquity.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground">Median</div>
                </div>
                <div>
                  <div className="text-lg font-semibold text-green-400">
                    ${result.bestCaseEquity.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground">Best Case</div>
                </div>
              </div>
              <div className="mt-2 text-xs text-center text-muted-foreground">
                95% CI: ${result.finalEquity95CI.lower.toFixed(0)} - ${result.finalEquity95CI.upper.toFixed(0)}
              </div>
            </div>

            {/* Trade Stats */}
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-muted/30 rounded p-2">
                <div className="font-semibold">{result.tradeStats.totalTrades}</div>
                <div className="text-muted-foreground">Trades</div>
              </div>
              <div className="bg-muted/30 rounded p-2">
                <div className="font-semibold">{result.tradeStats.winRate}</div>
                <div className="text-muted-foreground">Win Rate</div>
              </div>
              <div className="bg-muted/30 rounded p-2">
                <div className="font-semibold">{result.tradeStats.avgReturnPips}</div>
                <div className="text-muted-foreground">Avg Return</div>
              </div>
              <div className="bg-muted/30 rounded p-2">
                <div className="font-semibold">{result.tradeStats.volatilityPips}</div>
                <div className="text-muted-foreground">Volatility</div>
              </div>
            </div>

            {/* Risk Assessment */}
            {result.riskAssessment && result.riskAssessment.length > 0 && (
              <div className="space-y-2">
                <span className="text-sm font-medium">Risk Assessment</span>
                {result.riskAssessment.map((assessment, index) => (
                  <div
                    key={index}
                    className={`flex items-start gap-2 text-sm p-2 rounded ${
                      assessment.level === 'CRITICAL'
                        ? 'bg-red-500/10 text-red-500'
                        : 'bg-yellow-500/10 text-yellow-500'
                    }`}
                  >
                    <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium">{assessment.message}</p>
                      <p className="text-xs opacity-80">{assessment.action}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Interpretation */}
            <div className="text-xs text-muted-foreground space-y-1 border-t pt-3">
              <p>{result.interpretation.probabilityOfRuin}</p>
              <p>{result.interpretation.maxDrawdown}</p>
              <p>{result.interpretation.expectedOutcome}</p>
            </div>

            {/* Simulation Info */}
            <div className="text-center text-xs text-muted-foreground">
              {result.simulationsRun.toLocaleString()} simulations on {result.tradesAnalyzed} trades
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default StressTestPanel;
