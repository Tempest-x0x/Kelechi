/**
 * Performance Dashboard Component
 * 
 * Displays real-time trading performance metrics:
 * - Live Win Rate (target 70%)
 * - Profit Factor (target > 2.0)
 * - Z-Score (statistical significance)
 * - Account Growth Curve vs 1:2.2 RRR benchmark
 * - Expectancy metrics
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { supabase } from '@/integrations/supabase/client';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Activity,
  Loader2,
  RefreshCw,
} from 'lucide-react';

// Target metrics
const TARGETS = {
  WIN_RATE: 70,
  PROFIT_FACTOR: 2.0,
  RRR_RATIO: 2.2,
  MIN_EXPECTANCY: 1.0,
  Z_SCORE_THRESHOLD: 1.96,
};

interface Trade {
  id: string;
  symbol: string;
  signal_type: 'BUY' | 'SELL';
  entry_price: number;
  stop_loss: number;
  take_profit_1: number;
  outcome: 'WIN' | 'LOSS' | 'PENDING' | 'EXPIRED';
  created_at: string;
  triggered_at: string | null;
}

interface PerformanceMetrics {
  totalTrades: number;
  wins: number;
  losses: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  avgWin: number;
  avgLoss: number;
  zScore: number;
  isStatisticallySignificant: boolean;
  maxDrawdownPercent: number;
}

interface EquityPoint {
  date: string;
  equity: number;
  benchmark: number;
}

function calculatePipsForTrade(trade: Trade): number {
  const pipValue = trade.symbol.includes('JPY') ? 0.01 : 0.0001;
  const isWin = trade.outcome === 'WIN';
  const exitPrice = isWin ? trade.take_profit_1 : trade.stop_loss;
  
  if (trade.signal_type === 'BUY') {
    return (exitPrice - trade.entry_price) / pipValue;
  } else {
    return (trade.entry_price - exitPrice) / pipValue;
  }
}

function calculateZScore(trades: Trade[]): number {
  const completedTrades = trades.filter(t => t.outcome === 'WIN' || t.outcome === 'LOSS');
  if (completedTrades.length < 10) return 0;
  
  const N = completedTrades.length;
  const wins = completedTrades.filter(t => t.outcome === 'WIN').length;
  const losses = N - wins;
  
  if (wins === 0 || losses === 0) return 0;
  
  // Count runs (streaks)
  let R = 1;
  for (let i = 1; i < completedTrades.length; i++) {
    if (completedTrades[i].outcome !== completedTrades[i - 1].outcome) R++;
  }
  
  const P = 2 * wins * losses;
  const numerator = N * (R - 0.5) - P;
  const denominator = Math.sqrt((P * (P - N)) / (N - 1));
  
  if (denominator === 0) return 0;
  return numerator / denominator;
}

function calculateMetrics(trades: Trade[]): PerformanceMetrics {
  const completedTrades = trades.filter(t => t.outcome === 'WIN' || t.outcome === 'LOSS');
  
  if (completedTrades.length === 0) {
    return {
      totalTrades: 0,
      wins: 0,
      losses: 0,
      winRate: 0,
      profitFactor: 0,
      expectancy: 0,
      avgWin: 0,
      avgLoss: 0,
      zScore: 0,
      isStatisticallySignificant: false,
      maxDrawdownPercent: 0,
    };
  }
  
  const wins = completedTrades.filter(t => t.outcome === 'WIN').length;
  const losses = completedTrades.filter(t => t.outcome === 'LOSS').length;
  const winRate = (wins / completedTrades.length) * 100;
  
  // Calculate P&L in pips
  const winningPips = completedTrades
    .filter(t => t.outcome === 'WIN')
    .reduce((sum, t) => sum + Math.abs(calculatePipsForTrade(t)), 0);
  
  const losingPips = completedTrades
    .filter(t => t.outcome === 'LOSS')
    .reduce((sum, t) => sum + Math.abs(calculatePipsForTrade(t)), 0);
  
  const profitFactor = losingPips > 0 ? winningPips / losingPips : winningPips > 0 ? Infinity : 0;
  
  const avgWin = wins > 0 ? winningPips / wins : 0;
  const avgLoss = losses > 0 ? losingPips / losses : 0;
  
  // Normalized expectancy
  const avgRisk = avgLoss > 0 ? avgLoss : 1;
  const normalizedAvgWin = avgWin / avgRisk;
  const expectancy = ((winRate / 100) * normalizedAvgWin) - ((1 - winRate / 100) * 1);
  
  // Max drawdown
  let peak = 0;
  let equity = 0;
  let maxDrawdown = 0;
  
  for (const trade of completedTrades) {
    equity += calculatePipsForTrade(trade);
    peak = Math.max(peak, equity);
    maxDrawdown = Math.max(maxDrawdown, peak > 0 ? (peak - equity) / peak : 0);
  }
  
  const zScore = calculateZScore(completedTrades);
  
  return {
    totalTrades: completedTrades.length,
    wins,
    losses,
    winRate,
    profitFactor,
    expectancy,
    avgWin,
    avgLoss,
    zScore,
    isStatisticallySignificant: Math.abs(zScore) > TARGETS.Z_SCORE_THRESHOLD,
    maxDrawdownPercent: maxDrawdown * 100,
  };
}

function generateEquityCurve(trades: Trade[]): EquityPoint[] {
  const completedTrades = trades
    .filter(t => t.outcome === 'WIN' || t.outcome === 'LOSS')
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  
  let equity = 0;
  let benchmarkEquity = 0;
  const avgRisk = 10; // Assume average risk of 10 pips
  
  return completedTrades.map((trade, index) => {
    const pnl = calculatePipsForTrade(trade);
    equity += pnl;
    
    // Benchmark: simulate 70% win rate with 1:2.2 RRR
    const isWinInBenchmark = (index % 10) < 7; // 70% wins
    benchmarkEquity += isWinInBenchmark ? avgRisk * TARGETS.RRR_RATIO : -avgRisk;
    
    return {
      date: new Date(trade.created_at).toLocaleDateString(),
      equity: Math.round(equity * 10) / 10,
      benchmark: Math.round(benchmarkEquity * 10) / 10,
    };
  });
}

export function PerformanceDashboard() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchTrades = async () => {
    try {
      const { data, error } = await supabase
        .from('trading_opportunities')
        .select('id, symbol, signal_type, entry_price, stop_loss, take_profit_1, outcome, created_at, triggered_at')
        .in('outcome', ['WIN', 'LOSS', 'PENDING', 'EXPIRED'])
        .order('created_at', { ascending: false })
        .limit(200);

      if (error) throw error;

      const tradesData = (data || []) as Trade[];
      setTrades(tradesData);
      setMetrics(calculateMetrics(tradesData));
      setEquityCurve(generateEquityCurve(tradesData));
    } catch (error) {
      console.error('Failed to fetch trades:', error);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, []);

  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchTrades();
  };

  if (isLoading) {
    return (
      <Card className="bg-card border-border">
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  const getWinRateColor = (rate: number) => {
    if (rate >= TARGETS.WIN_RATE) return 'text-green-500';
    if (rate >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getProfitFactorColor = (pf: number) => {
    if (pf >= TARGETS.PROFIT_FACTOR) return 'text-green-500';
    if (pf >= 1.5) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getExpectancyColor = (exp: number) => {
    if (exp >= TARGETS.MIN_EXPECTANCY) return 'text-green-500';
    if (exp >= 0.5) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Performance Dashboard
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Win Rate */}
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Win Rate</span>
              <Target className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className={`text-2xl font-bold ${metrics ? getWinRateColor(metrics.winRate) : ''}`}>
              {metrics?.winRate.toFixed(1)}%
            </div>
            <Progress
              value={metrics?.winRate || 0}
              className="h-1 mt-2"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Target: {TARGETS.WIN_RATE}%</span>
              <span>{metrics?.wins}W / {metrics?.losses}L</span>
            </div>
          </div>

          {/* Profit Factor */}
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Profit Factor</span>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className={`text-2xl font-bold ${metrics ? getProfitFactorColor(metrics.profitFactor) : ''}`}>
              {metrics?.profitFactor === Infinity ? '∞' : metrics?.profitFactor.toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Target: {TARGETS.PROFIT_FACTOR.toFixed(1)}
            </div>
            {metrics && metrics.profitFactor >= TARGETS.PROFIT_FACTOR ? (
              <Badge variant="default" className="mt-2 bg-green-500/20 text-green-500">
                <CheckCircle className="h-3 w-3 mr-1" />
                On Target
              </Badge>
            ) : (
              <Badge variant="destructive" className="mt-2">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Below Target
              </Badge>
            )}
          </div>

          {/* Z-Score */}
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Z-Score</span>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="text-2xl font-bold">
              {metrics?.zScore.toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Threshold: ±{TARGETS.Z_SCORE_THRESHOLD}
            </div>
            {metrics?.isStatisticallySignificant ? (
              <Badge variant="secondary" className="mt-2">
                {metrics.zScore > 0 ? 'Positive Streak' : 'Negative Streak'}
              </Badge>
            ) : (
              <Badge variant="outline" className="mt-2">
                Random Distribution
              </Badge>
            )}
          </div>

          {/* Expectancy */}
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Expectancy</span>
              {(metrics?.expectancy || 0) >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-500" />
              )}
            </div>
            <div className={`text-2xl font-bold ${metrics ? getExpectancyColor(metrics.expectancy) : ''}`}>
              {metrics?.expectancy.toFixed(2)}R
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Min: {TARGETS.MIN_EXPECTANCY}R per trade
            </div>
            {metrics && metrics.expectancy < TARGETS.MIN_EXPECTANCY && (
              <Badge variant="destructive" className="mt-2">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Performance Warning
              </Badge>
            )}
          </div>
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-lg font-semibold">{metrics?.avgWin.toFixed(1)}</div>
            <div className="text-xs text-muted-foreground">Avg Win (pips)</div>
          </div>
          <div>
            <div className="text-lg font-semibold">{metrics?.avgLoss.toFixed(1)}</div>
            <div className="text-xs text-muted-foreground">Avg Loss (pips)</div>
          </div>
          <div>
            <div className="text-lg font-semibold">{metrics?.maxDrawdownPercent.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground">Max Drawdown</div>
          </div>
        </div>

        {/* Equity Curve */}
        {equityCurve.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Equity Curve vs 1:{TARGETS.RRR_RATIO} RRR Benchmark</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={equityCurve} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                  <Line
                    type="monotone"
                    dataKey="equity"
                    name="Actual"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmark"
                    name="Benchmark"
                    stroke="hsl(var(--muted-foreground))"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Warning Messages */}
        {metrics && (
          <div className="space-y-2">
            {metrics.winRate < 60 && (
              <div className="flex items-center gap-2 text-sm text-yellow-500 bg-yellow-500/10 p-2 rounded">
                <AlertTriangle className="h-4 w-4" />
                <span>Win rate below 60% - Review HTF bias alignment</span>
              </div>
            )}
            {metrics.expectancy < TARGETS.MIN_EXPECTANCY && (
              <div className="flex items-center gap-2 text-sm text-red-500 bg-red-500/10 p-2 rounded">
                <AlertTriangle className="h-4 w-4" />
                <span>Expectancy below {TARGETS.MIN_EXPECTANCY}R - Consider tightening entry criteria</span>
              </div>
            )}
            {metrics.isStatisticallySignificant && metrics.zScore < 0 && (
              <div className="flex items-center gap-2 text-sm text-yellow-500 bg-yellow-500/10 p-2 rounded">
                <AlertTriangle className="h-4 w-4" />
                <span>Significant negative streak detected - Review system parameters</span>
              </div>
            )}
          </div>
        )}

        {/* Trade Count */}
        <div className="text-center text-xs text-muted-foreground">
          Based on {metrics?.totalTrades || 0} completed trades
        </div>
      </CardContent>
    </Card>
  );
}

export default PerformanceDashboard;
