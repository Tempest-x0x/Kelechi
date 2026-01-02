import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { useBacktest, BacktestResult } from '@/hooks/useBacktest';
import { Loader2, TrendingUp, TrendingDown, Target, AlertTriangle, BarChart3 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

export function BacktestPanel() {
  const { result, isLoading, error, runBacktest } = useBacktest();
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [minConfidence, setMinConfidence] = useState(65);

  const handleRunBacktest = () => {
    runBacktest({
      startDate: startDate || undefined,
      endDate: endDate || undefined,
      minConfidence
    });
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          Strategy Backtest
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Start Date</Label>
            <Input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="h-8 text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">End Date</Label>
            <Input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="h-8 text-sm"
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label className="text-xs text-muted-foreground">Min Confidence</Label>
            <span className="text-xs font-medium">{minConfidence}%</span>
          </div>
          <Slider
            value={[minConfidence]}
            onValueChange={(v) => setMinConfidence(v[0])}
            min={50}
            max={90}
            step={5}
            className="w-full"
          />
        </div>

        <Button 
          onClick={handleRunBacktest} 
          disabled={isLoading}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Running Backtest...
            </>
          ) : (
            <>
              <BarChart3 className="h-4 w-4 mr-2" />
              Run Backtest
            </>
          )}
        </Button>

        {error && (
          <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Results */}
        {result && <BacktestResults result={result} />}
      </CardContent>
    </Card>
  );
}

function BacktestResults({ result }: { result: BacktestResult }) {
  const isProfit = result.totalPips > 0;

  return (
    <div className="space-y-4 pt-4 border-t border-border">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3">
        <StatBox 
          label="Win Rate" 
          value={`${result.winRate.toFixed(1)}%`}
          subValue={`${result.wins}W / ${result.losses}L`}
          positive={result.winRate >= 50}
        />
        <StatBox 
          label="Total Pips" 
          value={`${isProfit ? '+' : ''}${result.totalPips.toFixed(1)}`}
          subValue={`${result.totalSignals} trades`}
          positive={isProfit}
        />
        <StatBox 
          label="Profit Factor" 
          value={result.profitFactor === Infinity ? '∞' : result.profitFactor.toFixed(2)}
          subValue="Gross P / Gross L"
          positive={result.profitFactor >= 1}
        />
        <StatBox 
          label="Max Drawdown" 
          value={`${result.maxDrawdownPips.toFixed(1)} pips`}
          subValue="Peak to trough"
          positive={false}
          isNeutral
        />
      </div>

      {/* Period Info */}
      <div className="text-xs text-muted-foreground text-center">
        {new Date(result.period.start).toLocaleDateString()} - {new Date(result.period.end).toLocaleDateString()}
      </div>

      {/* Pattern Performance */}
      {Object.keys(result.patternPerformance).length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Pattern Performance</h4>
          <div className="space-y-1">
            {Object.entries(result.patternPerformance)
              .sort((a, b) => b[1].winRate - a[1].winRate)
              .slice(0, 5)
              .map(([pattern, stats]) => (
                <div key={pattern} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{pattern}</span>
                  <div className="flex items-center gap-2">
                    <span className={stats.winRate >= 50 ? 'text-green-500' : 'text-red-500'}>
                      {stats.winRate.toFixed(0)}%
                    </span>
                    <span className="text-muted-foreground">
                      ({stats.wins}W/{stats.losses}L)
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Recent Trades */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium">Recent Trades</h4>
        <ScrollArea className="h-[200px]">
          <div className="space-y-1">
            {result.trades.slice(-20).reverse().map((trade, idx) => (
              <div 
                key={idx}
                className="flex items-center justify-between p-2 rounded-lg bg-muted/50 text-xs"
              >
                <div className="flex items-center gap-2">
                  <Badge 
                    variant={trade.signalType === 'BUY' ? 'default' : 'destructive'}
                    className="text-[10px] px-1.5 py-0"
                  >
                    {trade.signalType}
                  </Badge>
                  <span className="text-muted-foreground">
                    {new Date(trade.entryTime).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={trade.outcome === 'WIN' ? 'text-green-500' : 'text-red-500'}>
                    {trade.pips > 0 ? '+' : ''}{trade.pips.toFixed(1)} pips
                  </span>
                  {trade.outcome === 'WIN' ? (
                    <TrendingUp className="h-3 w-3 text-green-500" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-red-500" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

function StatBox({ 
  label, 
  value, 
  subValue, 
  positive, 
  isNeutral = false 
}: { 
  label: string; 
  value: string; 
  subValue: string; 
  positive: boolean;
  isNeutral?: boolean;
}) {
  return (
    <div className="p-3 rounded-lg bg-muted/50">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-lg font-bold ${
        isNeutral ? 'text-foreground' : positive ? 'text-green-500' : 'text-red-500'
      }`}>
        {value}
      </div>
      <div className="text-[10px] text-muted-foreground">{subValue}</div>
    </div>
  );
}
