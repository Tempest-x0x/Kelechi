import { memo, useEffect, useState } from 'react';
import { BarChart3, TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { cn } from '@/lib/utils';

interface PatternStat {
  id: string;
  pattern_name: string;
  signal_type: 'BUY' | 'SELL';
  occurrences: number;
  win_rate_4h: number | null;
  win_rate_12h: number | null;
  win_rate_24h: number | null;
  win_rate_48h: number | null;
  avg_pips_4h: number | null;
  avg_pips_12h: number | null;
  avg_pips_24h: number | null;
  avg_pips_48h: number | null;
}

interface PatternStatsPanelProps {
  detectedPatterns?: string[];
  className?: string;
}

// Map detected pattern names to database pattern names
const patternMapping: Record<string, string> = {
  'RSI oversold': 'rsi_oversold',
  'RSI overbought': 'rsi_overbought',
  'MACD bullish': 'macd_bullish_cross',
  'MACD bearish': 'macd_bearish_cross',
  'Bollinger Lower': 'bb_lower_touch',
  'Bollinger Upper': 'bb_upper_touch',
  'Bullish Engulfing': 'bullish_engulfing',
  'Bearish Engulfing': 'bearish_engulfing',
  'Strong Uptrend': 'golden_cross',
  'Strong Downtrend': 'death_cross',
};

export const PatternStatsPanel = memo(function PatternStatsPanel({
  detectedPatterns = [],
  className,
}: PatternStatsPanelProps) {
  const [allStats, setAllStats] = useState<PatternStat[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      const { data, error } = await supabase
        .from('pattern_statistics')
        .select('*')
        .order('occurrences', { ascending: false });

      if (!error && data) {
        setAllStats(data as PatternStat[]);
      }
      setIsLoading(false);
    }

    fetchStats();
  }, []);

  if (isLoading) {
    return (
      <div className={cn("trading-card p-4", className)}>
        <h3 className="font-semibold flex items-center gap-2 mb-3">
          <BarChart3 className="h-4 w-4" />
          Historical Pattern Statistics
        </h3>
        <div className="space-y-2 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-12 bg-muted rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (allStats.length === 0) {
    return (
      <div className={cn("trading-card p-4", className)}>
        <h3 className="font-semibold flex items-center gap-2 mb-3">
          <BarChart3 className="h-4 w-4" />
          Historical Pattern Statistics
        </h3>
        <p className="text-sm text-muted-foreground text-center py-4">
          No pattern statistics available. Import historical data to enable this feature.
        </p>
      </div>
    );
  }

  // Find matching stats for detected patterns
  const matchedPatterns: PatternStat[] = [];
  const unmatchedPatterns = [...allStats];

  detectedPatterns.forEach((pattern) => {
    Object.entries(patternMapping).forEach(([key, dbName]) => {
      if (pattern.toLowerCase().includes(key.toLowerCase())) {
        const stat = allStats.find((s) => s.pattern_name === dbName);
        if (stat && !matchedPatterns.includes(stat)) {
          matchedPatterns.push(stat);
          const idx = unmatchedPatterns.indexOf(stat);
          if (idx > -1) unmatchedPatterns.splice(idx, 1);
        }
      }
    });
  });

  const formatPatternName = (name: string) => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getWinRateColor = (rate: number | null) => {
    if (rate === null) return 'text-muted-foreground';
    if (rate >= 52) return 'text-bullish';
    if (rate >= 48) return 'text-neutral';
    return 'text-bearish';
  };

  const getWinRateIcon = (rate: number | null) => {
    if (rate === null) return null;
    if (rate >= 52) return <CheckCircle className="h-3.5 w-3.5 text-bullish" />;
    if (rate >= 48) return <AlertTriangle className="h-3.5 w-3.5 text-neutral" />;
    return <AlertTriangle className="h-3.5 w-3.5 text-bearish" />;
  };

  const renderPatternRow = (stat: PatternStat, isDetected: boolean) => (
    <div
      key={stat.id}
      className={cn(
        'p-3 rounded-lg border transition-colors',
        isDetected
          ? 'bg-primary/5 border-primary/30'
          : 'bg-secondary/30 border-transparent'
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {stat.signal_type === 'BUY' ? (
            <TrendingUp className="h-4 w-4 text-bullish" />
          ) : (
            <TrendingDown className="h-4 w-4 text-bearish" />
          )}
          <span className="font-medium text-sm">
            {formatPatternName(stat.pattern_name)}
          </span>
          {isDetected && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-primary/20 text-primary rounded">
              DETECTED
            </span>
          )}
        </div>
        <span className="text-xs text-muted-foreground">
          n={stat.occurrences.toLocaleString()}
        </span>
      </div>

      <div className="grid grid-cols-4 gap-2 text-xs">
        <div className="text-center">
          <p className="text-muted-foreground mb-0.5">4H</p>
          <div className="flex items-center justify-center gap-1">
            {getWinRateIcon(stat.win_rate_4h)}
            <span className={getWinRateColor(stat.win_rate_4h)}>
              {stat.win_rate_4h?.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="text-center">
          <p className="text-muted-foreground mb-0.5">12H</p>
          <div className="flex items-center justify-center gap-1">
            {getWinRateIcon(stat.win_rate_12h)}
            <span className={getWinRateColor(stat.win_rate_12h)}>
              {stat.win_rate_12h?.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="text-center">
          <p className="text-muted-foreground mb-0.5">24H</p>
          <div className="flex items-center justify-center gap-1">
            {getWinRateIcon(stat.win_rate_24h)}
            <span className={getWinRateColor(stat.win_rate_24h)}>
              {stat.win_rate_24h?.toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="text-center">
          <p className="text-muted-foreground mb-0.5">48H</p>
          <div className="flex items-center justify-center gap-1">
            {getWinRateIcon(stat.win_rate_48h)}
            <span className={getWinRateColor(stat.win_rate_48h)}>
              {stat.win_rate_48h?.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className={cn("trading-card p-4", className)}>
      <h3 className="font-semibold flex items-center gap-2 mb-1">
        <BarChart3 className="h-4 w-4" />
        Historical Pattern Statistics
      </h3>
      <p className="text-xs text-muted-foreground mb-4">
        25 years of EUR/USD M1 data (2001-2025)
      </p>

      {matchedPatterns.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-primary mb-2">Currently Detected</p>
          <div className="space-y-2">
            {matchedPatterns.map((stat) => renderPatternRow(stat, true))}
          </div>
        </div>
      )}

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-2">All Patterns</p>
        <div className="space-y-2 max-h-[300px] overflow-y-auto scrollbar-thin">
          {unmatchedPatterns.slice(0, 6).map((stat) => renderPatternRow(stat, false))}
        </div>
      </div>
    </div>
  );
});
