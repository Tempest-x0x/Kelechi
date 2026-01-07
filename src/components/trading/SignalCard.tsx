import { memo } from 'react';
import { TrendingUp, TrendingDown, Pause, Target, Shield, Zap, Clock } from 'lucide-react';
import { Prediction } from '@/types/trading';
import { cn } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';

interface SignalCardProps {
  prediction: Prediction | null;
  isLoading?: boolean;
  currentPrice?: number | null;
  onGenerateSignal?: () => void;
}

export const SignalCard = memo(function SignalCard({ 
  prediction, 
  isLoading, 
  currentPrice,
  onGenerateSignal 
}: SignalCardProps) {
  if (isLoading) {
    return (
      <div className="trading-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-6 w-24 bg-muted animate-pulse rounded" />
        </div>
        <div className="space-y-3">
          <div className="h-16 bg-muted animate-pulse rounded" />
          <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
          <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="trading-card p-6">
        <div className="text-center py-8">
          <Zap className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">No signal generated yet</p>
          <p className="text-sm text-muted-foreground mt-1">Analyze the market with AI</p>
          {onGenerateSignal && (
            <button
              onClick={onGenerateSignal}
              className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
            >
              Generate AI Signal
            </button>
          )}
        </div>
      </div>
    );
  }

  const signalColors = {
    BUY: 'signal-buy',
    SELL: 'signal-sell',
  };

  const SignalIcon = prediction.signal_type === 'BUY' ? TrendingUp : TrendingDown;

  return (
    <div className="trading-card p-6 relative overflow-hidden">
      {/* Signal badge */}
      <div className="flex items-center justify-between mb-4">
        <div className={cn('flex items-center gap-2 px-3 py-1.5 rounded-full font-semibold text-sm', signalColors[prediction.signal_type])}>
          <SignalIcon className="h-4 w-4" />
          {prediction.signal_type}
        </div>
        <div className="text-right">
          <span className="text-2xl font-bold">{prediction.confidence.toFixed(0)}%</span>
          <p className="text-xs text-muted-foreground">Confidence</p>
        </div>
      </div>

      {/* Price targets */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-secondary/50 rounded-lg p-3">
          <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
            <Target className="h-3 w-3" />
            Entry
          </div>
          <p className="font-mono font-semibold">{prediction.entry_price.toFixed(5)}</p>
        </div>
        <div className="bg-secondary/50 rounded-lg p-3">
          <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
            <Shield className="h-3 w-3" />
            Stop Loss
          </div>
          <p className="font-mono font-semibold text-bearish">
            {prediction.stop_loss?.toFixed(5) || '—'}
          </p>
        </div>
        {prediction.take_profit_1 && (
          <div className="bg-secondary/50 rounded-lg p-3">
            <div className="text-xs text-muted-foreground mb-1">TP1</div>
            <p className="font-mono font-semibold text-bullish">
              {prediction.take_profit_1.toFixed(5)}
            </p>
          </div>
        )}
        {prediction.take_profit_2 && (
          <div className="bg-secondary/50 rounded-lg p-3">
            <div className="text-xs text-muted-foreground mb-1">TP2</div>
            <p className="font-mono font-semibold text-bullish">
              {prediction.take_profit_2.toFixed(5)}
            </p>
          </div>
        )}
      </div>

      {/* Trend info */}
      <div className="flex items-center gap-4 mb-4 text-sm">
        <div className={cn(
          'flex items-center gap-1',
          prediction.trend_direction === 'BULLISH' && 'text-bullish',
          prediction.trend_direction === 'BEARISH' && 'text-bearish',
          prediction.trend_direction === 'NEUTRAL' && 'text-muted-foreground'
        )}>
          {prediction.trend_direction === 'BULLISH' && <TrendingUp className="h-4 w-4" />}
          {prediction.trend_direction === 'BEARISH' && <TrendingDown className="h-4 w-4" />}
          {prediction.trend_direction === 'NEUTRAL' && <Pause className="h-4 w-4" />}
          <span>{prediction.trend_direction}</span>
        </div>
        <div className="text-muted-foreground">
          Strength: {prediction.trend_strength?.toFixed(0)}%
        </div>
      </div>

      {/* Reasoning */}
      {prediction.reasoning && (
        <div className="bg-secondary/30 rounded-lg p-3 mb-3">
          <p className="text-sm text-muted-foreground leading-relaxed">{prediction.reasoning}</p>
        </div>
      )}

      {/* Patterns */}
      {prediction.patterns_detected && prediction.patterns_detected.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {prediction.patterns_detected.map((pattern, i) => (
            <span
              key={i}
              className="px-2 py-0.5 bg-primary/10 text-primary text-xs rounded-full"
            >
              {pattern}
            </span>
          ))}
        </div>
      )}

      {/* Timestamp */}
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Clock className="h-3 w-3" />
        Generated {formatDistanceToNow(new Date(prediction.created_at), { addSuffix: true })}
      </div>
    </div>
  );
});