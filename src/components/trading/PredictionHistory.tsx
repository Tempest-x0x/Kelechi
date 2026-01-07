import { memo } from 'react';
import { TrendingUp, TrendingDown, CheckCircle, XCircle, Clock, Hourglass } from 'lucide-react';
import { Prediction } from '@/types/trading';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';

interface PredictionHistoryProps {
  predictions: Prediction[];
  isLoading?: boolean;
}

export const PredictionHistory = memo(function PredictionHistory({
  predictions,
  isLoading,
}: PredictionHistoryProps) {
  if (isLoading) {
    return (
      <div className="trading-card p-4">
        <h3 className="font-semibold mb-4">Prediction History</h3>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-muted animate-pulse rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="trading-card p-4">
        <h3 className="font-semibold mb-4">Prediction History</h3>
        <div className="text-center py-8">
          <Clock className="h-10 w-10 text-muted-foreground mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">No predictions yet</p>
        </div>
      </div>
    );
  }

  // Calculate stats
  const completedPredictions = predictions.filter((p) => p.outcome === 'WIN' || p.outcome === 'LOSS');
  const wins = predictions.filter((p) => p.outcome === 'WIN').length;
  const losses = predictions.filter((p) => p.outcome === 'LOSS').length;
  const pending = predictions.filter((p) => p.outcome === 'PENDING').length;
  const winRate = completedPredictions.length > 0 ? (wins / completedPredictions.length) * 100 : 0;

  const signalIcons: Record<string, typeof TrendingUp> = {
    BUY: TrendingUp,
    SELL: TrendingDown,
  };

  const signalColors: Record<string, string> = {
    BUY: 'text-bullish',
    SELL: 'text-bearish',
  };

  const outcomeIcons: Record<string, typeof CheckCircle> = {
    WIN: CheckCircle,
    LOSS: XCircle,
    PENDING: Hourglass,
    EXPIRED: Clock,
  };

  const outcomeColors: Record<string, string> = {
    WIN: 'text-bullish',
    LOSS: 'text-bearish',
    PENDING: 'text-primary',
    EXPIRED: 'text-muted-foreground',
  };

  return (
    <div className="trading-card p-4">
      <h3 className="font-semibold mb-4">Prediction History</h3>

      {/* Stats summary */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-secondary/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold">{predictions.length}</p>
          <p className="text-xs text-muted-foreground">Total</p>
        </div>
        <div className="bg-secondary/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-bullish">{wins}</p>
          <p className="text-xs text-muted-foreground">Wins</p>
        </div>
        <div className="bg-secondary/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-bearish">{losses}</p>
          <p className="text-xs text-muted-foreground">Losses</p>
        </div>
        <div className="bg-secondary/50 rounded-lg p-2 text-center">
          <p className={cn('text-lg font-bold', winRate >= 50 ? 'text-bullish' : 'text-bearish')}>
            {winRate.toFixed(0)}%
          </p>
          <p className="text-xs text-muted-foreground">Win Rate</p>
        </div>
      </div>

      {/* Pending count */}
      {pending > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 bg-primary/10 rounded-lg mb-4">
          <Hourglass className="h-4 w-4 text-primary" />
          <span className="text-sm">{pending} prediction{pending > 1 ? 's' : ''} pending</span>
        </div>
      )}

      {/* History list */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto scrollbar-thin">
        {predictions.filter(p => p.signal_type === 'BUY' || p.signal_type === 'SELL').map((prediction) => {
          const SignalIcon = signalIcons[prediction.signal_type] || TrendingUp;
          const outcomeKey = prediction.outcome || 'PENDING';
          const OutcomeIcon = outcomeIcons[outcomeKey] || Hourglass;

          return (
            <div
              key={prediction.id}
              className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg hover:bg-secondary/50 transition-colors"
            >
              {/* Signal */}
              <div className={cn('p-2 rounded-lg bg-card', signalColors[prediction.signal_type])}>
                <SignalIcon className="h-4 w-4" />
              </div>

              {/* Details */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={cn('font-semibold text-sm', signalColors[prediction.signal_type])}>
                    {prediction.signal_type}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    @ {prediction.entry_price.toFixed(5)}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground truncate">
                  {format(new Date(prediction.created_at), 'MMM d, HH:mm')} • {prediction.confidence.toFixed(0)}% confidence
                </p>
              </div>

              {/* Outcome */}
              <div className={cn('flex items-center gap-1', outcomeColors[outcomeKey] || 'text-muted-foreground')}>
                <OutcomeIcon className="h-4 w-4" />
                <span className="text-xs font-medium">{outcomeKey}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});