import { memo } from 'react';
import { Newspaper, Clock } from 'lucide-react';

interface SentimentPanelProps {
  sentiment: unknown;
  isLoading?: boolean;
  onFetch?: () => void;
}

export const SentimentPanel = memo(function SentimentPanel({
  isLoading,
}: SentimentPanelProps) {
  if (isLoading) {
    return (
      <div className="trading-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold flex items-center gap-2">
            <Newspaper className="h-4 w-4" />
            Market Sentiment
          </h3>
        </div>
        <div className="space-y-3 animate-pulse">
          <div className="h-16 bg-muted rounded" />
          <div className="h-4 bg-muted rounded w-3/4" />
        </div>
      </div>
    );
  }

  // Show "Coming Soon" state
  return (
    <div className="trading-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold flex items-center gap-2">
          <Newspaper className="h-4 w-4" />
          Market Sentiment
        </h3>
      </div>
      
      <div className="text-center py-8">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary/10 mb-4">
          <Clock className="h-6 w-6 text-primary" />
        </div>
        <h4 className="font-medium text-foreground mb-2">Coming Soon</h4>
        <p className="text-sm text-muted-foreground max-w-[200px] mx-auto">
          Real-time news sentiment analysis is being integrated with Alpha Vantage API
        </p>
      </div>
    </div>
  );
});
