import { memo } from 'react';
import { TechnicalIndicators as TechnicalIndicatorsType } from '@/types/trading';
import { cn } from '@/lib/utils';

interface TechnicalIndicatorsProps {
  indicators: TechnicalIndicatorsType | null;
  currentPrice?: number;
}

export const TechnicalIndicators = memo(function TechnicalIndicators({
  indicators,
  currentPrice,
}: TechnicalIndicatorsProps) {
  if (!indicators) {
    return (
      <div className="trading-card p-4">
        <h3 className="font-semibold mb-3">Technical Indicators</h3>
        <p className="text-sm text-muted-foreground">Generate a prediction to see indicators</p>
      </div>
    );
  }

  const getRSIColor = (rsi: number) => {
    if (rsi >= 70) return 'text-bearish';
    if (rsi <= 30) return 'text-bullish';
    return 'text-foreground';
  };

  const getRSILabel = (rsi: number) => {
    if (rsi >= 70) return 'Overbought';
    if (rsi <= 30) return 'Oversold';
    return 'Neutral';
  };

  const getMACDSignal = (macd: { value: number; signal: number; histogram: number }) => {
    if (macd.histogram > 0) return { label: 'Bullish', color: 'text-bullish' };
    if (macd.histogram < 0) return { label: 'Bearish', color: 'text-bearish' };
    return { label: 'Neutral', color: 'text-muted-foreground' };
  };

  const getEMASignal = () => {
    if (indicators.ema9 > indicators.ema21 && indicators.ema21 > indicators.ema50) {
      return { label: 'Bullish Alignment', color: 'text-bullish' };
    }
    if (indicators.ema9 < indicators.ema21 && indicators.ema21 < indicators.ema50) {
      return { label: 'Bearish Alignment', color: 'text-bearish' };
    }
    return { label: 'Mixed', color: 'text-muted-foreground' };
  };

  const getBBPosition = () => {
    if (!currentPrice) return null;
    const range = indicators.bollingerBands.upper - indicators.bollingerBands.lower;
    const position = (currentPrice - indicators.bollingerBands.lower) / range;
    if (position >= 0.8) return { label: 'Near Upper Band', color: 'text-bearish' };
    if (position <= 0.2) return { label: 'Near Lower Band', color: 'text-bullish' };
    return { label: 'Middle Range', color: 'text-muted-foreground' };
  };

  const macdSignal = getMACDSignal(indicators.macd);
  const emaSignal = getEMASignal();
  const bbPosition = getBBPosition();

  return (
    <div className="trading-card p-4">
      <h3 className="font-semibold mb-4">Technical Indicators</h3>

      <div className="space-y-4">
        {/* RSI */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-muted-foreground">RSI (14)</span>
            <span className={cn('text-sm font-medium', getRSIColor(indicators.rsi))}>
              {getRSILabel(indicators.rsi)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full transition-all',
                  indicators.rsi >= 70 && 'bg-bearish',
                  indicators.rsi <= 30 && 'bg-bullish',
                  indicators.rsi > 30 && indicators.rsi < 70 && 'bg-primary'
                )}
                style={{ width: `${Math.min(100, indicators.rsi)}%` }}
              />
            </div>
            <span className="font-mono text-sm font-medium w-12 text-right">
              {indicators.rsi.toFixed(1)}
            </span>
          </div>
        </div>

        {/* MACD */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-muted-foreground">MACD</span>
            <span className={cn('text-sm font-medium', macdSignal.color)}>
              {macdSignal.label}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Value</span>
              <span className="font-mono">{indicators.macd.value.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Signal</span>
              <span className="font-mono">{indicators.macd.signal.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Hist</span>
              <span className={cn('font-mono', indicators.macd.histogram > 0 ? 'text-bullish' : 'text-bearish')}>
                {indicators.macd.histogram.toFixed(5)}
              </span>
            </div>
          </div>
        </div>

        {/* EMAs */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-muted-foreground">EMAs</span>
            <span className={cn('text-sm font-medium', emaSignal.color)}>
              {emaSignal.label}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">EMA 9</span>
              <span className="font-mono">{indicators.ema9.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">EMA 21</span>
              <span className="font-mono">{indicators.ema21.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">EMA 50</span>
              <span className="font-mono">{indicators.ema50.toFixed(5)}</span>
            </div>
          </div>
        </div>

        {/* Bollinger Bands */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-muted-foreground">Bollinger Bands</span>
            {bbPosition && (
              <span className={cn('text-sm font-medium', bbPosition.color)}>
                {bbPosition.label}
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Upper</span>
              <span className="font-mono">{indicators.bollingerBands.upper.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Middle</span>
              <span className="font-mono">{indicators.bollingerBands.middle.toFixed(5)}</span>
            </div>
            <div className="bg-secondary/50 rounded p-1.5 text-center">
              <span className="text-muted-foreground block">Lower</span>
              <span className="font-mono">{indicators.bollingerBands.lower.toFixed(5)}</span>
            </div>
          </div>
        </div>

        {/* Stochastic */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-muted-foreground">Stochastic</span>
            <span className={cn(
              'text-sm font-medium',
              indicators.stochastic.k >= 80 && 'text-bearish',
              indicators.stochastic.k <= 20 && 'text-bullish',
              indicators.stochastic.k > 20 && indicators.stochastic.k < 80 && 'text-muted-foreground'
            )}>
              {indicators.stochastic.k >= 80 ? 'Overbought' : indicators.stochastic.k <= 20 ? 'Oversold' : 'Neutral'}
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">%K:</span>
              <span className="font-mono">{indicators.stochastic.k.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">%D:</span>
              <span className="font-mono">{indicators.stochastic.d.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default TechnicalIndicators;
