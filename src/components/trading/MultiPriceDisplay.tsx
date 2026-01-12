import { useEffect, useState, useCallback } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { CurrencyPair } from '@/hooks/useCurrencyPairs';
import { cn } from '@/lib/utils';

interface PriceData {
  symbol: string;
  price: number;
  prevPrice: number;
  change: number;
  changePercent: number;
}

interface MultiPriceDisplayProps {
  pairs: CurrencyPair[];
  selectedSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

export function MultiPriceDisplay({ 
  pairs, 
  selectedSymbol, 
  onSelectSymbol 
}: MultiPriceDisplayProps) {
  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [isLoading, setIsLoading] = useState(true);

  const fetchPrices = useCallback(async () => {
    try {
      // Fetch latest candle for each active pair from price_history
      const pricePromises = pairs.map(async (pair) => {
        const { data } = await supabase
          .from('price_history')
          .select('close, timestamp')
          .eq('symbol', pair.symbol)
          .eq('timeframe', '1h')
          .order('timestamp', { ascending: false })
          .limit(2);

        if (data && data.length >= 1) {
          const currentPrice = Number(data[0].close);
          const prevPrice = data.length >= 2 ? Number(data[1].close) : currentPrice;
          const change = currentPrice - prevPrice;
          const changePercent = prevPrice > 0 ? (change / prevPrice) * 100 : 0;

          return {
            symbol: pair.symbol,
            price: currentPrice,
            prevPrice,
            change,
            changePercent
          };
        }
        return null;
      });

      const results = await Promise.all(pricePromises);
      const priceMap: Record<string, PriceData> = {};
      results.forEach(result => {
        if (result) {
          priceMap[result.symbol] = result;
        }
      });
      setPrices(priceMap);
    } catch (err) {
      console.error('Failed to fetch prices:', err);
    } finally {
      setIsLoading(false);
    }
  }, [pairs]);

  useEffect(() => {
    if (pairs.length > 0) {
      fetchPrices();
      const interval = setInterval(fetchPrices, 60000); // Refresh every minute
      return () => clearInterval(interval);
    }
  }, [pairs, fetchPrices]);

  const formatPrice = (price: number, symbol: string) => {
    // JPY pairs and gold have different decimal places
    if (symbol.includes('JPY')) {
      return price.toFixed(3);
    }
    if (symbol.includes('XAU')) {
      return price.toFixed(2);
    }
    return price.toFixed(5);
  };

  const formatChange = (change: number, symbol: string) => {
    const decimals = symbol.includes('JPY') || symbol.includes('XAU') ? 2 : 4;
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(decimals)}`;
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
        {pairs.slice(0, 6).map((pair) => (
          <div 
            key={pair.id} 
            className="bg-card/50 rounded-lg border border-border/50 p-3 animate-pulse"
          >
            <div className="h-4 bg-muted rounded w-16 mb-2"></div>
            <div className="h-5 bg-muted rounded w-20"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
      {pairs.map((pair) => {
        const priceData = prices[pair.symbol];
        const isSelected = pair.symbol === selectedSymbol;
        const trend = priceData?.change ?? 0;

        return (
          <button
            key={pair.id}
            onClick={() => onSelectSymbol(pair.symbol)}
            className={cn(
              "text-left bg-card/50 rounded-lg border p-3 transition-all hover:border-primary/50",
              isSelected 
                ? "border-primary bg-primary/5 ring-1 ring-primary/20" 
                : "border-border/50"
            )}
          >
            <div className="flex items-center justify-between mb-1">
              <span className={cn(
                "text-xs font-medium",
                isSelected ? "text-primary" : "text-muted-foreground"
              )}>
                {pair.symbol}
              </span>
              {trend > 0 ? (
                <TrendingUp className="h-3 w-3 text-emerald-500" />
              ) : trend < 0 ? (
                <TrendingDown className="h-3 w-3 text-rose-500" />
              ) : (
                <Minus className="h-3 w-3 text-muted-foreground" />
              )}
            </div>
            <div className="font-mono text-sm font-semibold">
              {priceData ? formatPrice(priceData.price, pair.symbol) : '--.-----'}
            </div>
            {priceData && (
              <div className={cn(
                "text-xs font-mono",
                trend > 0 ? "text-emerald-500" : trend < 0 ? "text-rose-500" : "text-muted-foreground"
              )}>
                {formatChange(priceData.change, pair.symbol)} ({priceData.changePercent >= 0 ? '+' : ''}{priceData.changePercent.toFixed(2)}%)
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
