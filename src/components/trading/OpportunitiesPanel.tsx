import { TrendingUp, TrendingDown, Clock, Target, Shield, Zap, RefreshCw, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TradingOpportunity } from '@/hooks/useOpportunities';
import { formatDistanceToNow, differenceInMinutes } from 'date-fns';

interface OpportunitiesPanelProps {
  opportunities: TradingOpportunity[];
  isLoading: boolean;
  isScanning: boolean;
  lastScanned: Date | null;
  onScan: () => void;
}

export function OpportunitiesPanel({
  opportunities,
  isLoading,
  isScanning,
  lastScanned,
  onScan
}: OpportunitiesPanelProps) {
  const formatPrice = (price: number | null) => {
    if (price === null) return '-';
    return price.toFixed(5);
  };

  const getTimeRemaining = (expiresAt: string) => {
    const mins = differenceInMinutes(new Date(expiresAt), new Date());
    if (mins <= 0) return 'Expired';
    if (mins < 60) return `${mins}m left`;
    const hours = Math.floor(mins / 60);
    const remainingMins = mins % 60;
    return `${hours}h ${remainingMins}m left`;
  };

  if (isLoading) {
    return (
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Active Opportunities
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Active Opportunities
            {opportunities.length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {opportunities.length}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {lastScanned && (
              <span className="text-xs text-muted-foreground">
                Scanned {formatDistanceToNow(lastScanned, { addSuffix: true })}
              </span>
            )}
            <Button 
              size="sm" 
              variant="outline" 
              onClick={onScan}
              disabled={isScanning}
              className="gap-1"
            >
              <RefreshCw className={`h-3 w-3 ${isScanning ? 'animate-spin' : ''}`} />
              {isScanning ? 'Scanning...' : 'Scan Now'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {opportunities.length === 0 ? (
          <div className="text-center py-8 space-y-3">
            <AlertCircle className="h-10 w-10 text-muted-foreground mx-auto" />
            <div>
              <p className="text-muted-foreground">No active opportunities</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                The scanner runs every 5 minutes looking for high-probability setups
              </p>
            </div>
            <Button 
              size="sm" 
              onClick={onScan}
              disabled={isScanning}
              className="mt-2"
            >
              {isScanning ? 'Scanning...' : 'Scan Now'}
            </Button>
          </div>
        ) : (
          opportunities.map((opp) => (
            <div
              key={opp.id}
              className={`p-4 rounded-lg border ${
                opp.signal_type === 'BUY'
                  ? 'border-emerald-500/30 bg-emerald-500/5'
                  : 'border-rose-500/30 bg-rose-500/5'
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  {/* Symbol Badge */}
                  <Badge variant="outline" className="font-mono text-xs">
                    {opp.symbol || 'EUR/USD'}
                  </Badge>
                  {opp.signal_type === 'BUY' ? (
                    <TrendingUp className="h-5 w-5 text-emerald-500" />
                  ) : (
                    <TrendingDown className="h-5 w-5 text-rose-500" />
                  )}
                  <span className={`font-bold text-lg ${
                    opp.signal_type === 'BUY' ? 'text-emerald-500' : 'text-rose-500'
                  }`}>
                    {opp.signal_type}
                  </span>
                  <Badge 
                    variant={opp.confidence >= 75 ? 'default' : 'secondary'}
                    className={opp.confidence >= 75 ? 'bg-primary' : ''}
                  >
                    {opp.confidence.toFixed(0)}%
                  </Badge>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {getTimeRemaining(opp.expires_at)}
                </div>
              </div>

              {/* Price levels */}
              <div className="grid grid-cols-2 gap-3 mb-3 text-sm">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Entry:</span>
                  <span className="font-mono">{formatPrice(opp.entry_price)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-rose-400" />
                  <span className="text-muted-foreground">SL:</span>
                  <span className="font-mono text-rose-400">{formatPrice(opp.stop_loss)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-emerald-400" />
                  <span className="text-muted-foreground">TP1:</span>
                  <span className="font-mono text-emerald-400">{formatPrice(opp.take_profit_1)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-emerald-400" />
                  <span className="text-muted-foreground">TP2:</span>
                  <span className="font-mono text-emerald-400">{formatPrice(opp.take_profit_2)}</span>
                </div>
              </div>

              {/* Pattern stats */}
              {opp.pattern_stats && opp.pattern_stats.length > 0 && (
                <div className="mb-3">
                  <p className="text-xs text-muted-foreground mb-1">Historical pattern performance:</p>
                  <div className="flex flex-wrap gap-1">
                    {opp.pattern_stats.map((stat: any, idx: number) => (
                      <Badge 
                        key={idx} 
                        variant="outline" 
                        className={`text-xs ${
                          stat.win_rate_24h >= 51 
                            ? 'border-emerald-500/50 text-emerald-400' 
                            : 'border-rose-500/50 text-rose-400'
                        }`}
                      >
                        {stat.pattern_name.replace(/_/g, ' ')}: {stat.win_rate_24h?.toFixed(1)}%
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Detected patterns */}
              {opp.patterns_detected && opp.patterns_detected.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {opp.patterns_detected.slice(0, 3).map((pattern, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {pattern}
                    </Badge>
                  ))}
                </div>
              )}

              {/* Reasoning preview */}
              {opp.reasoning && (
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {opp.reasoning.split('\n')[0]}
                </p>
              )}

              {/* Created time */}
              <p className="text-xs text-muted-foreground/60 mt-2">
                Detected {formatDistanceToNow(new Date(opp.created_at), { addSuffix: true })}
              </p>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
