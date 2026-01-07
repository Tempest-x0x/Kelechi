import { useLearnings } from '@/hooks/useLearnings';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Brain, CheckCircle, XCircle, TrendingUp, TrendingDown, Lightbulb, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';

export function LearningsPanel() {
  const { learnings, isLoading, stats, refetch } = useLearnings();

  const winRate = stats.total > 0 
    ? ((stats.successes / stats.total) * 100).toFixed(1) 
    : '0.0';

  return (
    <Card className="bg-card/50 backdrop-blur border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Learnings
          </CardTitle>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={refetch}
            disabled={isLoading}
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
        </div>
        
        {/* Stats Summary */}
        <div className="grid grid-cols-3 gap-2 mt-3">
          <div className="bg-muted/50 rounded-lg p-2 text-center">
            <div className="text-xl font-bold text-foreground">{stats.total}</div>
            <div className="text-xs text-muted-foreground">Total</div>
          </div>
          <div className="bg-bullish/10 rounded-lg p-2 text-center">
            <div className="text-xl font-bold text-bullish">{stats.successes}</div>
            <div className="text-xs text-muted-foreground">Wins</div>
          </div>
          <div className="bg-bearish/10 rounded-lg p-2 text-center">
            <div className="text-xl font-bold text-bearish">{stats.failures}</div>
            <div className="text-xs text-muted-foreground">Losses</div>
          </div>
        </div>

        {/* Win Rate */}
        {stats.total > 0 && (
          <div className="mt-3 p-3 bg-primary/10 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Win Rate</span>
              <span className={cn(
                "text-lg font-bold",
                parseFloat(winRate) >= 50 ? "text-bullish" : "text-bearish"
              )}>
                {winRate}%
              </span>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : learnings.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Lightbulb className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No learnings yet</p>
            <p className="text-xs mt-1">Learnings will appear as predictions are evaluated</p>
          </div>
        ) : (
          <ScrollArea className="h-[350px]">
            <div className="space-y-3 pr-3">
              {learnings.map((learning) => {
                const isWin = !!learning.success_factors;
                const patternContext = learning.pattern_context as Record<string, unknown> | null;
                const marketConditions = learning.market_conditions as Record<string, unknown> | null;
                const signalType = (patternContext?.signal_type as string) || 'UNKNOWN';
                const patterns = patternContext?.patterns as string[] | undefined;
                const entryPrice = marketConditions?.entry_price as number | undefined;
                const outcomePrice = marketConditions?.outcome_price as number | undefined;
                
                return (
                  <div 
                    key={learning.id}
                    className={cn(
                      "p-3 rounded-lg border",
                      isWin 
                        ? "bg-bullish/5 border-bullish/20" 
                        : "bg-bearish/5 border-bearish/20"
                    )}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {isWin ? (
                          <CheckCircle className="h-4 w-4 text-bullish" />
                        ) : (
                          <XCircle className="h-4 w-4 text-bearish" />
                        )}
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs",
                            signalType === 'BUY' && "border-bullish text-bullish",
                            signalType === 'SELL' && "border-bearish text-bearish"
                          )}
                        >
                          {signalType === 'BUY' && <TrendingUp className="h-3 w-3 mr-1" />}
                          {signalType === 'SELL' && <TrendingDown className="h-3 w-3 mr-1" />}
                          {signalType}
                        </Badge>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {format(new Date(learning.created_at), 'MMM d, HH:mm')}
                      </span>
                    </div>

                    {/* Lesson */}
                    <p className="text-sm text-foreground mb-2">
                      {learning.lesson_extracted}
                    </p>

                    {/* Price Movement */}
                    {entryPrice !== undefined && outcomePrice !== undefined && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>Entry: {entryPrice.toFixed(5)}</span>
                        <span>→</span>
                        <span>Exit: {outcomePrice.toFixed(5)}</span>
                      </div>
                    )}

                    {/* Patterns */}
                    {patterns && patterns.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {patterns.slice(0, 2).map((pattern, idx) => (
                          <Badge key={idx} variant="secondary" className="text-[10px] px-1.5 py-0">
                            {pattern}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
