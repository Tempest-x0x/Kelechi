import { useState } from 'react';
import { ThemeProvider } from 'next-themes';
import { Header } from '@/components/trading/Header';
import { PriceDisplay } from '@/components/trading/PriceDisplay';
import { CandlestickChart } from '@/components/trading/CandlestickChart';
import { TimeframeSelector } from '@/components/trading/TimeframeSelector';
import { SignalCard } from '@/components/trading/SignalCard';
import TechnicalIndicators from '@/components/trading/TechnicalIndicators';
import { SentimentPanel } from '@/components/trading/SentimentPanel';
import { PatternStatsPanel } from '@/components/trading/PatternStatsPanel';
import { PredictionHistory } from '@/components/trading/PredictionHistory';
import { LearningsPanel } from '@/components/trading/LearningsPanel';
import { useForexData } from '@/hooks/useForexData';
import { usePrediction } from '@/hooks/usePrediction';
import { usePredictionHistory } from '@/hooks/usePredictionHistory';
import { Timeframe } from '@/types/trading';
import { Loader2 } from 'lucide-react';

const Index = () => {
  const [timeframe, setTimeframe] = useState<Timeframe>('1h');
  
  const { data: forexData, isLoading: forexLoading, error: forexError } = useForexData(timeframe);
  const { prediction, isLoading: predictionLoading, generatePrediction } = usePrediction();
  const { predictions, isLoading: historyLoading } = usePredictionHistory();

  const handleGeneratePrediction = () => {
    if (forexData?.candles) {
      // Sentiment disabled - pass null
      generatePrediction(forexData.candles, timeframe, null);
    }
  };

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        
        <main className="container mx-auto px-4 py-6">
          {/* Price Header */}
          <div className="mb-6">
            <PriceDisplay 
              price={forexData?.currentPrice ?? null}
              previousPrice={forexData?.candles?.[1]?.close ?? null}
              isLoading={forexLoading}
            />
          </div>

          {/* Main Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column - Chart & Controls */}
            <div className="lg:col-span-8 space-y-4">
              <div className="flex items-center justify-between">
                <TimeframeSelector 
                  value={timeframe} 
                  onChange={setTimeframe} 
                />
              </div>
              
              <div className="bg-card rounded-xl border border-border p-4 min-h-[400px]">
                {forexLoading ? (
                  <div className="h-[400px] flex items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : forexError ? (
                  <div className="h-[400px] flex items-center justify-center text-destructive">
                    {forexError}
                  </div>
                ) : forexData?.candles ? (
                  <CandlestickChart 
                    candles={forexData.candles}
                    currentPrice={forexData.currentPrice}
                    entryPrice={prediction?.entry_price}
                    stopLoss={prediction?.stop_loss ?? undefined}
                    takeProfit1={prediction?.take_profit_1}
                    takeProfit2={prediction?.take_profit_2}
                    indicators={prediction?.technical_indicators}
                  />
                ) : (
                  <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                    No data available
                  </div>
                )}
              </div>

              {/* Technical Indicators */}
              <TechnicalIndicators 
                indicators={prediction?.technical_indicators ?? null}
                currentPrice={forexData?.currentPrice}
              />
            </div>

            {/* Right Column - Signal & Sentiment */}
            <div className="lg:col-span-4 space-y-4">
              <SignalCard 
                prediction={prediction}
                isLoading={predictionLoading}
                onGenerateSignal={handleGeneratePrediction}
                currentPrice={forexData?.currentPrice ?? null}
              />
              
              <SentimentPanel 
                sentiment={null}
                isLoading={false}
              />
              
              <PatternStatsPanel 
                detectedPatterns={prediction?.patterns_detected as string[] | undefined}
              />
            </div>
          </div>

          {/* Prediction History & Learnings */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PredictionHistory 
                predictions={predictions}
                isLoading={historyLoading}
              />
            </div>
            <div className="lg:col-span-1">
              <LearningsPanel />
            </div>
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
};

export default Index;
