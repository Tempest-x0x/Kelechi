import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ThemeProvider } from 'next-themes';
import { supabase } from '@/integrations/supabase/client';
import { DashboardHeader } from '@/components/dashboard/DashboardHeader';
import { PriceDisplay } from '@/components/trading/PriceDisplay';
import { TradingViewChart } from '@/components/trading/TradingViewChart';
import { TimeframeSelector } from '@/components/trading/TimeframeSelector';
import { CurrencyPairSelector } from '@/components/trading/CurrencyPairSelector';
import { MultiPriceDisplay } from '@/components/trading/MultiPriceDisplay';
import { SignalCard } from '@/components/trading/SignalCard';
import TechnicalIndicators from '@/components/trading/TechnicalIndicators';
import { SentimentPanel } from '@/components/trading/SentimentPanel';
import { PatternStatsPanel } from '@/components/trading/PatternStatsPanel';
import { OpportunitiesPanel } from '@/components/trading/OpportunitiesPanel';
import { PredictionHistory } from '@/components/trading/PredictionHistory';
import { LearningsPanel } from '@/components/trading/LearningsPanel';
import { BacktestPanel } from '@/components/trading/BacktestPanel';
import { UpgradeModal } from '@/components/dashboard/UpgradeModal';
import { useForexData } from '@/hooks/useForexData';
import { usePrediction } from '@/hooks/usePrediction';
import { usePredictionHistory } from '@/hooks/usePredictionHistory';
import { useOpportunities } from '@/hooks/useOpportunities';
import { useSubscription } from '@/hooks/useSubscription';
import { useAdmin } from '@/hooks/useAdmin';
import { useWhitelist } from '@/hooks/useWhitelist';
import { useCurrencyPairs } from '@/hooks/useCurrencyPairs';
import { Timeframe } from '@/types/trading';
import { Loader2, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';
import { User } from '@supabase/supabase-js';

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [timeframe, setTimeframe] = useState<Timeframe>('1h');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('EUR/USD');
  const [user, setUser] = useState<User | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [isVerifyingPayment, setIsVerifyingPayment] = useState(false);
  
  // Currency pairs
  const { activePairs, isLoading: pairsLoading } = useCurrencyPairs();
  
  // Forex data for selected symbol
  const { data: forexData, isLoading: forexLoading, error: forexError } = useForexData(timeframe, selectedSymbol);
  const { prediction, isLoading: predictionLoading, generatePrediction } = usePrediction();
  const { predictions, isLoading: historyLoading } = usePredictionHistory();
  
  // Get all active opportunities (not filtered by symbol)
  const { opportunities, isLoading: opportunitiesLoading, isScanning, lastScanned, triggerScan } = useOpportunities();
  const { subscription, isLoading: subscriptionLoading, hasActiveSubscription, waitForSubscription } = useSubscription();
  const { isAdmin, isLoading: adminLoading } = useAdmin();
  const { isWhitelisted, isLoading: whitelistLoading } = useWhitelist();

  // Allow access if user has subscription OR is admin OR is whitelisted
  const hasAccess = hasActiveSubscription || isAdmin || isWhitelisted;

  // Handle payment callback verification
  useEffect(() => {
    const reference = searchParams.get('reference');
    const trxref = searchParams.get('trxref');
    const paymentRef = reference || trxref;

    if (paymentRef && !hasActiveSubscription && !isVerifyingPayment) {
      setIsVerifyingPayment(true);
      
      // Clear the query params
      setSearchParams({}, { replace: true });
      
      // Wait for webhook to process and update subscription
      waitForSubscription(12000).then((sub) => {
        setIsVerifyingPayment(false);
        if (sub) {
          const planName = sub.plan_type === 'lifetime' ? 'Lifetime' : 
                          sub.plan_type === 'funded' ? 'Funded Trader' : 'Retail';
          toast.success(
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <div>
                <p className="font-semibold">Welcome to ForexTell AI!</p>
                <p className="text-sm text-muted-foreground">{planName} plan activated successfully</p>
              </div>
            </div>
          );
        } else {
          toast.error(
            'Payment verification pending. If your subscription doesn\'t appear within a few minutes, please contact support.',
            { duration: 8000 }
          );
        }
      });
    }
  }, [searchParams, hasActiveSubscription, isVerifyingPayment, waitForSubscription, setSearchParams]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/auth');
      } else {
        setUser(session.user);
      }
      setIsAuthLoading(false);
    });

    const { data: { subscription: authSubscription } } = supabase.auth.onAuthStateChange((_, session) => {
      if (!session) {
        navigate('/auth');
      } else {
        setUser(session.user);
      }
    });

    return () => authSubscription.unsubscribe();
  }, [navigate]);

  const handleGeneratePrediction = () => {
    if (forexData?.candles) {
      generatePrediction(forexData.candles, timeframe, null);
    }
  };

  const handleScan = async () => {
    try {
      // Scan all active pairs
      const symbols = activePairs.map(p => p.symbol);
      const result = await triggerScan(symbols);
      if (result.opportunitiesFound && result.opportunitiesFound > 0) {
        toast.success(`Found ${result.opportunitiesFound} new opportunity(ies)!`);
      } else if (result.scanned) {
        toast.info(result.message || 'No opportunities found');
      } else {
        toast.info(result.message || 'Scan skipped');
      }
    } catch (err) {
      toast.error('Failed to scan for opportunities');
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate('/');
  };

  if (isAuthLoading || subscriptionLoading || adminLoading || whitelistLoading || isVerifyingPayment) {
    return (
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
        <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          {isVerifyingPayment && (
            <p className="text-muted-foreground animate-pulse">Verifying your payment...</p>
          )}
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        <DashboardHeader 
          user={user} 
          subscription={subscription}
          onLogout={handleLogout} 
        />
        
        <main className="container mx-auto px-4 py-6">
          {/* Multi-Price Display Grid */}
          {!pairsLoading && activePairs.length > 0 && (
            <div className="mb-6">
              <MultiPriceDisplay
                pairs={activePairs}
                selectedSymbol={selectedSymbol}
                onSelectSymbol={setSelectedSymbol}
              />
            </div>
          )}

          {/* Price Header for Selected Pair */}
          <div className="mb-6">
            <PriceDisplay 
              symbol={selectedSymbol}
              price={forexData?.currentPrice ?? null}
              previousPrice={forexData?.candles?.[forexData.candles.length - 2]?.close ?? null}
              isLoading={forexLoading}
            />
          </div>

          {/* Main Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column - Chart & Controls */}
            <div className="lg:col-span-8 space-y-4">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div className="flex items-center gap-3">
                  <CurrencyPairSelector
                    pairs={activePairs}
                    selectedSymbol={selectedSymbol}
                    onSelect={setSelectedSymbol}
                    isLoading={pairsLoading}
                  />
                  <TimeframeSelector 
                    value={timeframe} 
                    onChange={setTimeframe} 
                  />
                </div>
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
                  <TradingViewChart 
                    candles={forexData.candles}
                    currentPrice={forexData.currentPrice}
                    entryPrice={prediction?.entry_price}
                    stopLoss={prediction?.stop_loss ?? undefined}
                    takeProfit1={prediction?.take_profit_1}
                    takeProfit2={prediction?.take_profit_2}
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

            {/* Right Column - Opportunities & Signal */}
            <div className="lg:col-span-4 space-y-4">
              <OpportunitiesPanel 
                opportunities={opportunities}
                isLoading={opportunitiesLoading}
                isScanning={isScanning}
                lastScanned={lastScanned}
                onScan={handleScan}
              />
              
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

          {/* Prediction History, Learnings & Backtest */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <PredictionHistory 
                predictions={predictions}
                isLoading={historyLoading}
              />
            </div>
            <div className="lg:col-span-1">
              <LearningsPanel />
            </div>
            <div className="lg:col-span-1">
              <BacktestPanel />
            </div>
          </div>
        </main>

        {/* Upgrade Modal for non-subscribers */}
        {/* Upgrade Modal - blocks access for non-subscribers (admins bypass) */}
        {!hasAccess && <UpgradeModal forceBlock={true} />}
      </div>
    </ThemeProvider>
  );
};

export default Dashboard;
