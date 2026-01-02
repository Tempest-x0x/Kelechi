import { useState, useEffect } from 'react';
import { ThemeProvider } from 'next-themes';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Activity, 
  TrendingUp, 
  Shield, 
  Target, 
  CheckCircle2, 
  ArrowRight,
  Star,
  Zap,
  Clock,
  BarChart3,
  Moon,
  Sun
} from 'lucide-react';
import { useTheme } from 'next-themes';
import { supabase } from '@/integrations/supabase/client';
import { PricingCard } from '@/components/landing/PricingCard';
import { FAQSection } from '@/components/landing/FAQSection';

const ThemeToggle = () => {
  const { theme, setTheme } = useTheme();
  
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
    >
      <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
    </Button>
  );
};

const Landing = () => {
  const navigate = useNavigate();
  const [winRate, setWinRate] = useState<number | null>(null);
  const [totalPredictions, setTotalPredictions] = useState<number>(0);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Check auth status
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsLoggedIn(!!session);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_, session) => {
      setIsLoggedIn(!!session);
    });

    // Fetch win rate from predictions
    const fetchStats = async () => {
      const { data, error } = await supabase
        .from('predictions')
        .select('outcome')
        .not('outcome', 'is', null);
      
      if (!error && data) {
        const wins = data.filter(p => p.outcome === 'WIN').length;
        const total = data.length;
        if (total > 0) {
          setWinRate(Math.round((wins / total) * 100));
          setTotalPredictions(total);
        }
      }
    };

    fetchStats();
    return () => subscription.unsubscribe();
  }, []);

  const features = [
    {
      icon: Target,
      title: 'Directional Bias',
      description: 'Clear BUY/SELL/HOLD direction every trading day with confidence scoring'
    },
    {
      icon: Shield,
      title: 'Invalidation Levels',
      description: 'Know exactly where your bias is wrong. Protect your capital with precision'
    },
    {
      icon: TrendingUp,
      title: 'Target Levels',
      description: 'Two take-profit targets based on pattern analysis and historical data'
    },
    {
      icon: BarChart3,
      title: 'Pattern Recognition',
      description: 'AI-powered pattern detection with historical win rate statistics'
    }
  ];

  const howItWorks = [
    {
      step: 1,
      title: 'Get Morning Bias',
      description: 'Receive your directional bias and key levels before market opens'
    },
    {
      step: 2,
      title: 'Wait for Entry',
      description: 'Let price come to your entry zone. No chasing. No FOMO.'
    },
    {
      step: 3,
      title: 'Manage with Rules',
      description: 'Use invalidation level as your stop. Exit at targets.'
    },
    {
      step: 4,
      title: 'Review & Learn',
      description: 'Track record and AI learnings help refine future biases'
    }
  ];

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Activity className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="font-semibold text-lg leading-none">ForexTell AI</h1>
                <p className="text-xs text-muted-foreground">EUR/USD Decision Engine</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Link to="/track-record" className="text-sm text-muted-foreground hover:text-foreground transition-colors hidden sm:block">
                Track Record
              </Link>
              <ThemeToggle />
              {isLoggedIn ? (
                <Button onClick={() => navigate('/dashboard')}>
                  Dashboard
                </Button>
              ) : (
                <Button onClick={() => navigate('/auth')}>
                  Get Started
                </Button>
              )}
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-20 lg:py-32">
          <div className="container mx-auto px-4 text-center">
            <Badge variant="secondary" className="mb-6">
              <Zap className="h-3 w-3 mr-1" />
              EUR/USD Only • Focused Precision
            </Badge>
            
            <h1 className="text-4xl lg:text-6xl font-bold mb-6 leading-tight">
              Institutional-Style<br />
              <span className="text-primary">EUR/USD Decision Engine</span>
            </h1>
            
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Directional bias + invalidation levels. No signals. No gambling.<br />
              Just clear, actionable trading decisions.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Button size="lg" onClick={() => navigate('/auth')} className="text-lg px-8">
                Start Trading Smarter
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button size="lg" variant="outline" onClick={() => navigate('/track-record')} className="text-lg px-8">
                View Track Record
              </Button>
            </div>

            {/* Stats */}
            {winRate !== null && (
              <div className="flex flex-wrap justify-center gap-8 text-center">
                <div className="bg-card border border-border rounded-xl p-6 min-w-[140px]">
                  <div className="text-3xl font-bold text-bullish">{winRate}%</div>
                  <div className="text-sm text-muted-foreground">Win Rate</div>
                </div>
                <div className="bg-card border border-border rounded-xl p-6 min-w-[140px]">
                  <div className="text-3xl font-bold">{totalPredictions}</div>
                  <div className="text-sm text-muted-foreground">Predictions</div>
                </div>
                <div className="bg-card border border-border rounded-xl p-6 min-w-[140px]">
                  <div className="text-3xl font-bold text-primary">EUR/USD</div>
                  <div className="text-sm text-muted-foreground">Focus Pair</div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 bg-secondary/30">
          <div className="container mx-auto px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">What You Get</h2>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Everything you need to make informed trading decisions. Nothing you don't.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {features.map((feature, index) => (
                <div key={index} className="bg-card border border-border rounded-xl p-6 hover:border-primary/50 transition-colors">
                  <div className="p-3 bg-primary/10 rounded-lg w-fit mb-4">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="py-20">
          <div className="container mx-auto px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">How It Works</h2>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Simple, systematic, and repeatable. The way trading should be.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
              {howItWorks.map((item, index) => (
                <div key={index} className="relative">
                  <div className="bg-card border border-border rounded-xl p-6 h-full">
                    <div className="w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold mb-4">
                      {item.step}
                    </div>
                    <h3 className="font-semibold mb-2">{item.title}</h3>
                    <p className="text-sm text-muted-foreground">{item.description}</p>
                  </div>
                  {index < howItWorks.length - 1 && (
                    <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                      <ArrowRight className="h-6 w-6 text-muted-foreground" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section className="py-20 bg-secondary/30" id="pricing">
          <div className="container mx-auto px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4">Simple Pricing</h2>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Choose the plan that fits your trading style. Cancel anytime.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              <PricingCard
                name="Retail"
                priceUSD={49}
                priceNGN={76000}
                period="month"
                features={[
                  'Daily directional bias',
                  'Entry & invalidation levels',
                  'Target levels',
                  'Pattern detection',
                  'Basic email alerts'
                ]}
                planType="retail"
              />
              <PricingCard
                name="Funded Trader"
                priceUSD={199}
                priceNGN={310000}
                period="month"
                features={[
                  'Everything in Retail',
                  'Session timing filters',
                  'Prop firm compliance mode',
                  'Advanced pattern stats',
                  'Priority support',
                  'Multiple timeframes'
                ]}
                planType="funded"
                popular
              />
              <PricingCard
                name="Lifetime"
                priceUSD={999}
                priceNGN={1550000}
                period="one-time"
                features={[
                  'Everything in Funded Trader',
                  'Lifetime access forever',
                  'First 50 customers only',
                  'All future features included',
                  'VIP support channel',
                  'Early access to new pairs'
                ]}
                planType="lifetime"
              />
            </div>
          </div>
        </section>

        {/* FAQ Section */}
        <FAQSection />

        {/* CTA Section */}
        <section className="py-20 bg-primary/5">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-4">Ready to Trade Smarter?</h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Join traders who've stopped gambling and started making informed decisions.
            </p>
            <Button size="lg" onClick={() => navigate('/auth')} className="text-lg px-8">
              Get Started Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 border-t border-border">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary rounded-lg">
                  <Activity className="h-4 w-4 text-primary-foreground" />
                </div>
                <span className="font-semibold">ForexTell AI</span>
              </div>
              <div className="flex gap-6 text-sm text-muted-foreground">
                <Link to="/track-record" className="hover:text-foreground transition-colors">Track Record</Link>
                <a href="#pricing" className="hover:text-foreground transition-colors">Pricing</a>
              </div>
              <p className="text-sm text-muted-foreground">
                © {new Date().getFullYear()} ForexTell AI. Not financial advice.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </ThemeProvider>
  );
};

export default Landing;
