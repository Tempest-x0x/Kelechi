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
  ArrowRight,
  Zap,
  BarChart3,
  Moon,
  Sun,
  Check
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
      className="rounded-full"
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
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsLoggedIn(!!session);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_, session) => {
      setIsLoggedIn(!!session);
    });

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
      description: 'Clear BUY/SELL direction every trading day with confidence scoring'
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
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled 
            ? 'bg-background/80 backdrop-blur-lg border-b border-border shadow-soft' 
            : 'bg-transparent'
        }`}>
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="h-16 lg:h-20 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary rounded-xl shadow-primary-glow">
                  <Activity className="h-5 w-5 text-primary-foreground" />
                </div>
                <div>
                  <h1 className="font-semibold text-lg leading-none">ForexTell AI</h1>
                  <p className="text-xs text-muted-foreground hidden sm:block">EUR/USD Decision Engine</p>
                </div>
              </div>

              <div className="flex items-center gap-2 sm:gap-4">
                <Link 
                  to="/track-record" 
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors hidden sm:block"
                >
                  Track Record
                </Link>
                <ThemeToggle />
                {isLoggedIn ? (
                  <Button onClick={() => navigate('/dashboard')} className="rounded-full px-6">
                    Dashboard
                  </Button>
                ) : (
                  <Button onClick={() => navigate('/auth')} className="rounded-full px-6 shadow-primary-glow">
                    Get Started
                  </Button>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="pt-32 pb-20 lg:pt-44 lg:pb-32">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto text-center">
              <Badge 
                variant="secondary" 
                className="mb-8 px-4 py-2 rounded-full text-sm font-medium bg-primary/5 text-primary border-primary/20 hover:bg-primary/10"
              >
                <Zap className="h-3.5 w-3.5 mr-2" />
                EUR/USD Only • Focused Precision
              </Badge>
              
              <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold mb-8 leading-[1.1] tracking-tight animate-fade-in-up">
                Institutional-Style
                <br />
                <span className="text-gradient-primary">EUR/USD Decision Engine</span>
              </h1>
              
              <p className="text-lg sm:text-xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
                Directional bias + invalidation levels. No signals. No gambling.
                <br className="hidden sm:block" />
                Just clear, actionable trading decisions.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-20">
                <Button 
                  size="lg" 
                  onClick={() => navigate('/auth')} 
                  className="text-base px-8 h-14 rounded-full shadow-primary-glow hover:shadow-lg transition-all"
                >
                  Start Trading Smarter
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
                <Button 
                  size="lg" 
                  variant="outline" 
                  onClick={() => navigate('/track-record')} 
                  className="text-base px-8 h-14 rounded-full hover:bg-muted/50"
                >
                  View Track Record
                </Button>
              </div>

              {/* Stats */}
              {winRate !== null && (
                <div className="flex flex-col sm:flex-row justify-center gap-4 sm:gap-0 sm:divide-x divide-border">
                  <div className="px-8 py-4">
                    <div className="text-4xl lg:text-5xl font-bold text-[hsl(var(--bullish))]">{winRate}%</div>
                    <div className="text-sm text-muted-foreground mt-1">Win Rate</div>
                  </div>
                  <div className="px-8 py-4">
                    <div className="text-4xl lg:text-5xl font-bold">{totalPredictions}</div>
                    <div className="text-sm text-muted-foreground mt-1">Predictions</div>
                  </div>
                  <div className="px-8 py-4">
                    <div className="text-4xl lg:text-5xl font-bold text-primary">EUR/USD</div>
                    <div className="text-sm text-muted-foreground mt-1">Focus Pair</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-24 lg:py-32 section-muted">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16 lg:mb-20">
              <h2 className="text-3xl lg:text-4xl font-bold mb-4">What You Get</h2>
              <p className="text-muted-foreground max-w-xl mx-auto text-lg">
                Everything you need to make informed trading decisions. Nothing you don't.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8 max-w-6xl mx-auto">
              {features.map((feature, index) => (
                <div 
                  key={index} 
                  className="premium-card p-8 group"
                >
                  <div className="p-4 bg-primary/5 rounded-2xl w-fit mb-6 group-hover:bg-primary/10 transition-colors">
                    <feature.icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="font-semibold text-lg mb-3">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="py-24 lg:py-32">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16 lg:mb-20">
              <h2 className="text-3xl lg:text-4xl font-bold mb-4">How It Works</h2>
              <p className="text-muted-foreground max-w-xl mx-auto text-lg">
                Simple, systematic, and repeatable. The way trading should be.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
              {howItWorks.map((item, index) => (
                <div key={index} className="relative">
                  <div className="premium-card p-8 h-full">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-primary/80 text-primary-foreground flex items-center justify-center font-bold text-lg mb-6 shadow-primary-glow">
                      {item.step}
                    </div>
                    <h3 className="font-semibold text-lg mb-3">{item.title}</h3>
                    <p className="text-muted-foreground leading-relaxed">{item.description}</p>
                  </div>
                  {index < howItWorks.length - 1 && (
                    <div className="hidden lg:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                      <ArrowRight className="h-5 w-5 text-primary/40" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section className="py-24 lg:py-32 section-muted" id="pricing">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16 lg:mb-20">
              <h2 className="text-3xl lg:text-4xl font-bold mb-4">Simple, Transparent Pricing</h2>
              <p className="text-muted-foreground max-w-xl mx-auto text-lg">
                Choose the plan that fits your trading style. Cancel anytime.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 lg:gap-8 max-w-5xl mx-auto">
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
        <section className="py-24 lg:py-32 gradient-subtle">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="max-w-2xl mx-auto">
              <h2 className="text-3xl lg:text-4xl font-bold mb-6">Ready to Trade Smarter?</h2>
              <p className="text-muted-foreground mb-10 text-lg leading-relaxed">
                Join traders who've stopped gambling and started making informed decisions.
              </p>
              <Button 
                size="lg" 
                onClick={() => navigate('/auth')} 
                className="text-base px-10 h-14 rounded-full shadow-primary-glow hover:shadow-lg transition-all"
              >
                Get Started Now
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 lg:py-16 border-t border-border">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary rounded-xl">
                  <Activity className="h-4 w-4 text-primary-foreground" />
                </div>
                <span className="font-semibold">ForexTell AI</span>
              </div>
              <div className="flex gap-8 text-sm text-muted-foreground">
                <Link to="/track-record" className="hover:text-foreground transition-colors">
                  Track Record
                </Link>
                <a href="#pricing" className="hover:text-foreground transition-colors">
                  Pricing
                </a>
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