import { useState, useEffect } from 'react';
import { ThemeProvider } from 'next-themes';
import { Link } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Activity, 
  ArrowLeft, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  CheckCircle2,
  XCircle,
  Clock,
  Moon,
  Sun
} from 'lucide-react';
import { useTheme } from 'next-themes';
import { format } from 'date-fns';

interface Prediction {
  id: string;
  created_at: string;
  signal_type: string;
  confidence: number;
  entry_price: number;
  stop_loss: number | null;
  take_profit_1: number | null;
  outcome: string | null;
  outcome_price: number | null;
}

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

const TrackRecord = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    wins: 0,
    losses: 0,
    pending: 0,
    winRate: 0
  });

  useEffect(() => {
    const fetchPredictions = async () => {
      const { data, error } = await supabase
        .from('predictions')
        .select('id, created_at, signal_type, confidence, entry_price, stop_loss, take_profit_1, outcome, outcome_price')
        .order('created_at', { ascending: false })
        .limit(100);

      if (!error && data) {
        setPredictions(data);
        
        const wins = data.filter(p => p.outcome === 'WIN').length;
        const losses = data.filter(p => p.outcome === 'LOSS').length;
        const pending = data.filter(p => !p.outcome).length;
        const decided = wins + losses;
        
        setStats({
          total: data.length,
          wins,
          losses,
          pending,
          winRate: decided > 0 ? Math.round((wins / decided) * 100) : 0
        });
      }
      
      setIsLoading(false);
    };

    fetchPredictions();
  }, []);

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'BUY':
        return <TrendingUp className="h-4 w-4" />;
      case 'SELL':
        return <TrendingDown className="h-4 w-4" />;
      default:
        return <Minus className="h-4 w-4" />;
    }
  };

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'BUY':
        return 'bg-bullish text-bullish-foreground';
      case 'SELL':
        return 'bg-bearish text-bearish-foreground';
      default:
        return 'bg-neutral text-neutral-foreground';
    }
  };

  const getOutcomeIcon = (outcome: string | null) => {
    switch (outcome) {
      case 'WIN':
        return <CheckCircle2 className="h-5 w-5 text-bullish" />;
      case 'LOSS':
        return <XCircle className="h-5 w-5 text-bearish" />;
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />;
    }
  };

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back
                </Button>
              </Link>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary rounded-lg">
                  <Activity className="h-5 w-5 text-primary-foreground" />
                </div>
                <div>
                  <h1 className="font-semibold text-lg leading-none">Track Record</h1>
                  <p className="text-xs text-muted-foreground">Public Performance History</p>
                </div>
              </div>
            </div>

            <ThemeToggle />
          </div>
        </header>

        <main className="container mx-auto px-4 py-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{stats.total}</div>
                <div className="text-sm text-muted-foreground">Total Predictions</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-bullish">{stats.wins}</div>
                <div className="text-sm text-muted-foreground">Wins</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-bearish">{stats.losses}</div>
                <div className="text-sm text-muted-foreground">Losses</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-muted-foreground">{stats.pending}</div>
                <div className="text-sm text-muted-foreground">Pending</div>
              </CardContent>
            </Card>
            <Card className="col-span-2 md:col-span-1">
              <CardContent className="pt-6">
                <div className={`text-2xl font-bold ${stats.winRate >= 50 ? 'text-bullish' : 'text-bearish'}`}>
                  {stats.winRate}%
                </div>
                <div className="text-sm text-muted-foreground">Win Rate</div>
              </CardContent>
            </Card>
          </div>

          {/* Predictions Table */}
          <Card>
            <CardHeader>
              <CardTitle>All Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : predictions.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">No predictions yet</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Date</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Bias</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Confidence</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Entry</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Invalidation</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Target</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((prediction) => (
                        <tr key={prediction.id} className="border-b border-border/50 hover:bg-secondary/30">
                          <td className="py-3 px-2 text-sm">
                            {format(new Date(prediction.created_at), 'MMM d, yyyy HH:mm')}
                          </td>
                          <td className="py-3 px-2">
                            <Badge className={getSignalColor(prediction.signal_type)}>
                              {getSignalIcon(prediction.signal_type)}
                              <span className="ml-1">{prediction.signal_type}</span>
                            </Badge>
                          </td>
                          <td className="py-3 px-2 text-sm font-mono">
                            {prediction.confidence}%
                          </td>
                          <td className="py-3 px-2 text-sm font-mono">
                            {prediction.entry_price?.toFixed(5)}
                          </td>
                          <td className="py-3 px-2 text-sm font-mono text-bearish">
                            {prediction.stop_loss?.toFixed(5) || '-'}
                          </td>
                          <td className="py-3 px-2 text-sm font-mono text-bullish">
                            {prediction.take_profit_1?.toFixed(5) || '-'}
                          </td>
                          <td className="py-3 px-2">
                            <div className="flex items-center gap-2">
                              {getOutcomeIcon(prediction.outcome)}
                              <span className="text-sm">
                                {prediction.outcome || 'Pending'}
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Disclaimer */}
          <p className="text-center text-sm text-muted-foreground mt-8">
            Past performance does not guarantee future results. This is not financial advice.
          </p>
        </main>
      </div>
    </ThemeProvider>
  );
};

export default TrackRecord;
