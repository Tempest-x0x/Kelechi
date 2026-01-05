import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Copy, Loader2, Sun, Moon, TrendingUp, TrendingDown } from "lucide-react";
import { format } from "date-fns";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { CandlestickChart } from "@/components/trading/CandlestickChart";
import { Candle } from "@/types/trading";

const DailyBiasPanel = () => {
  const { toast } = useToast();
  const [loadingMorning, setLoadingMorning] = useState(false);
  const [loadingEvening, setLoadingEvening] = useState(false);
  
  const [morningBias, setMorningBias] = useState("");
  const [eveningRecap, setEveningRecap] = useState("");
  const [biasDirection, setBiasDirection] = useState<"BULLISH" | "BEARISH" | "NEUTRAL" | null>(null);
  const [priceData, setPriceData] = useState<Candle[]>([]);
  const [keyLevels, setKeyLevels] = useState<{ high?: number; low?: number; resistance?: number; support?: number }>({});

  const generateMorningBias = async () => {
    setLoadingMorning(true);
    try {
      const { data, error } = await supabase.functions.invoke("generate-daily-bias", {
        body: {
          type: "morning",
          date: format(new Date(), "yyyy-MM-dd"),
        },
      });

      if (error) throw error;

      setMorningBias(data.post || "");
      setBiasDirection(data.direction || null);
      
      console.log("Morning Bias Response:", data);
      console.log("Price Data received:", data.priceData?.length || 0, "candles");
      
      if (data.priceData && data.priceData.length > 0) {
        setPriceData(
          data.priceData.map((p: any) => ({
            timestamp: p.timestamp,
            open: Number(p.open),
            high: Number(p.high),
            low: Number(p.low),
            close: Number(p.close),
          }))
        );
        setKeyLevels(data.keyLevels || {});
      } else {
        console.warn("No price data returned from generate-daily-bias");
      }

      toast({
        title: "Morning Bias Generated",
        description: `Direction: ${data.direction}`,
      });
    } catch (error) {
      console.error("Error generating morning bias:", error);
      toast({
        title: "Error",
        description: "Failed to generate morning bias. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoadingMorning(false);
    }
  };

  const generateEveningRecap = async () => {
    setLoadingEvening(true);
    try {
      const { data, error } = await supabase.functions.invoke("generate-daily-bias", {
        body: {
          type: "evening",
          date: format(new Date(), "yyyy-MM-dd"),
        },
      });

      if (error) throw error;

      setEveningRecap(data.post || "");
      
      console.log("Evening Recap Response:", data);
      console.log("Price Data received:", data.priceData?.length || 0, "candles");
      
      if (data.priceData && data.priceData.length > 0) {
        setPriceData(
          data.priceData.map((p: any) => ({
            timestamp: p.timestamp,
            open: Number(p.open),
            high: Number(p.high),
            low: Number(p.low),
            close: Number(p.close),
          }))
        );
        setKeyLevels(data.keyLevels || {});
      } else {
        console.warn("No price data returned from generate-daily-bias");
      }

      toast({
        title: "Evening Recap Generated",
        description: "Recap is ready for review",
      });
    } catch (error) {
      console.error("Error generating evening recap:", error);
      toast({
        title: "Error",
        description: "Failed to generate evening recap. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoadingEvening(false);
    }
  };

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: `${label} copied to clipboard`,
    });
  };

  const getBiasIcon = () => {
    if (biasDirection === "BULLISH") return <TrendingUp className="h-5 w-5 text-green-500" />;
    if (biasDirection === "BEARISH") return <TrendingDown className="h-5 w-5 text-red-500" />;
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Current Date Display */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Daily Bias Generator
            {getBiasIcon()}
          </CardTitle>
          <CardDescription>
            {format(new Date(), "EEEE, MMMM d, yyyy")}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button onClick={generateMorningBias} disabled={loadingMorning}>
              {loadingMorning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sun className="mr-2 h-4 w-4" />
                  Generate Morning Bias
                </>
              )}
            </Button>
            <Button onClick={generateEveningRecap} disabled={loadingEvening} variant="outline">
              {loadingEvening ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Moon className="mr-2 h-4 w-4" />
                  Generate Evening Recap
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Price Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Daily Price Action (48H)</CardTitle>
          <CardDescription>
            EUR/USD for {format(new Date(), "MMMM d, yyyy")}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {priceData.length > 0 ? (
            <CandlestickChart
              candles={priceData}
              height={300}
              takeProfit1={keyLevels.resistance}
              stopLoss={keyLevels.support}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground border border-dashed rounded-lg">
              {loadingMorning || loadingEvening 
                ? "Loading chart data..." 
                : "Generate morning bias or evening recap to see the chart"}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Morning Bias Post */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Sun className="h-5 w-5 text-yellow-500" />
                Morning Bias (X Post)
              </CardTitle>
              <CardDescription>Directional bias with invalidation levels</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(morningBias, "Morning Bias")}
              disabled={!morningBias}
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Textarea
            value={morningBias}
            onChange={(e) => setMorningBias(e.target.value)}
            placeholder="Morning bias will appear here...

Example format:
EUR/USD Daily Bias - Jan 6, 2026

Direction: BEARISH
Confidence: 70%

Key Levels:
• Invalidation: 1.1760 (bias flip if broken)
• Target 1: 1.1720
• Target 2: 1.1700

Watch for: Rejection at 200 EMA (1.1756)

Updates at market close."
            className="min-h-[280px] font-mono text-sm"
          />
        </CardContent>
      </Card>

      {/* Evening Recap Post */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Moon className="h-5 w-5 text-blue-400" />
                Evening Recap (X Post)
              </CardTitle>
              <CardDescription>End of day summary and tomorrow's outlook</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(eveningRecap, "Evening Recap")}
              disabled={!eveningRecap}
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Textarea
            value={eveningRecap}
            onChange={(e) => setEveningRecap(e.target.value)}
            placeholder="Evening recap will appear here...

Example format:
EUR/USD Recap - Jan 6, 2026

Bias: BEARISH ✅ Called it right
High: 1.1752 (within 8 pips of invalidation)
Low: 1.1718 (hit Target 1)

Tomorrow: Watching for continuation

Follow for daily EUR/USD analysis."
            className="min-h-[220px] font-mono text-sm"
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default DailyBiasPanel;
