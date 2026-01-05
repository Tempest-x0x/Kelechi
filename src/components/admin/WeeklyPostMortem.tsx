import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon, Copy, Loader2, FileText } from "lucide-react";
import { format, startOfWeek, endOfWeek, subWeeks } from "date-fns";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { CandlestickChart } from "@/components/trading/CandlestickChart";
import { Candle } from "@/types/trading";

const WeeklyPostMortem = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [dateRange, setDateRange] = useState<{ from: Date; to?: Date }>({
    from: startOfWeek(subWeeks(new Date(), 1), { weekStartsOn: 1 }),
    to: endOfWeek(subWeeks(new Date(), 1), { weekStartsOn: 1 }),
  });
  
  const [weeklyReview, setWeeklyReview] = useState("");
  const [weekAheadOutlook, setWeekAheadOutlook] = useState("");
  const [detailedAnalysis, setDetailedAnalysis] = useState("");
  const [priceData, setPriceData] = useState<Candle[]>([]);
  const [keyLevels, setKeyLevels] = useState<{ highest?: number; lowest?: number }>({});

  const generateAnalysis = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke("generate-weekly-analysis", {
        body: {
          startDate: format(dateRange.from, "yyyy-MM-dd"),
          endDate: format(dateRange.to, "yyyy-MM-dd"),
        },
      });

      if (error) throw error;

      setWeeklyReview(data.weeklyReview || "");
      setWeekAheadOutlook(data.weekAheadOutlook || "");
      setDetailedAnalysis(data.detailedAnalysis || "");
      setPriceData(
        data.priceData?.map((p: any) => ({
          timestamp: p.timestamp,
          open: Number(p.open),
          high: Number(p.high),
          low: Number(p.low),
          close: Number(p.close),
        })) || []
      );
      setKeyLevels(data.keyLevels || {});

      toast({
        title: "Analysis Generated",
        description: "Weekly post-mortem is ready for review",
      });
    } catch (error) {
      console.error("Error generating analysis:", error);
      toast({
        title: "Error",
        description: "Failed to generate analysis. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: `${label} copied to clipboard`,
    });
  };

  return (
    <div className="space-y-6">
      {/* Date Range Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Weekly Post-Mortem Generator
          </CardTitle>
          <CardDescription>
            Generate AI-powered weekly analysis for X posts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-4 items-end">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Date Range</label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-[280px] justify-start text-left font-normal",
                      !dateRange && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {dateRange?.from ? (
                      dateRange.to ? (
                        <>
                          {format(dateRange.from, "LLL dd")} - {format(dateRange.to, "LLL dd, y")}
                        </>
                      ) : (
                        format(dateRange.from, "LLL dd, y")
                      )
                    ) : (
                      <span>Pick a date range</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    initialFocus
                    mode="range"
                    defaultMonth={dateRange?.from}
                    selected={dateRange}
                    onSelect={(range) => {
                      if (range?.from) {
                        setDateRange({ from: range.from, to: range.to });
                      }
                    }}
                    numberOfMonths={2}
                    className="pointer-events-auto"
                  />
                </PopoverContent>
              </Popover>
            </div>
            <Button onClick={generateAnalysis} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Analysis"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Weekly Price Chart */}
      {priceData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Weekly Price Action</CardTitle>
            <CardDescription>
              EUR/USD from {format(dateRange.from, "MMM dd")} to {dateRange.to ? format(dateRange.to, "MMM dd, yyyy") : ""}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CandlestickChart
              candles={priceData}
              height={350}
              takeProfit1={keyLevels.highest}
              stopLoss={keyLevels.lowest}
            />
          </CardContent>
        </Card>
      )}

      {/* Weekly Review Post */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Week in Review (X Post)</CardTitle>
              <CardDescription>Post this to summarize last week's performance</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(weeklyReview, "Weekly Review")}
              disabled={!weeklyReview}
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Textarea
            value={weeklyReview}
            onChange={(e) => setWeeklyReview(e.target.value)}
            placeholder="Weekly review will appear here..."
            className="min-h-[200px] font-mono text-sm"
          />
        </CardContent>
      </Card>

      {/* Week Ahead Outlook Post */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Week Ahead Outlook (X Post)</CardTitle>
              <CardDescription>Post this for next week's outlook</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(weekAheadOutlook, "Week Ahead Outlook")}
              disabled={!weekAheadOutlook}
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Textarea
            value={weekAheadOutlook}
            onChange={(e) => setWeekAheadOutlook(e.target.value)}
            placeholder="Week ahead outlook will appear here..."
            className="min-h-[200px] font-mono text-sm"
          />
        </CardContent>
      </Card>

      {/* Detailed Analysis */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Detailed Analysis (Internal)</CardTitle>
              <CardDescription>Full breakdown for your reference</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(detailedAnalysis, "Detailed Analysis")}
              disabled={!detailedAnalysis}
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Textarea
            value={detailedAnalysis}
            onChange={(e) => setDetailedAnalysis(e.target.value)}
            placeholder="Detailed analysis will appear here..."
            className="min-h-[300px] font-mono text-sm"
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default WeeklyPostMortem;
