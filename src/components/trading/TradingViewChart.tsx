import { useEffect, useRef, memo } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickSeries, CandlestickData, Time } from "lightweight-charts";
import { Candle } from "@/types/trading";

interface TradingViewChartProps {
  candles: Candle[];
  height?: number;
  entryPrice?: number;
  stopLoss?: number;
  takeProfit1?: number;
  takeProfit2?: number;
  currentPrice?: number;
}

const TradingViewChart = memo(({ 
  candles, 
  height = 400,
  entryPrice,
  stopLoss,
  takeProfit1,
  takeProfit2,
}: TradingViewChartProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    // Create chart
    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "hsl(var(--muted-foreground))",
      },
      grid: {
        vertLines: { color: "hsl(var(--border))" },
        horzLines: { color: "hsl(var(--border))" },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: "hsl(var(--muted-foreground))",
          width: 1,
          style: 2,
        },
        horzLine: {
          color: "hsl(var(--muted-foreground))",
          width: 1,
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: "hsl(var(--border))",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: "hsl(var(--border))",
        timeVisible: true,
        secondsVisible: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    seriesRef.current = candlestickSeries;

    // Transform candles data - deduplicate by timestamp and sort ascending
    const seenTimes = new Set<number>();
    const chartData: CandlestickData<Time>[] = candles
      .map((candle) => ({
        time: Math.floor(new Date(candle.timestamp).getTime() / 1000) as Time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }))
      .filter((item) => {
        const timeNum = item.time as number;
        if (seenTimes.has(timeNum)) {
          return false; // Skip duplicate
        }
        seenTimes.add(timeNum);
        return true;
      })
      .sort((a, b) => (a.time as number) - (b.time as number));

    if (chartData.length === 0) return;

    candlestickSeries.setData(chartData);

    // Add price lines for entry, SL, TP
    if (entryPrice) {
      candlestickSeries.createPriceLine({
        price: entryPrice,
        color: "#3b82f6",
        lineWidth: 2,
        lineStyle: 0,
        axisLabelVisible: true,
        title: "Entry",
      });
    }

    if (stopLoss) {
      candlestickSeries.createPriceLine({
        price: stopLoss,
        color: "#ef4444",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "SL",
      });
    }

    if (takeProfit1) {
      candlestickSeries.createPriceLine({
        price: takeProfit1,
        color: "#22c55e",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "TP1",
      });
    }

    if (takeProfit2) {
      candlestickSeries.createPriceLine({
        price: takeProfit2,
        color: "#16a34a",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "TP2",
      });
    }

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
          width: containerRef.current.clientWidth 
        });
      }
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [candles, height, entryPrice, stopLoss, takeProfit1, takeProfit2]);

  if (candles.length === 0) {
    return (
      <div 
        className="flex items-center justify-center text-muted-foreground border border-dashed rounded-lg"
        style={{ height }}
      >
        No chart data available
      </div>
    );
  }

  return <div ref={containerRef} style={{ height }} />;
});

TradingViewChart.displayName = "TradingViewChart";

export { TradingViewChart };
