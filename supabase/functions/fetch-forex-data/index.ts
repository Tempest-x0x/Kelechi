import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

type Timeframe = "1min" | "5min" | "15min" | "30min" | "1h" | "4h" | "1d" | "15m";

type CachedResponse = {
  success: true;
  symbol: string;
  currentPrice: number;
  candles: Array<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number | null;
  }>;
  candleCount: number;
  meta: { source: "cache"; warning?: string };
};

function normalizeInterval(timeframe: string): { interval: string; normalizedTimeframe: string } {
  const intervalMap: Record<string, { interval: string; normalizedTimeframe: string }> = {
    "1min": { interval: "1min", normalizedTimeframe: "1min" },
    "5min": { interval: "5min", normalizedTimeframe: "5min" },
    "15min": { interval: "15min", normalizedTimeframe: "15min" },
    "30min": { interval: "30min", normalizedTimeframe: "30min" },
    "1h": { interval: "1h", normalizedTimeframe: "1h" },
    "4h": { interval: "4h", normalizedTimeframe: "4h" },
    "1d": { interval: "1day", normalizedTimeframe: "1d" },
    // Back-compat
    "15m": { interval: "15min", normalizedTimeframe: "15min" },
  };

  return intervalMap[timeframe] || { interval: "1h", normalizedTimeframe: "1h" };
}

function defaultOutputsizeFor(timeframe: string): number {
  const defaults: Record<string, number> = {
    "1min": 180,
    "5min": 150,
    "15min": 150,
    "30min": 150,
    "1h": 200,
    "4h": 200,
    "1d": 200,
  };
  return defaults[timeframe] ?? 200;
}

function clampOutputsize(value: unknown, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  const raw = Number.isFinite(n) ? n : fallback;
  // TwelveData free-tier burns credits quickly; keep it reasonable.
  return Math.min(Math.max(Math.floor(raw), 60), 300);
}

function isTwelveDataQuotaError(msg: string) {
  const m = msg.toLowerCase();
  return (
    m.includes("run out of api credits") ||
    m.includes("current limit") ||
    m.includes("credits were used") ||
    m.includes("rate limit") ||
    m.includes("too many requests")
  );
}

function cacheFreshnessMs(timeframe: string) {
  // If the latest cached candle is recent enough, return cache to avoid burning credits.
  const ms: Record<string, number> = {
    "1min": 20_000,
    "5min": 55_000,
    "15min": 110_000,
    "30min": 170_000,
    "1h": 600_000,
    "4h": 1_800_000,
    "1d": 7_200_000,
  };
  return ms[timeframe] ?? 600_000;
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
  const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const supabase = createClient(supabaseUrl, supabaseKey);

  const readCache = async (timeframe: string, warning?: string): Promise<CachedResponse | null> => {
    const { data, error } = await supabase
      .from("price_history")
      .select("timestamp, open, high, low, close, volume")
      .eq("symbol", "EUR/USD")
      .eq("timeframe", timeframe)
      .order("timestamp", { ascending: true })
      .limit(300);

    if (error) {
      console.error("Error reading cached price data:", error);
      return null;
    }

    if (!data || data.length === 0) return null;

    const candles = data.map((row) => ({
      timestamp: row.timestamp,
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume: row.volume === null ? null : Number(row.volume),
    }));

    const currentPrice = candles[candles.length - 1]?.close || 0;

    return {
      success: true,
      symbol: "EUR/USD",
      currentPrice,
      candles,
      candleCount: candles.length,
      meta: { source: "cache", warning },
    };
  };

  const getLatestCachedTimestamp = async (timeframe: string): Promise<Date | null> => {
    const { data, error } = await supabase
      .from("price_history")
      .select("timestamp")
      .eq("symbol", "EUR/USD")
      .eq("timeframe", timeframe)
      .order("timestamp", { ascending: false })
      .limit(1);

    if (error || !data || data.length === 0) return null;

    const ts = String((data[0] as any).timestamp);
    const d = new Date(ts);
    return Number.isFinite(d.getTime()) ? d : null;
  };

  try {
    const body = await req.json().catch(() => ({}));
    const timeframeRaw = (body?.timeframe ?? "1h") as Timeframe;
    const { interval, normalizedTimeframe } = normalizeInterval(timeframeRaw);

    const outputsize = clampOutputsize(body?.outputsize, defaultOutputsizeFor(normalizedTimeframe));

    // 1) Serve cache if it's fresh enough (prevents burning credits for repeated polling)
    const latestCached = await getLatestCachedTimestamp(normalizedTimeframe);
    if (latestCached) {
      const ageMs = Date.now() - latestCached.getTime();
      if (ageMs >= 0 && ageMs < cacheFreshnessMs(normalizedTimeframe)) {
        const cached = await readCache(normalizedTimeframe, "Serving cached data to reduce API usage");
        if (cached) {
          return new Response(JSON.stringify(cached), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      }
    }

    const TWELVE_DATA_API_KEY = Deno.env.get("TWELVE_DATA_API_KEY");
    if (!TWELVE_DATA_API_KEY) {
      // If API key missing, try cache anyway.
      const cached = await readCache(normalizedTimeframe, "API key missing; serving cache");
      if (cached) {
        return new Response(JSON.stringify(cached), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      return new Response(
        JSON.stringify({ success: false, error: "TWELVE_DATA_API_KEY is not configured" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    console.log(
      `Fetching EUR/USD data with timeframe: ${normalizedTimeframe} (interval ${interval}), outputsize: ${outputsize}`,
    );

    const url = `https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=${interval}&outputsize=${outputsize}&apikey=${TWELVE_DATA_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json().catch(() => ({}));

    if (!response.ok || data?.status === "error") {
      const message = String(data?.message || `Twelve Data request failed (status ${response.status})`);
      console.error("Twelve Data API error:", message);

      // 2) If quota/rate-limited, serve cached data instead of returning 500
      if (isTwelveDataQuotaError(message)) {
        const cached = (await readCache(normalizedTimeframe, message)) ?? (await readCache("1h", message));
        if (cached) {
          return new Response(JSON.stringify(cached), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        // No cache available at all: still return 200 to avoid "Edge function returned 500" UX.
        return new Response(
          JSON.stringify({
            success: false,
            error: message,
            quotaExceeded: true,
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } },
        );
      }

      // 3) Other API errors: attempt cache fallback, otherwise return 200 with error
      const cached = (await readCache(normalizedTimeframe, message)) ?? (await readCache("1h", message));
      if (cached) {
        return new Response(JSON.stringify(cached), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      return new Response(
        JSON.stringify({
          success: false,
          error: message,
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (!data?.values || !Array.isArray(data.values)) {
      console.error("Unexpected response format:", data);

      const cached = (await readCache(normalizedTimeframe, "Invalid API response format")) ??
        (await readCache("1h", "Invalid API response format"));
      if (cached) {
        return new Response(JSON.stringify(cached), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      return new Response(
        JSON.stringify({ success: false, error: "Invalid response format from Twelve Data" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    console.log(`Received ${data.values.length} candles from Twelve Data`);

    const candles = data.values
      .map((candle: any) => ({
        timestamp: candle.datetime,
        open: parseFloat(candle.open),
        high: parseFloat(candle.high),
        low: parseFloat(candle.low),
        close: parseFloat(candle.close),
        volume: candle.volume ? parseFloat(candle.volume) : null,
      }))
      .reverse();

    const currentPrice = candles[candles.length - 1]?.close || 0;

    // Cache last 120 candles in backend DB (avoid huge inserts)
    const priceHistoryData = candles.slice(-120).map((c: any) => ({
      symbol: "EUR/USD",
      timestamp: c.timestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume,
      timeframe: normalizedTimeframe,
    }));

    await supabase.from("price_history").delete().eq("symbol", "EUR/USD").eq("timeframe", normalizedTimeframe);

    const { error: insertError } = await supabase.from("price_history").insert(priceHistoryData);
    if (insertError) {
      console.error("Error caching price data:", insertError);
    }

    return new Response(
      JSON.stringify({
        success: true,
        symbol: "EUR/USD",
        currentPrice,
        candles,
        candleCount: candles.length,
        meta: { ...data.meta, source: "twelvedata" },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (error) {
    console.error("Error fetching forex data:", error);

    // Final attempt: return any cached data we have.
    const cached = (await readCache("1h", error instanceof Error ? error.message : "Unknown error")) ??
      (await readCache("4h", error instanceof Error ? error.message : "Unknown error"));
    if (cached) {
      return new Response(JSON.stringify(cached), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Return 200 to avoid crashing clients that treat non-2xx as a hard failure.
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
