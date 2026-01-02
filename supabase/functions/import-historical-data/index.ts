import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface CandleInput {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { 
      symbol = "EUR/USD", 
      timeframe = "1h", 
      candles 
    }: { symbol: string; timeframe: string; candles: CandleInput[] } = body;

    if (!candles || !Array.isArray(candles) || candles.length === 0) {
      return new Response(
        JSON.stringify({ success: false, error: "No candles provided" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`Importing ${candles.length} candles for ${symbol} ${timeframe}`);

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Format candles for database
    const records = candles.map(c => ({
      symbol,
      timeframe,
      timestamp: c.timestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume || 0
    }));

    // Insert in smaller batches to avoid timeouts
    const BATCH_SIZE = 5000;
    let inserted = 0;
    let errors = 0;

    for (let i = 0; i < records.length; i += BATCH_SIZE) {
      const batch = records.slice(i, i + BATCH_SIZE);
      
      const { error } = await supabase
        .from('price_history')
        .upsert(batch, { 
          onConflict: 'symbol,timeframe,timestamp',
          ignoreDuplicates: false
        });

      if (error) {
        console.error(`Batch error at index ${i}:`, error.message);
        errors++;
      } else {
        inserted += batch.length;
        console.log(`Inserted batch ${Math.floor(i / BATCH_SIZE) + 1}, total: ${inserted}`);
      }
    }

    const result = {
      success: true,
      inserted,
      errors,
      symbol,
      timeframe,
      firstTimestamp: candles[0]?.timestamp,
      lastTimestamp: candles[candles.length - 1]?.timestamp
    };

    console.log(`Import complete: ${inserted} rows inserted, ${errors} errors`);

    return new Response(
      JSON.stringify(result),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Import error:", error);
    return new Response(
      JSON.stringify({ success: false, error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
