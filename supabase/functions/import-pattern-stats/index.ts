import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface PatternInput {
  pattern_name: string;
  signal_type: string;
  occurrences: number;
  outcomes?: {
    "4_candles"?: { win_rate: number; avg_pips: number; sample_size: number };
    "12_candles"?: { win_rate: number; avg_pips: number; sample_size: number };
    "24_candles"?: { win_rate: number; avg_pips: number; sample_size: number };
    "48_candles"?: { win_rate: number; avg_pips: number; sample_size: number };
  };
  // Legacy format support
  win_rate_4h?: number;
  win_rate_12h?: number;
  win_rate_24h?: number;
  win_rate_48h?: number;
  avg_pips_4h?: number;
  avg_pips_12h?: number;
  avg_pips_24h?: number;
  avg_pips_48h?: number;
  sample_size?: number;
  data_start_date?: string;
  data_end_date?: string;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const { patterns, symbol = 'EUR/USD', clearExisting = false, clearSymbol = false } = body;

    if (!patterns || !Array.isArray(patterns)) {
      throw new Error("Invalid data: 'patterns' must be an array");
    }

    console.log(`Importing ${patterns.length} pattern statistics for ${symbol}...`);

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Clear existing data if requested
    if (clearExisting) {
      console.log("Clearing ALL existing pattern statistics...");
      const { error: deleteError } = await supabase
        .from("pattern_statistics")
        .delete()
        .neq("id", "00000000-0000-0000-0000-000000000000");
      
      if (deleteError) {
        console.error("Error clearing data:", deleteError);
      }
    } else if (clearSymbol) {
      // Only clear data for this specific symbol
      console.log(`Clearing existing pattern statistics for ${symbol}...`);
      const { error: deleteError } = await supabase
        .from("pattern_statistics")
        .delete()
        .eq("symbol", symbol);
      
      if (deleteError) {
        console.error("Error clearing symbol data:", deleteError);
      }
    }

    // Transform and insert the data - handle both new and legacy formats
    const insertData = patterns.map((p: PatternInput) => {
      // Check if using new nested outcomes format
      const hasOutcomes = p.outcomes && typeof p.outcomes === 'object';
      
      return {
        pattern_name: p.pattern_name,
        signal_type: p.signal_type,
        occurrences: p.occurrences,
        symbol: symbol,
        win_rate_4h: hasOutcomes ? p.outcomes?.["4_candles"]?.win_rate : p.win_rate_4h,
        win_rate_12h: hasOutcomes ? p.outcomes?.["12_candles"]?.win_rate : p.win_rate_12h,
        win_rate_24h: hasOutcomes ? p.outcomes?.["24_candles"]?.win_rate : p.win_rate_24h,
        win_rate_48h: hasOutcomes ? p.outcomes?.["48_candles"]?.win_rate : p.win_rate_48h,
        avg_pips_4h: hasOutcomes ? p.outcomes?.["4_candles"]?.avg_pips : p.avg_pips_4h,
        avg_pips_12h: hasOutcomes ? p.outcomes?.["12_candles"]?.avg_pips : p.avg_pips_12h,
        avg_pips_24h: hasOutcomes ? p.outcomes?.["24_candles"]?.avg_pips : p.avg_pips_24h,
        avg_pips_48h: hasOutcomes ? p.outcomes?.["48_candles"]?.avg_pips : p.avg_pips_48h,
        sample_size: hasOutcomes 
          ? p.outcomes?.["4_candles"]?.sample_size || p.occurrences 
          : p.sample_size || p.occurrences,
        data_start_date: p.data_start_date || null,
        data_end_date: p.data_end_date || null,
      };
    });

    console.log(`Prepared ${insertData.length} records for insertion`);

    // Use upsert with the unique constraint on (symbol, pattern_name, signal_type)
    const { data, error: insertError } = await supabase
      .from("pattern_statistics")
      .upsert(insertData, {
        onConflict: 'symbol,pattern_name,signal_type',
        ignoreDuplicates: false
      })
      .select();

    if (insertError) {
      console.error("Error inserting data:", insertError);
      throw new Error(`Failed to insert data: ${insertError.message}`);
    }

    console.log(`Successfully imported ${data?.length || 0} pattern statistics for ${symbol}`);

    return new Response(
      JSON.stringify({
        success: true,
        message: `Imported ${data?.length || 0} pattern statistics for ${symbol}`,
        symbol,
        count: data?.length || 0,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error importing pattern statistics:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
