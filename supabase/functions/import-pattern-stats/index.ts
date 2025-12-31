import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { patterns, clearExisting = true } = await req.json();

    if (!patterns || !Array.isArray(patterns)) {
      throw new Error("Invalid data: 'patterns' must be an array");
    }

    console.log(`Importing ${patterns.length} pattern statistics...`);

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Clear existing data if requested
    if (clearExisting) {
      console.log("Clearing existing pattern statistics...");
      const { error: deleteError } = await supabase
        .from("pattern_statistics")
        .delete()
        .neq("id", "00000000-0000-0000-0000-000000000000"); // Delete all rows
      
      if (deleteError) {
        console.error("Error clearing data:", deleteError);
      }
    }

    // Transform and insert the data
    const insertData = patterns.map((p: any) => ({
      pattern_name: p.pattern,
      signal_type: p.signal,
      occurrences: p.occurrences,
      win_rate_4h: p.results["4_candles"]?.["win_rate_%"] || null,
      win_rate_12h: p.results["12_candles"]?.["win_rate_%"] || null,
      win_rate_24h: p.results["24_candles"]?.["win_rate_%"] || null,
      win_rate_48h: p.results["48_candles"]?.["win_rate_%"] || null,
      avg_pips_4h: p.results["4_candles"]?.avg_pips || null,
      avg_pips_12h: p.results["12_candles"]?.avg_pips || null,
      avg_pips_24h: p.results["24_candles"]?.avg_pips || null,
      avg_pips_48h: p.results["48_candles"]?.avg_pips || null,
      sample_size: p.occurrences,
      data_start_date: p.data_start_date || null,
      data_end_date: p.data_end_date || null,
    }));

    console.log("Inserting pattern statistics:", JSON.stringify(insertData, null, 2));

    const { data, error: insertError } = await supabase
      .from("pattern_statistics")
      .insert(insertData)
      .select();

    if (insertError) {
      console.error("Error inserting data:", insertError);
      throw new Error(`Failed to insert data: ${insertError.message}`);
    }

    console.log(`Successfully imported ${data?.length || 0} pattern statistics`);

    return new Response(
      JSON.stringify({
        success: true,
        message: `Imported ${data?.length || 0} pattern statistics`,
        data,
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
