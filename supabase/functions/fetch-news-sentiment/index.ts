import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Return disabled response - real news sentiment coming soon
  console.log("Sentiment analysis is disabled - returning coming soon response");
  
  return new Response(
    JSON.stringify({
      success: true,
      disabled: true,
      message: "Real news sentiment analysis coming soon. Currently integrating Alpha Vantage news API.",
      sentiment: null,
    }),
    { headers: { ...corsHeaders, "Content-Type": "application/json" } }
  );
});
