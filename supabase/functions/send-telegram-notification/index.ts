import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN");
const TELEGRAM_CHANNEL_ID = Deno.env.get("TELEGRAM_CHANNEL_ID");

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface OpportunityNotification {
  signal_type: 'BUY' | 'SELL';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit_1: number;
  take_profit_2: number;
  reasoning: string;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHANNEL_ID) {
      console.error("Telegram credentials not configured");
      return new Response(
        JSON.stringify({ error: "Telegram not configured" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const opportunity: OpportunityNotification = await req.json();
    console.log("Sending Telegram notification for:", opportunity.signal_type);
    
    const emoji = opportunity.signal_type === 'BUY' ? '🟢' : '🔴';
    const arrow = opportunity.signal_type === 'BUY' ? '⬆️' : '⬇️';
    
    const message = `
${emoji} *NEW ${opportunity.signal_type} SIGNAL* ${arrow}

📊 *EUR/USD*
💯 Confidence: ${opportunity.confidence.toFixed(0)}%

📍 Entry: ${opportunity.entry_price.toFixed(5)}
🛑 Stop Loss: ${opportunity.stop_loss.toFixed(5)}
🎯 TP1: ${opportunity.take_profit_1.toFixed(5)}
🎯 TP2: ${opportunity.take_profit_2.toFixed(5)}

📝 ${opportunity.reasoning?.split('\n')[0] || 'Technical signal detected'}

⏰ ${new Date().toLocaleString('en-US', { timeZone: 'UTC' })} UTC

_ForexTell AI - Not financial advice_
    `.trim();

    const response = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHANNEL_ID,
          text: message,
          parse_mode: 'Markdown',
        }),
      }
    );

    const result = await response.json();
    console.log("Telegram API response:", result);
    
    if (!result.ok) {
      throw new Error(result.description || 'Telegram API error');
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error: any) {
    console.error('Telegram notification error:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
