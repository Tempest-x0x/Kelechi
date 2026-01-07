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
  is_reversal?: boolean;
  previous_signal?: {
    signal_type: string;
    confidence: number;
    created_at: string;
  };
}

interface OutcomeNotification {
  type: 'outcome';
  signal_type: 'BUY' | 'SELL';
  outcome: 'WIN' | 'LOSS' | 'EXPIRED';
  confidence: number;
  entry_price: number;
  outcome_price: number;
  stop_loss: number | null;
  take_profit_1: number | null;
  created_at: string;
}

function formatDuration(startDate: string): string {
  const start = new Date(startDate);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();
  
  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

function formatOutcomeMessage(notification: OutcomeNotification): string {
  const { signal_type, outcome, confidence, entry_price, outcome_price, created_at } = notification;
  
  const pipsMove = Math.abs(outcome_price - entry_price) * 10000;
  const isProfit = (signal_type === 'BUY' && outcome_price > entry_price) ||
                   (signal_type === 'SELL' && outcome_price < entry_price);
  
  const duration = formatDuration(created_at);
  const timestamp = new Date().toLocaleString('en-US', { timeZone: 'UTC' });
  
  if (outcome === 'WIN') {
    return `
✅ *SIGNAL RESULT: WIN* 🎯

📊 *EUR/USD ${signal_type}*
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
🎯 TP Hit: ${outcome_price.toFixed(5)}
📈 +${pipsMove.toFixed(1)} pips profit

⏱️ Duration: ${duration}
⏰ ${timestamp} UTC

_ForexTell AI - Trade closed successfully_
    `.trim();
  } else if (outcome === 'LOSS') {
    return `
❌ *SIGNAL RESULT: LOSS* 🛑

📊 *EUR/USD ${signal_type}*
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
🛑 SL Hit: ${outcome_price.toFixed(5)}
📉 -${pipsMove.toFixed(1)} pips loss

⏱️ Duration: ${duration}
⏰ ${timestamp} UTC

_ForexTell AI - Stop loss triggered_
    `.trim();
  } else {
    // EXPIRED
    const pipsDirection = isProfit ? '+' : '-';
    const pipsEmoji = isProfit ? '📈' : '📉';
    
    return `
⏳ *SIGNAL EXPIRED*

📊 *EUR/USD ${signal_type}*
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
📍 Exit: ${outcome_price.toFixed(5)}
${pipsEmoji} ${pipsDirection}${pipsMove.toFixed(1)} pips (unrealized)

⏱️ Duration: ${duration} (expired)
⏰ ${timestamp} UTC

_ForexTell AI - Signal expired without hitting SL/TP_
    `.trim();
  }
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

    const payload = await req.json();
    
    let message: string;
    
    // Check if this is an outcome notification
    if (payload.type === 'outcome') {
      const outcomeNotification = payload as OutcomeNotification;
      console.log("Sending Telegram outcome notification:", outcomeNotification.outcome, outcomeNotification.signal_type);
      message = formatOutcomeMessage(outcomeNotification);
    } else {
      // Original opportunity notification
      const opportunity = payload as OpportunityNotification;
      console.log("Sending Telegram notification for:", opportunity.signal_type, opportunity.is_reversal ? "(REVERSAL)" : "");
      
      const emoji = opportunity.signal_type === 'BUY' ? '🟢' : '🔴';
      const arrow = opportunity.signal_type === 'BUY' ? '⬆️' : '⬇️';
      
      // Build reversal header if applicable
      let reversalHeader = '';
      if (opportunity.is_reversal && opportunity.previous_signal) {
        const prevEmoji = opportunity.previous_signal.signal_type === 'BUY' ? '🟢' : '🔴';
        reversalHeader = `⚠️ *SIGNAL REVERSAL*

Previous ${prevEmoji} ${opportunity.previous_signal.signal_type} signal (${opportunity.previous_signal.confidence.toFixed(0)}%) has been superseded.

`;
      }
      
      message = `
${reversalHeader}${emoji} *NEW ${opportunity.signal_type} SIGNAL* ${arrow}

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
    }

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
