import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN");
const TELEGRAM_CHANNEL_ID = Deno.env.get("TELEGRAM_CHANNEL_ID");

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface OpportunityNotification {
  type: 'signal';
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
  symbol?: string;
  reasoning?: string;
  learning_summary?: string;
  patterns_detected?: string[];
}

type TelegramNotification = OpportunityNotification | OutcomeNotification;

function formatSignalMessage(opportunity: OpportunityNotification): string {
  const emoji = opportunity.signal_type === 'BUY' ? '🟢' : '🔴';
  const arrow = opportunity.signal_type === 'BUY' ? '⬆️' : '⬇️';
  
  let reversalHeader = '';
  if (opportunity.is_reversal && opportunity.previous_signal) {
    const prevEmoji = opportunity.previous_signal.signal_type === 'BUY' ? '🟢' : '🔴';
    reversalHeader = `⚠️ *SIGNAL REVERSAL*

Previous ${prevEmoji} ${opportunity.previous_signal.signal_type} signal (${opportunity.previous_signal.confidence.toFixed(0)}%) has been superseded.

`;
  }
  
  // Support symbol in signal notifications too
  const symbol = (opportunity as any).symbol || 'EUR/USD';
  
  return `
${reversalHeader}${emoji} *NEW ${opportunity.signal_type} SIGNAL* ${arrow}

📊 *${symbol}*
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

function formatOutcomeMessage(notification: OutcomeNotification): string {
  const { 
    signal_type, outcome, confidence, entry_price, outcome_price, created_at,
    symbol = 'EUR/USD', reasoning, learning_summary, patterns_detected 
  } = notification;
  
  const pipsMove = Math.abs(outcome_price - entry_price) * 10000;
  const isProfit = (signal_type === 'BUY' && outcome_price > entry_price) || 
                   (signal_type === 'SELL' && outcome_price < entry_price);
  
  // Calculate duration
  const startTime = new Date(created_at);
  const endTime = new Date();
  const durationMs = endTime.getTime() - startTime.getTime();
  const hours = Math.floor(durationMs / (1000 * 60 * 60));
  const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
  const durationStr = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  
  const signalEmoji = signal_type === 'BUY' ? '🟢' : '🔴';
  
  // Format patterns for display
  const patternsStr = patterns_detected && patterns_detected.length > 0 
    ? patterns_detected.map(p => p.replace(/_/g, ' ')).join(', ')
    : null;
  
  // Build analysis section if reasoning/learning available
  const analysisSection = (reasoning || learning_summary) ? `
📝 *Analysis:*
${reasoning ? reasoning.split('\n')[0].slice(0, 150) : ''}

🧠 *AI Learning:*
${learning_summary ? learning_summary.slice(0, 200) : 'Insight recorded for future signals'}
${patternsStr ? `\n🔍 Patterns: ${patternsStr}` : ''}` : '';
  
  if (outcome === 'WIN') {
    return `
✅ *SIGNAL RESULT: WIN* 🎯

📊 ${symbol} ${signalEmoji} ${signal_type}
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
🎯 TP Hit: ${outcome_price.toFixed(5)}
📈 +${pipsMove.toFixed(1)} pips profit
${analysisSection}
⏱️ Duration: ${durationStr}
⏰ ${new Date().toLocaleString('en-US', { timeZone: 'UTC' })} UTC

_ForexTell AI - Trade closed successfully_
    `.trim();
  } else if (outcome === 'LOSS') {
    return `
❌ *SIGNAL RESULT: LOSS* 🛑

📊 ${symbol} ${signalEmoji} ${signal_type}
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
🛑 SL Hit: ${outcome_price.toFixed(5)}
📉 -${pipsMove.toFixed(1)} pips loss
${analysisSection}
⏱️ Duration: ${durationStr}
⏰ ${new Date().toLocaleString('en-US', { timeZone: 'UTC' })} UTC

_ForexTell AI - Stop loss triggered_
    `.trim();
  } else {
    // EXPIRED
    const pipsDisplay = isProfit ? `+${pipsMove.toFixed(1)}` : `-${pipsMove.toFixed(1)}`;
    return `
⏳ *SIGNAL EXPIRED*

📊 ${symbol} ${signalEmoji} ${signal_type}
💯 Confidence: ${confidence.toFixed(0)}%

📍 Entry: ${entry_price.toFixed(5)}
📍 Exit: ${outcome_price.toFixed(5)}
${isProfit ? '📈' : '📉'} ${pipsDisplay} pips (unrealized)
${analysisSection}
⏱️ Duration: ${durationStr} (expired)
⏰ ${new Date().toLocaleString('en-US', { timeZone: 'UTC' })} UTC

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

    const payload: TelegramNotification = await req.json();
    
    let message: string;
    
    // Determine notification type and format message accordingly
    if (payload.type === 'outcome') {
      console.log("Sending Telegram outcome notification:", payload.outcome, payload.signal_type);
      message = formatOutcomeMessage(payload as OutcomeNotification);
    } else {
      // Default to signal notification (for backwards compatibility)
      const signalPayload = payload as OpportunityNotification;
      console.log("Sending Telegram signal notification:", signalPayload.signal_type, signalPayload.is_reversal ? "(REVERSAL)" : "");
      message = formatSignalMessage(signalPayload);
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
