/**
 * Monte Carlo Stress Test Function
 * 
 * Takes the last 100 trades, shuffles their order 10,000 times,
 * and simulates 10,000 different "equity paths".
 * 
 * Output:
 * - Probability of Ruin (chance of account hitting zero)
 * - 95% Confidence Interval for drawdown
 * - Final equity distribution
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface Trade {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL';
  entry_price: number;
  exit_price: number;
  pnl_pips: number;
  outcome: 'WIN' | 'LOSS';
  created_at: string;
  closed_at: string;
}

interface MonteCarloResult {
  probabilityOfRuin: number;
  drawdown95CI: { lower: number; upper: number };
  finalEquity95CI: { lower: number; upper: number };
  medianFinalEquity: number;
  worstCaseEquity: number;
  bestCaseEquity: number;
  simulationsRun: number;
  tradesAnalyzed: number;
  avgReturn: number;
  volatility: number;
}

function runMonteCarloSimulation(
  trades: Trade[],
  simulations: number = 10000,
  startingCapital: number = 10000
): MonteCarloResult {
  if (trades.length < 10) {
    return {
      probabilityOfRuin: 0,
      drawdown95CI: { lower: 0, upper: 0 },
      finalEquity95CI: { lower: startingCapital, upper: startingCapital },
      medianFinalEquity: startingCapital,
      worstCaseEquity: startingCapital,
      bestCaseEquity: startingCapital,
      simulationsRun: 0,
      tradesAnalyzed: trades.length,
      avgReturn: 0,
      volatility: 0,
    };
  }

  const results: { finalEquity: number; maxDrawdown: number; ruined: boolean }[] = [];
  const pnls = trades.map(t => t.pnl_pips);

  for (let sim = 0; sim < simulations; sim++) {
    // Fisher-Yates shuffle
    const shuffledPnls = [...pnls];
    for (let i = shuffledPnls.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffledPnls[i], shuffledPnls[j]] = [shuffledPnls[j], shuffledPnls[i]];
    }

    // Simulate equity path
    let equity = startingCapital;
    let peak = startingCapital;
    let maxDrawdown = 0;
    let ruined = false;

    for (const pnl of shuffledPnls) {
      equity += pnl;

      if (equity <= 0) {
        ruined = true;
        equity = 0;
        break;
      }

      peak = Math.max(peak, equity);
      const currentDrawdown = peak > 0 ? (peak - equity) / peak : 0;
      maxDrawdown = Math.max(maxDrawdown, currentDrawdown);
    }

    results.push({
      finalEquity: equity,
      maxDrawdown,
      ruined,
    });
  }

  // Calculate statistics
  const ruinCount = results.filter(r => r.ruined).length;
  const probabilityOfRuin = (ruinCount / simulations) * 100;

  // Sort results for percentiles
  const sortedEquities = results.map(r => r.finalEquity).sort((a, b) => a - b);
  const sortedDrawdowns = results.map(r => r.maxDrawdown).sort((a, b) => a - b);

  const percentile = (arr: number[], p: number) => {
    const index = Math.floor(arr.length * p);
    return arr[Math.min(index, arr.length - 1)];
  };

  // Calculate average return and volatility
  const avgReturn = pnls.reduce((a, b) => a + b, 0) / pnls.length;
  const variance = pnls.reduce((sum, pnl) => sum + Math.pow(pnl - avgReturn, 2), 0) / pnls.length;
  const volatility = Math.sqrt(variance);

  return {
    probabilityOfRuin,
    drawdown95CI: {
      lower: percentile(sortedDrawdowns, 0.025) * 100,
      upper: percentile(sortedDrawdowns, 0.975) * 100,
    },
    finalEquity95CI: {
      lower: percentile(sortedEquities, 0.025),
      upper: percentile(sortedEquities, 0.975),
    },
    medianFinalEquity: percentile(sortedEquities, 0.5),
    worstCaseEquity: sortedEquities[0],
    bestCaseEquity: sortedEquities[sortedEquities.length - 1],
    simulationsRun: simulations,
    tradesAnalyzed: trades.length,
    avgReturn,
    volatility,
  };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Parse request body
    const body = await req.json().catch(() => ({}));
    const {
      tradeCount = 100,
      simulations = 10000,
      startingCapital = 10000,
      symbol,
    } = body;

    console.log(`Running Monte Carlo stress test: ${simulations} simulations on last ${tradeCount} trades`);

    // Fetch completed trades
    let query = supabase
      .from('trading_opportunities')
      .select('id, symbol, signal_type, entry_price, stop_loss, take_profit_1, outcome, created_at, triggered_at')
      .in('outcome', ['WIN', 'LOSS'])
      .order('created_at', { ascending: false })
      .limit(tradeCount);

    if (symbol) {
      query = query.eq('symbol', symbol);
    }

    const { data: opportunities, error: fetchError } = await query;

    if (fetchError) {
      throw new Error(`Failed to fetch trades: ${fetchError.message}`);
    }

    if (!opportunities || opportunities.length === 0) {
      return new Response(
        JSON.stringify({
          success: true,
          message: "No completed trades found for analysis",
          result: null,
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Convert opportunities to trade format with P&L
    const trades: Trade[] = opportunities.map(opp => {
      const isWin = opp.outcome === 'WIN';
      const direction = opp.signal_type as 'BUY' | 'SELL';
      const entryPrice = opp.entry_price;
      const exitPrice = isWin ? opp.take_profit_1 : opp.stop_loss;

      // Calculate P&L in pips
      let pnlPips = 0;
      const pipValue = opp.symbol.includes('JPY') ? 0.01 : 0.0001;

      if (direction === 'BUY') {
        pnlPips = (exitPrice - entryPrice) / pipValue;
      } else {
        pnlPips = (entryPrice - exitPrice) / pipValue;
      }

      return {
        id: opp.id,
        symbol: opp.symbol,
        direction,
        entry_price: entryPrice,
        exit_price: exitPrice,
        pnl_pips: pnlPips,
        outcome: opp.outcome as 'WIN' | 'LOSS',
        created_at: opp.created_at,
        closed_at: opp.triggered_at || opp.created_at,
      };
    });

    // Run Monte Carlo simulation
    const result = runMonteCarloSimulation(trades, simulations, startingCapital);

    // Generate risk assessment
    const riskAssessment = [];
    
    if (result.probabilityOfRuin > 5) {
      riskAssessment.push({
        level: 'CRITICAL',
        message: `High probability of ruin: ${result.probabilityOfRuin.toFixed(2)}%`,
        action: 'Reduce position sizes immediately',
      });
    } else if (result.probabilityOfRuin > 1) {
      riskAssessment.push({
        level: 'WARNING',
        message: `Elevated probability of ruin: ${result.probabilityOfRuin.toFixed(2)}%`,
        action: 'Consider reducing risk per trade',
      });
    }

    if (result.drawdown95CI.upper > 30) {
      riskAssessment.push({
        level: 'WARNING',
        message: `95% CI for max drawdown is ${result.drawdown95CI.upper.toFixed(1)}%`,
        action: 'Implement position sizing rules to limit drawdown',
      });
    }

    const winRate = (trades.filter(t => t.outcome === 'WIN').length / trades.length) * 100;

    return new Response(
      JSON.stringify({
        success: true,
        message: `Monte Carlo analysis complete: ${simulations} simulations on ${trades.length} trades`,
        result: {
          ...result,
          interpretation: {
            probabilityOfRuin: `${result.probabilityOfRuin.toFixed(2)}% chance of account reaching zero`,
            maxDrawdown: `95% confident max drawdown will be between ${result.drawdown95CI.lower.toFixed(1)}% and ${result.drawdown95CI.upper.toFixed(1)}%`,
            finalEquity: `95% confident final equity will be between $${result.finalEquity95CI.lower.toFixed(0)} and $${result.finalEquity95CI.upper.toFixed(0)}`,
            expectedOutcome: `Median final equity: $${result.medianFinalEquity.toFixed(0)}`,
          },
          tradeStats: {
            totalTrades: trades.length,
            winRate: winRate.toFixed(1) + '%',
            avgReturnPips: result.avgReturn.toFixed(2),
            volatilityPips: result.volatility.toFixed(2),
          },
          riskAssessment,
        },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error) {
    console.error("Stress test error:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
