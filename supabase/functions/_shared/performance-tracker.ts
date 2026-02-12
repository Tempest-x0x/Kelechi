/**
 * Performance Tracker & Statistical Validation
 * 
 * Implements:
 * - Z-Score Performance Monitor (statistical significance)
 * - Expectancy (EV) Audit
 * - Slippage Guard
 * - Live metrics dashboard data
 */

import { PERFORMANCE_CONFIG, ICT_CONFIG, getPipValue, priceToPips } from './ict-config.ts';

// ============================================
// TYPES
// ============================================
export interface Trade {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  requestedEntryPrice: number;  // For slippage tracking
  actualFilledPrice: number;    // For slippage tracking
  stopLoss: number;
  takeProfit: number;
  outcome: 'WIN' | 'LOSS';
  pnlPips: number;
  pnlPercent: number;
  timestamp: string;
  closedAt: string;
}

export interface PerformanceMetrics {
  totalTrades: number;
  wins: number;
  losses: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  sharpeRatio: number;
  zScore: number;
  isStatisticallySignificant: boolean;
  confidenceWarning: boolean;
  performanceDegraded: boolean;
}

export interface SlippageReport {
  symbol: string;
  avgSlippagePips: number;
  maxSlippagePips: number;
  slippagePercent: number;  // As % of RRR target
  isPaused: boolean;
  trades: number;
}

export interface AuditReport {
  timestamp: string;
  period: string;
  metrics: PerformanceMetrics;
  slippageBySymbol: SlippageReport[];
  warnings: string[];
  recommendations: string[];
  isPaused: boolean;
}

// ============================================
// Z-SCORE CALCULATOR
// ============================================

/**
 * Calculate Z-Score of win/loss sequence
 * 
 * Formula: Z = (N * (R - 0.5) - P) / sqrt((P * (P - N)) / (N - 1))
 * Where:
 *   N = total trades
 *   R = number of streaks (runs)
 *   P = 2 * wins * losses
 * 
 * If Z-Score > 1.96, indicates significant "streaky" dependence
 */
export function calculateZScore(trades: Trade[]): number {
  if (trades.length < 10) return 0;  // Not enough data
  
  const N = trades.length;
  const wins = trades.filter(t => t.outcome === 'WIN').length;
  const losses = N - wins;
  
  if (wins === 0 || losses === 0) return 0;  // All same outcome
  
  // Count runs (streaks)
  let R = 1;
  for (let i = 1; i < trades.length; i++) {
    if (trades[i].outcome !== trades[i - 1].outcome) {
      R++;
    }
  }
  
  const P = 2 * wins * losses;
  
  // Calculate Z-Score
  const numerator = N * (R - 0.5) - P;
  const denominator = Math.sqrt((P * (P - N)) / (N - 1));
  
  if (denominator === 0) return 0;
  
  return numerator / denominator;
}

/**
 * Check if Z-Score indicates statistical significance
 */
export function isStatisticallySignificant(zScore: number): boolean {
  return Math.abs(zScore) > PERFORMANCE_CONFIG.Z_SCORE_SIGNIFICANCE;
}

// ============================================
// EXPECTANCY CALCULATOR
// ============================================

/**
 * Calculate Expectancy (EV)
 * 
 * Formula: EV = (Win% * AvgWin) - (Loss% * AvgLoss)
 * 
 * If EV < 1.0 (not making at least 1 unit of risk per trade on average),
 * the bot should pause and output a "Performance Degradation" report.
 */
export function calculateExpectancy(trades: Trade[]): {
  expectancy: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
} {
  if (trades.length === 0) {
    return { expectancy: 0, winRate: 0, avgWin: 0, avgLoss: 0 };
  }
  
  const wins = trades.filter(t => t.outcome === 'WIN');
  const losses = trades.filter(t => t.outcome === 'LOSS');
  
  const winRate = wins.length / trades.length;
  const lossRate = losses.length / trades.length;
  
  const avgWin = wins.length > 0
    ? wins.reduce((sum, t) => sum + t.pnlPips, 0) / wins.length
    : 0;
  
  const avgLoss = losses.length > 0
    ? Math.abs(losses.reduce((sum, t) => sum + t.pnlPips, 0) / losses.length)
    : 0;
  
  // Normalize to units of risk
  // AvgWin should be ~2.2x AvgLoss for our 1:2.2 RRR
  const avgRisk = avgLoss > 0 ? avgLoss : 1;
  const normalizedAvgWin = avgWin / avgRisk;
  const normalizedAvgLoss = 1;  // 1 unit of risk
  
  const expectancy = (winRate * normalizedAvgWin) - (lossRate * normalizedAvgLoss);
  
  return { expectancy, winRate, avgWin, avgLoss };
}

// ============================================
// PROFIT FACTOR CALCULATOR
// ============================================

/**
 * Calculate Profit Factor
 * 
 * Formula: Gross Profit / Gross Loss
 * Target: > 2.0
 */
export function calculateProfitFactor(trades: Trade[]): number {
  const grossProfit = trades
    .filter(t => t.outcome === 'WIN')
    .reduce((sum, t) => sum + t.pnlPips, 0);
  
  const grossLoss = Math.abs(
    trades
      .filter(t => t.outcome === 'LOSS')
      .reduce((sum, t) => sum + t.pnlPips, 0)
  );
  
  if (grossLoss === 0) return grossProfit > 0 ? Infinity : 0;
  
  return grossProfit / grossLoss;
}

// ============================================
// SLIPPAGE MONITOR
// ============================================

/**
 * Calculate slippage for a single trade
 */
export function calculateSlippage(trade: Trade): number {
  return Math.abs(trade.actualFilledPrice - trade.requestedEntryPrice);
}

/**
 * Check if slippage exceeds threshold
 * 
 * If spread + slippage eats more than 15% of the 2.2 RRR target,
 * immediately stop trading that specific pair for the day.
 */
export function checkSlippageThreshold(
  trade: Trade,
  symbol: string
): { exceeds: boolean; slippagePips: number; slippagePercent: number } {
  const slippage = calculateSlippage(trade);
  const slippagePips = priceToPips(slippage, symbol);
  
  // Calculate expected profit in pips (based on RRR)
  const riskPips = priceToPips(Math.abs(trade.entryPrice - trade.stopLoss), symbol);
  const expectedProfitPips = riskPips * ICT_CONFIG.RRR_RATIO;
  
  const slippagePercent = (slippagePips / expectedProfitPips) * 100;
  
  return {
    exceeds: slippagePercent > PERFORMANCE_CONFIG.MAX_SLIPPAGE_PERCENT,
    slippagePips,
    slippagePercent,
  };
}

/**
 * Analyze slippage by symbol
 */
export function analyzeSlippageBySymbol(trades: Trade[]): SlippageReport[] {
  const symbolGroups = new Map<string, Trade[]>();
  
  for (const trade of trades) {
    const existing = symbolGroups.get(trade.symbol) || [];
    existing.push(trade);
    symbolGroups.set(trade.symbol, existing);
  }
  
  const reports: SlippageReport[] = [];
  
  for (const [symbol, symbolTrades] of symbolGroups) {
    const slippages = symbolTrades.map(t => {
      const result = checkSlippageThreshold(t, symbol);
      return result.slippagePips;
    });
    
    const avgSlippagePips = slippages.reduce((a, b) => a + b, 0) / slippages.length;
    const maxSlippagePips = Math.max(...slippages);
    
    // Calculate average slippage percent
    const slippagePercents = symbolTrades.map(t => {
      const result = checkSlippageThreshold(t, symbol);
      return result.slippagePercent;
    });
    const avgSlippagePercent = slippagePercents.reduce((a, b) => a + b, 0) / slippagePercents.length;
    
    reports.push({
      symbol,
      avgSlippagePips,
      maxSlippagePips,
      slippagePercent: avgSlippagePercent,
      isPaused: avgSlippagePercent > PERFORMANCE_CONFIG.MAX_SLIPPAGE_PERCENT,
      trades: symbolTrades.length,
    });
  }
  
  return reports;
}

// ============================================
// MAX DRAWDOWN CALCULATOR
// ============================================

/**
 * Calculate maximum drawdown
 */
export function calculateMaxDrawdown(trades: Trade[]): { pips: number; percent: number } {
  if (trades.length === 0) return { pips: 0, percent: 0 };
  
  let peak = 0;
  let equity = 0;
  let maxDrawdownPips = 0;
  
  for (const trade of trades) {
    equity += trade.pnlPips;
    peak = Math.max(peak, equity);
    const drawdown = peak - equity;
    maxDrawdownPips = Math.max(maxDrawdownPips, drawdown);
  }
  
  // Calculate percent drawdown relative to peak
  const maxDrawdownPercent = peak > 0 ? (maxDrawdownPips / peak) * 100 : 0;
  
  return { pips: maxDrawdownPips, percent: maxDrawdownPercent };
}

// ============================================
// PERFORMANCE TRACKER CLASS
// ============================================

export class PerformanceTracker {
  private trades: Trade[] = [];
  private pausedSymbols: Set<string> = new Set();
  private lastAudit: AuditReport | null = null;
  
  constructor(existingTrades: Trade[] = []) {
    this.trades = existingTrades;
  }
  
  /**
   * Add a completed trade
   */
  addTrade(trade: Trade): void {
    this.trades.push(trade);
    
    // Check slippage for this symbol
    const slippageCheck = checkSlippageThreshold(trade, trade.symbol);
    if (slippageCheck.exceeds) {
      this.pausedSymbols.add(trade.symbol);
      console.log(`⚠️ Symbol ${trade.symbol} paused due to excessive slippage (${slippageCheck.slippagePercent.toFixed(1)}%)`);
    }
  }
  
  /**
   * Check if a symbol is paused due to slippage
   */
  isSymbolPaused(symbol: string): boolean {
    return this.pausedSymbols.has(symbol);
  }
  
  /**
   * Reset paused symbols (e.g., at start of new day)
   */
  resetPausedSymbols(): void {
    this.pausedSymbols.clear();
  }
  
  /**
   * Get recent trades for analysis
   */
  getRecentTrades(count: number = PERFORMANCE_CONFIG.WIN_RATE_LOOKBACK_TRADES): Trade[] {
    return this.trades.slice(-count);
  }
  
  /**
   * Calculate all performance metrics
   */
  calculateMetrics(): PerformanceMetrics {
    const trades = this.trades;
    
    if (trades.length === 0) {
      return {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        winRate: 0,
        profitFactor: 0,
        expectancy: 0,
        avgWin: 0,
        avgLoss: 0,
        maxDrawdown: 0,
        maxDrawdownPercent: 0,
        sharpeRatio: 0,
        zScore: 0,
        isStatisticallySignificant: false,
        confidenceWarning: false,
        performanceDegraded: false,
      };
    }
    
    const wins = trades.filter(t => t.outcome === 'WIN').length;
    const losses = trades.filter(t => t.outcome === 'LOSS').length;
    const winRate = (wins / trades.length) * 100;
    
    const profitFactor = calculateProfitFactor(trades);
    const { expectancy, avgWin, avgLoss } = calculateExpectancy(trades);
    const { pips: maxDrawdown, percent: maxDrawdownPercent } = calculateMaxDrawdown(trades);
    const zScore = calculateZScore(trades);
    
    // Calculate Sharpe Ratio (simplified)
    const returns = trades.map(t => t.pnlPips);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    const sharpeRatio = stdDev > 0 ? avgReturn / stdDev : 0;
    
    // Check recent performance
    const recentTrades = this.getRecentTrades();
    const recentWinRate = recentTrades.length > 0
      ? (recentTrades.filter(t => t.outcome === 'WIN').length / recentTrades.length) * 100
      : winRate;
    
    const confidenceWarning = recentWinRate < PERFORMANCE_CONFIG.WIN_RATE_WARNING_THRESHOLD
      && recentTrades.length >= PERFORMANCE_CONFIG.WIN_RATE_LOOKBACK_TRADES;
    
    const performanceDegraded = expectancy < PERFORMANCE_CONFIG.MIN_EXPECTANCY;
    
    return {
      totalTrades: trades.length,
      wins,
      losses,
      winRate,
      profitFactor,
      expectancy,
      avgWin,
      avgLoss,
      maxDrawdown,
      maxDrawdownPercent,
      sharpeRatio,
      zScore,
      isStatisticallySignificant: isStatisticallySignificant(zScore),
      confidenceWarning,
      performanceDegraded,
    };
  }
  
  /**
   * Generate audit report (called at end of NY session daily)
   */
  generateAuditReport(): AuditReport {
    const metrics = this.calculateMetrics();
    const slippageReports = analyzeSlippageBySymbol(this.trades);
    const warnings: string[] = [];
    const recommendations: string[] = [];
    
    // Check for confidence warning
    if (metrics.confidenceWarning) {
      warnings.push(
        `⚠️ CONFIDENCE WARNING: Win rate dropped to ${metrics.winRate.toFixed(1)}% ` +
        `over last ${PERFORMANCE_CONFIG.WIN_RATE_LOOKBACK_TRADES} trades ` +
        `(target: ${PERFORMANCE_CONFIG.WIN_RATE_WARNING_THRESHOLD}%)`
      );
    }
    
    // Check for statistical significance
    if (metrics.isStatisticallySignificant) {
      if (metrics.zScore > 0) {
        warnings.push(
          `📊 Z-Score ${metrics.zScore.toFixed(2)} indicates significant positive streak dependency`
        );
      } else {
        warnings.push(
          `📊 Z-Score ${metrics.zScore.toFixed(2)} indicates significant negative streak dependency`
        );
      }
    }
    
    // Check expectancy
    if (metrics.performanceDegraded) {
      warnings.push(
        `🚨 PERFORMANCE DEGRADATION: Expectancy ${metrics.expectancy.toFixed(2)} ` +
        `< ${PERFORMANCE_CONFIG.MIN_EXPECTANCY} - BOT SHOULD PAUSE`
      );
      recommendations.push('Review entry criteria and consider tightening liquidity sweep parameters');
    }
    
    // Check profit factor
    if (metrics.profitFactor < ICT_CONFIG.TARGET_WIN_RATE / 100) {
      warnings.push(
        `⚠️ Profit factor ${metrics.profitFactor.toFixed(2)} below target ${PERFORMANCE_CONFIG.MIN_EXPECTANCY}`
      );
    }
    
    // Check slippage
    const pausedSymbols = slippageReports.filter(r => r.isPaused);
    if (pausedSymbols.length > 0) {
      warnings.push(
        `🛑 ${pausedSymbols.length} symbol(s) paused due to slippage: ` +
        pausedSymbols.map(s => s.symbol).join(', ')
      );
      recommendations.push('Consider widening entry zones or reducing position sizes for high-slippage pairs');
    }
    
    // General recommendations
    if (metrics.winRate < 60) {
      recommendations.push('Review HTF bias alignment - consider requiring stronger 15m structure confirmation');
    }
    
    if (metrics.maxDrawdownPercent > 15) {
      recommendations.push('Implement position sizing reduction after consecutive losses');
    }
    
    this.lastAudit = {
      timestamp: new Date().toISOString(),
      period: 'daily',
      metrics,
      slippageBySymbol: slippageReports,
      warnings,
      recommendations,
      isPaused: metrics.performanceDegraded,
    };
    
    return this.lastAudit;
  }
  
  /**
   * Check if trading should be paused
   */
  shouldPause(): { pause: boolean; reason: string | null } {
    const metrics = this.calculateMetrics();
    
    if (metrics.performanceDegraded) {
      return {
        pause: true,
        reason: `Expectancy ${metrics.expectancy.toFixed(2)} below minimum ${PERFORMANCE_CONFIG.MIN_EXPECTANCY}`,
      };
    }
    
    if (metrics.confidenceWarning) {
      return {
        pause: true,
        reason: `Win rate ${metrics.winRate.toFixed(1)}% below ${PERFORMANCE_CONFIG.WIN_RATE_WARNING_THRESHOLD}%`,
      };
    }
    
    return { pause: false, reason: null };
  }
  
  /**
   * Get dashboard metrics for UI
   */
  getDashboardMetrics(): {
    liveWinRate: number;
    profitFactor: number;
    zScore: number;
    expectancy: number;
    targetWinRate: number;
    targetProfitFactor: number;
    equityCurve: { x: string; y: number }[];
    benchmarkCurve: { x: string; y: number }[];
  } {
    const metrics = this.calculateMetrics();
    
    // Generate equity curve
    let equity = 0;
    const equityCurve = this.trades.map(t => {
      equity += t.pnlPips;
      return { x: t.timestamp, y: equity };
    });
    
    // Generate benchmark curve (theoretical 1:2.2 RRR with target win rate)
    const avgRisk = metrics.avgLoss > 0 ? metrics.avgLoss : 10;
    const benchmarkWinRate = ICT_CONFIG.TARGET_WIN_RATE / 100;
    let benchmarkEquity = 0;
    const benchmarkCurve = this.trades.map((t, i) => {
      // Simulate expected outcome
      const isWin = (i % Math.round(1 / benchmarkWinRate)) !== 0;
      benchmarkEquity += isWin ? avgRisk * ICT_CONFIG.RRR_RATIO : -avgRisk;
      return { x: t.timestamp, y: benchmarkEquity };
    });
    
    return {
      liveWinRate: metrics.winRate,
      profitFactor: metrics.profitFactor,
      zScore: metrics.zScore,
      expectancy: metrics.expectancy,
      targetWinRate: ICT_CONFIG.TARGET_WIN_RATE,
      targetProfitFactor: PERFORMANCE_CONFIG.MIN_EXPECTANCY,
      equityCurve,
      benchmarkCurve,
    };
  }
}

// ============================================
// MONTE CARLO STRESS TEST
// ============================================

export interface MonteCarloResult {
  probabilityOfRuin: number;
  drawdown95CI: { lower: number; upper: number };
  finalEquity95CI: { lower: number; upper: number };
  medianFinalEquity: number;
  worstCaseEquity: number;
  bestCaseEquity: number;
}

/**
 * Run Monte Carlo simulation
 * 
 * Takes last 100 trades, shuffles their order 10,000 times,
 * and simulates 10,000 different "equity paths"
 */
export function runMonteCarloStressTest(
  trades: Trade[],
  simulations: number = PERFORMANCE_CONFIG.MONTE_CARLO_SIMULATIONS,
  startingCapital: number = 10000
): MonteCarloResult {
  const tradesToUse = trades.slice(-PERFORMANCE_CONFIG.MONTE_CARLO_TRADE_HISTORY);
  
  if (tradesToUse.length < 10) {
    return {
      probabilityOfRuin: 0,
      drawdown95CI: { lower: 0, upper: 0 },
      finalEquity95CI: { lower: startingCapital, upper: startingCapital },
      medianFinalEquity: startingCapital,
      worstCaseEquity: startingCapital,
      bestCaseEquity: startingCapital,
    };
  }
  
  const results: { finalEquity: number; maxDrawdown: number; ruined: boolean }[] = [];
  
  for (let sim = 0; sim < simulations; sim++) {
    // Shuffle trades
    const shuffled = [...tradesToUse].sort(() => Math.random() - 0.5);
    
    // Simulate equity path
    let equity = startingCapital;
    let peak = startingCapital;
    let maxDrawdown = 0;
    let ruined = false;
    
    for (const trade of shuffled) {
      // Convert pips to account currency (simplified: 1 pip = $1)
      equity += trade.pnlPips;
      
      if (equity <= 0) {
        ruined = true;
        break;
      }
      
      peak = Math.max(peak, equity);
      maxDrawdown = Math.max(maxDrawdown, (peak - equity) / peak);
    }
    
    results.push({
      finalEquity: Math.max(0, equity),
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
  
  const percentile = (arr: number[], p: number) => arr[Math.floor(arr.length * p)];
  
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
  };
}

export default PerformanceTracker;
