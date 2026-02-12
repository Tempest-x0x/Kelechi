export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: { value: number; signal: number; histogram: number };
  ema9: number;
  ema21: number;
  ema50: number;
  bollingerBands: { upper: number; middle: number; lower: number };
  stochastic: { k: number; d: number };
}

export interface Prediction {
  id: string;
  created_at: string;
  signal_type: 'BUY' | 'SELL';
  confidence: number;
  entry_price: number;
  take_profit_1?: number;
  take_profit_2?: number;
  stop_loss?: number;
  current_price_at_prediction: number;
  trend_direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  trend_strength: number;
  reasoning?: string;
  technical_indicators?: TechnicalIndicators;
  patterns_detected?: string[];
  sentiment_score?: number;
  outcome: 'WIN' | 'LOSS' | 'PENDING' | 'EXPIRED';
  outcome_price?: number;
  outcome_at?: string;
  expires_at: string;
}

export interface NewsItem {
  headline: string;
  source: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'high' | 'medium' | 'low';
}

export interface MarketSentiment {
  overall_sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  sentiment_score: number;
  summary: string;
  news_items: NewsItem[];
  key_factors: string[];
  generated_at: string;
}

export interface ForexData {
  symbol: string;
  currentPrice: number;
  candles: Candle[];
  meta?: any;
}

export type Timeframe = '1min' | '5min' | '15min' | '30min' | '1h' | '4h';

// ============================================
// ICT/SMC TYPES
// ============================================

export interface ICTConfig {
  RRR_RATIO: number;
  TARGET_WIN_RATE: number;
  MAX_TRADES_PER_PAIR_PER_DAY: number;
  MIN_CONFIDENCE_THRESHOLD: number;
  POSITION_EXPIRY_HOURS: number;
}

export interface Killzone {
  name: string;
  start: { hour: number; minute: number };
  end: { hour: number; minute: number };
}

export interface HTFBias {
  direction: 'BUY' | 'SELL' | null;
  hourlyTrend: 'ABOVE_200EMA' | 'BELOW_200EMA';
  ema200: number;
  confidence: number;
  reasons: string[];
}

export interface LiquiditySweep {
  detected: boolean;
  type: 'BUY' | 'SELL';
  sweepPrice: number;
  sweepLow: number;
  sweepHigh: number;
  candleIndex: number;
  liquidityLevel: number;
}

export interface FairValueGap {
  detected: boolean;
  type: 'BULLISH' | 'BEARISH';
  topPrice: number;
  bottomPrice: number;
  equilibrium: number;
  gapSizePips: number;
  isValid: boolean;
  age: number;
}

export interface MarketStructureShift {
  detected: boolean;
  direction: 'BULLISH' | 'BEARISH';
  displacementIndex: number;
}

export interface ICTSignal {
  valid: boolean;
  direction: 'BUY' | 'SELL' | null;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  htfBias: HTFBias | null;
  liquiditySweep: LiquiditySweep | null;
  fvg: FairValueGap | null;
  mss: MarketStructureShift | null;
  confidence: number;
  reasons: string[];
  invalidReasons: string[];
}

// Performance Tracking Types
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
  slippagePercent: number;
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

export interface MonteCarloResult {
  probabilityOfRuin: number;
  drawdown95CI: { lower: number; upper: number };
  finalEquity95CI: { lower: number; upper: number };
  medianFinalEquity: number;
  worstCaseEquity: number;
  bestCaseEquity: number;
}
