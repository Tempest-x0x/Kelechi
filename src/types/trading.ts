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
