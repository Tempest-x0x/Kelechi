-- ICT/SMC Institutional Order Flow System Tables
-- Migration for tracking ICT signals, slippage, and performance metrics

-- Add ICT-specific columns to trading_opportunities
ALTER TABLE trading_opportunities
ADD COLUMN IF NOT EXISTS entry_type TEXT DEFAULT 'MARKET', -- 'MARKET', 'LIMIT_FVG'
ADD COLUMN IF NOT EXISTS ict_signal_flow JSONB, -- HTF Bias, Liquidity Sweep, MSS, FVG data
ADD COLUMN IF NOT EXISTS requested_entry_price DECIMAL(18, 8), -- For slippage tracking
ADD COLUMN IF NOT EXISTS actual_filled_price DECIMAL(18, 8), -- Actual execution price
ADD COLUMN IF NOT EXISTS slippage_pips DECIMAL(10, 4), -- Calculated slippage
ADD COLUMN IF NOT EXISTS killzone TEXT, -- 'LONDON', 'NEW_YORK', or null
ADD COLUMN IF NOT EXISTS rrr_target DECIMAL(4, 2) DEFAULT 2.2;

-- Create performance_audits table for daily expectancy audits
CREATE TABLE IF NOT EXISTS performance_audits (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  audit_date DATE NOT NULL,
  period TEXT DEFAULT 'daily', -- 'daily', 'weekly', 'monthly'
  
  -- Core metrics
  total_trades INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  win_rate DECIMAL(5, 2) NOT NULL DEFAULT 0,
  profit_factor DECIMAL(6, 3),
  expectancy DECIMAL(6, 3),
  avg_win_pips DECIMAL(10, 4),
  avg_loss_pips DECIMAL(10, 4),
  
  -- Statistical metrics
  z_score DECIMAL(6, 3),
  is_statistically_significant BOOLEAN DEFAULT false,
  max_drawdown_percent DECIMAL(5, 2),
  sharpe_ratio DECIMAL(6, 3),
  
  -- Status
  confidence_warning BOOLEAN DEFAULT false,
  performance_degraded BOOLEAN DEFAULT false,
  is_paused BOOLEAN DEFAULT false,
  
  -- Details
  warnings TEXT[],
  recommendations TEXT[],
  slippage_by_symbol JSONB,
  
  UNIQUE(audit_date, period)
);

-- Create slippage_tracking table
CREATE TABLE IF NOT EXISTS slippage_tracking (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  opportunity_id UUID REFERENCES trading_opportunities(id),
  symbol TEXT NOT NULL,
  
  requested_price DECIMAL(18, 8) NOT NULL,
  filled_price DECIMAL(18, 8) NOT NULL,
  slippage_price DECIMAL(18, 8) NOT NULL,
  slippage_pips DECIMAL(10, 4) NOT NULL,
  slippage_percent DECIMAL(6, 3) NOT NULL, -- As % of RRR target
  
  -- Context
  spread_at_entry DECIMAL(10, 4),
  killzone TEXT,
  is_paused BOOLEAN DEFAULT false -- If this trade caused pair to be paused
);

-- Create paused_symbols table
CREATE TABLE IF NOT EXISTS paused_symbols (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  symbol TEXT NOT NULL,
  reason TEXT NOT NULL,
  paused_until TIMESTAMP WITH TIME ZONE, -- null = until manual unpause
  
  -- Context
  avg_slippage_pips DECIMAL(10, 4),
  slippage_percent DECIMAL(6, 3),
  last_trade_count INTEGER,
  
  is_active BOOLEAN DEFAULT true,
  unpaused_at TIMESTAMP WITH TIME ZONE,
  
  UNIQUE(symbol, created_at)
);

-- Create monte_carlo_results table
CREATE TABLE IF NOT EXISTS monte_carlo_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  
  -- Inputs
  trades_analyzed INTEGER NOT NULL,
  simulations INTEGER NOT NULL DEFAULT 10000,
  starting_capital DECIMAL(12, 2) DEFAULT 10000,
  symbol TEXT, -- null = all pairs
  
  -- Results
  probability_of_ruin DECIMAL(6, 3) NOT NULL,
  drawdown_95ci_lower DECIMAL(6, 3) NOT NULL,
  drawdown_95ci_upper DECIMAL(6, 3) NOT NULL,
  final_equity_95ci_lower DECIMAL(12, 2) NOT NULL,
  final_equity_95ci_upper DECIMAL(12, 2) NOT NULL,
  median_final_equity DECIMAL(12, 2) NOT NULL,
  worst_case_equity DECIMAL(12, 2) NOT NULL,
  best_case_equity DECIMAL(12, 2) NOT NULL,
  
  -- Stats
  avg_return DECIMAL(10, 4),
  volatility DECIMAL(10, 4),
  
  -- Risk assessment
  risk_assessment JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_perf_audits_date ON performance_audits(audit_date DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_symbol ON slippage_tracking(symbol);
CREATE INDEX IF NOT EXISTS idx_slippage_opportunity ON slippage_tracking(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_paused_active ON paused_symbols(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_monte_created ON monte_carlo_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_ict ON trading_opportunities(entry_type) WHERE entry_type = 'LIMIT_FVG';
CREATE INDEX IF NOT EXISTS idx_opportunities_killzone ON trading_opportunities(killzone) WHERE killzone IS NOT NULL;

-- Function to check if a symbol is paused
CREATE OR REPLACE FUNCTION is_symbol_paused(p_symbol TEXT)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM paused_symbols 
    WHERE symbol = p_symbol 
    AND is_active = true
    AND (paused_until IS NULL OR paused_until > NOW())
  );
END;
$$ LANGUAGE plpgsql;

-- Function to pause a symbol due to slippage
CREATE OR REPLACE FUNCTION pause_symbol_for_slippage(
  p_symbol TEXT,
  p_avg_slippage DECIMAL,
  p_slippage_percent DECIMAL,
  p_trade_count INTEGER
) RETURNS UUID AS $$
DECLARE
  v_id UUID;
BEGIN
  INSERT INTO paused_symbols (
    symbol, reason, paused_until, 
    avg_slippage_pips, slippage_percent, last_trade_count
  ) VALUES (
    p_symbol, 
    'Excessive slippage (' || p_slippage_percent || '% of RRR target)',
    (NOW() + INTERVAL '24 hours'), -- Pause until next day
    p_avg_slippage, p_slippage_percent, p_trade_count
  )
  RETURNING id INTO v_id;
  
  RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to unpause all symbols (called at start of new day)
CREATE OR REPLACE FUNCTION unpause_all_symbols()
RETURNS INTEGER AS $$
DECLARE
  v_count INTEGER;
BEGIN
  UPDATE paused_symbols 
  SET is_active = false, unpaused_at = NOW()
  WHERE is_active = true
  AND paused_until IS NOT NULL
  AND paused_until <= NOW();
  
  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL ON performance_audits TO authenticated;
GRANT ALL ON slippage_tracking TO authenticated;
GRANT ALL ON paused_symbols TO authenticated;
GRANT ALL ON monte_carlo_results TO authenticated;

-- Enable RLS
ALTER TABLE performance_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE slippage_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE paused_symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE monte_carlo_results ENABLE ROW LEVEL SECURITY;

-- RLS Policies (allow read for all authenticated users)
CREATE POLICY "Allow read for authenticated users" ON performance_audits
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read for authenticated users" ON slippage_tracking
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read for authenticated users" ON paused_symbols
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read for authenticated users" ON monte_carlo_results
  FOR SELECT TO authenticated USING (true);

-- Service role can do everything
CREATE POLICY "Service role full access" ON performance_audits
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access" ON slippage_tracking
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access" ON paused_symbols
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access" ON monte_carlo_results
  FOR ALL TO service_role USING (true) WITH CHECK (true);
