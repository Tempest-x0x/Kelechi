-- Create trading_opportunities table for storing high-probability setups
CREATE TABLE public.trading_opportunities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
  confidence NUMERIC NOT NULL,
  entry_price NUMERIC NOT NULL,
  current_price NUMERIC NOT NULL,
  stop_loss NUMERIC,
  take_profit_1 NUMERIC,
  take_profit_2 NUMERIC,
  patterns_detected JSONB,
  technical_indicators JSONB,
  pattern_stats JSONB,
  reasoning TEXT,
  status TEXT DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'EXPIRED', 'TRIGGERED')),
  triggered_at TIMESTAMP WITH TIME ZONE,
  outcome TEXT CHECK (outcome IN ('WIN', 'LOSS', 'PENDING', 'EXPIRED'))
);

-- Enable Row Level Security
ALTER TABLE public.trading_opportunities ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access to opportunities" 
ON public.trading_opportunities 
FOR SELECT 
USING (true);

-- Create policy for service role to manage
CREATE POLICY "Allow service role to manage opportunities" 
ON public.trading_opportunities 
FOR ALL 
USING (true)
WITH CHECK (true);

-- Create index for faster queries on active opportunities
CREATE INDEX idx_opportunities_status ON public.trading_opportunities(status);
CREATE INDEX idx_opportunities_expires_at ON public.trading_opportunities(expires_at);
CREATE INDEX idx_opportunities_created_at ON public.trading_opportunities(created_at DESC);