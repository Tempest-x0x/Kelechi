-- Create pattern_statistics table to store 25-year historical pattern data
CREATE TABLE public.pattern_statistics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pattern_name TEXT NOT NULL UNIQUE,
  signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
  occurrences INTEGER NOT NULL,
  win_rate_4h NUMERIC,
  win_rate_12h NUMERIC,
  win_rate_24h NUMERIC,
  win_rate_48h NUMERIC,
  avg_pips_4h NUMERIC,
  avg_pips_12h NUMERIC,
  avg_pips_24h NUMERIC,
  avg_pips_48h NUMERIC,
  sample_size INTEGER,
  data_start_date TIMESTAMP WITH TIME ZONE,
  data_end_date TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.pattern_statistics ENABLE ROW LEVEL SECURITY;

-- Allow public read access (pattern statistics are not sensitive)
CREATE POLICY "Allow public read access to pattern statistics"
ON public.pattern_statistics
FOR SELECT
USING (true);

-- Allow service role to manage pattern statistics
CREATE POLICY "Allow service role to manage pattern statistics"
ON public.pattern_statistics
FOR ALL
USING (true)
WITH CHECK (true);