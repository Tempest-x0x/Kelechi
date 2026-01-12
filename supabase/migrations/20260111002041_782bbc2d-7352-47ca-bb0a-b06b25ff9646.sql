-- Create the update_updated_at_column function if it doesn't exist
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create supported_currency_pairs table for managing available pairs
CREATE TABLE public.supported_currency_pairs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  pip_value NUMERIC NOT NULL DEFAULT 0.0001,
  is_active BOOLEAN DEFAULT true,
  has_pattern_stats BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.supported_currency_pairs ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Allow public read access to currency pairs"
ON public.supported_currency_pairs
FOR SELECT
USING (true);

-- Allow admins to manage currency pairs
CREATE POLICY "Admins can manage currency pairs"
ON public.supported_currency_pairs
FOR ALL
USING (has_role(auth.uid(), 'admin'::app_role))
WITH CHECK (has_role(auth.uid(), 'admin'::app_role));

-- Insert initial currency pairs based on imported pattern statistics
INSERT INTO public.supported_currency_pairs (symbol, display_name, pip_value, is_active, has_pattern_stats) VALUES
('EUR/USD', 'Euro / US Dollar', 0.0001, true, true),
('GBP/USD', 'British Pound / US Dollar', 0.0001, true, true),
('USD/JPY', 'US Dollar / Japanese Yen', 0.01, true, true),
('USD/CHF', 'US Dollar / Swiss Franc', 0.0001, true, true),
('AUD/USD', 'Australian Dollar / US Dollar', 0.0001, true, true),
('USD/CAD', 'US Dollar / Canadian Dollar', 0.0001, true, true),
('EUR/JPY', 'Euro / Japanese Yen', 0.01, true, true),
('GBP/JPY', 'British Pound / Japanese Yen', 0.01, true, true),
('AUD/JPY', 'Australian Dollar / Japanese Yen', 0.01, true, true),
('XAU/USD', 'Gold / US Dollar', 0.01, true, true);

-- Create trigger for updated_at
CREATE TRIGGER update_supported_currency_pairs_updated_at
BEFORE UPDATE ON public.supported_currency_pairs
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();