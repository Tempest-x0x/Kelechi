-- Add symbol column to trading_opportunities for multi-currency support
ALTER TABLE trading_opportunities ADD COLUMN IF NOT EXISTS symbol TEXT DEFAULT 'EUR/USD';

-- Create indexes for efficient symbol-based queries
CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON trading_opportunities(symbol);
CREATE INDEX IF NOT EXISTS idx_opportunities_symbol_status ON trading_opportunities(symbol, status);

-- Add index on price_history for symbol + timeframe combo (if not exists)
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_timeframe ON price_history(symbol, timeframe);

-- Update existing records to have EUR/USD as symbol (in case any are NULL)
UPDATE trading_opportunities SET symbol = 'EUR/USD' WHERE symbol IS NULL;