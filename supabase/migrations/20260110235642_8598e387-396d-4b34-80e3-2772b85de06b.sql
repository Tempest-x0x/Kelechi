-- Add symbol column to pattern_statistics for multi-currency support
ALTER TABLE pattern_statistics ADD COLUMN IF NOT EXISTS symbol TEXT DEFAULT 'EUR/USD';

-- Create index for efficient symbol-based queries
CREATE INDEX IF NOT EXISTS idx_pattern_statistics_symbol ON pattern_statistics(symbol);

-- Create unique constraint on symbol + pattern_name + signal_type
DROP INDEX IF EXISTS idx_pattern_statistics_unique;
CREATE UNIQUE INDEX idx_pattern_statistics_unique ON pattern_statistics(symbol, pattern_name, signal_type);

-- Update existing EUR/USD records
UPDATE pattern_statistics SET symbol = 'EUR/USD' WHERE symbol IS NULL;