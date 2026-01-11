-- Drop the old unique constraint on pattern_name only
ALTER TABLE pattern_statistics DROP CONSTRAINT IF EXISTS pattern_statistics_pattern_name_key;

-- Ensure we have the new unique constraint on symbol + pattern_name + signal_type
DROP INDEX IF EXISTS idx_pattern_statistics_unique;
CREATE UNIQUE INDEX idx_pattern_statistics_unique ON pattern_statistics(symbol, pattern_name, signal_type);