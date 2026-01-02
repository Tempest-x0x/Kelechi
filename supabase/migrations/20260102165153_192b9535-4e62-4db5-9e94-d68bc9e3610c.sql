-- Add linking columns to trading_opportunities
ALTER TABLE trading_opportunities 
ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS ai_learning_id UUID REFERENCES prediction_learnings(id);

-- Add opportunity_id to prediction_learnings for linking
ALTER TABLE prediction_learnings
ADD COLUMN IF NOT EXISTS opportunity_id UUID REFERENCES trading_opportunities(id);