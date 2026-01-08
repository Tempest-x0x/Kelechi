-- Add column to track when Telegram notification was sent (prevents duplicate notifications)
ALTER TABLE trading_opportunities 
ADD COLUMN IF NOT EXISTS notification_sent_at TIMESTAMPTZ;