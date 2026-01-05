-- Create whitelisted_emails table for admin-managed free access
CREATE TABLE public.whitelisted_emails (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  reason TEXT,
  added_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ -- NULL = never expires
);

-- Enable RLS
ALTER TABLE public.whitelisted_emails ENABLE ROW LEVEL SECURITY;

-- Admins can manage all whitelist entries
CREATE POLICY "Admins can manage whitelist"
ON public.whitelisted_emails
FOR ALL
TO authenticated
USING (public.has_role(auth.uid(), 'admin'))
WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Service role can read for verification
CREATE POLICY "Service role can read whitelist"
ON public.whitelisted_emails
FOR SELECT
USING (true);