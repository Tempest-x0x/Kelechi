import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';

interface Subscription {
  id: string;
  user_id: string;
  plan_type: string;
  status: string;
  amount: number;
  current_period_start: string;
  current_period_end: string | null;
  created_at: string;
  paystack_subscription_code: string | null;
  paystack_customer_code: string | null;
  paystack_reference: string | null;
}

export const useSubscription = () => {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSubscription = useCallback(async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        setIsLoading(false);
        return null;
      }

      // First try to get active subscription
      let { data, error: fetchError } = await supabase
        .from('subscriptions')
        .select('*')
        .eq('user_id', session.user.id)
        .eq('status', 'active')
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      // If no active, check for cancelled (still valid until period end)
      if (!data) {
        const result = await supabase
          .from('subscriptions')
          .select('*')
          .eq('user_id', session.user.id)
          .in('status', ['cancelled', 'expired'])
          .order('created_at', { ascending: false })
          .limit(1)
          .maybeSingle();
        
        data = result.data;
        fetchError = result.error;
      }

      if (fetchError) {
        console.error('Error fetching subscription:', fetchError);
        setError(fetchError.message);
        return null;
      } else {
        setSubscription(data);
        return data;
      }
    } catch (err: any) {
      console.error('Subscription error:', err);
      setError(err.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshSubscription = useCallback(async () => {
    setIsLoading(true);
    return await fetchSubscription();
  }, [fetchSubscription]);

  // Poll for subscription with timeout (for post-payment verification)
  const waitForSubscription = useCallback(async (maxWaitMs: number = 10000): Promise<Subscription | null> => {
    const startTime = Date.now();
    const pollInterval = 1500;

    while (Date.now() - startTime < maxWaitMs) {
      const sub = await fetchSubscription();
      if (sub?.status === 'active') {
        return sub;
      }
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    return null;
  }, [fetchSubscription]);

  useEffect(() => {
    fetchSubscription();

    // Listen for auth changes
    const { data: { subscription: authSubscription } } = supabase.auth.onAuthStateChange(() => {
      fetchSubscription();
    });

    return () => authSubscription.unsubscribe();
  }, [fetchSubscription]);

  const hasActiveSubscription = subscription?.status === 'active' || 
    (subscription?.status === 'cancelled' && subscription?.current_period_end && new Date(subscription.current_period_end) > new Date());

  const isLifetime = subscription?.plan_type === 'lifetime';

  const daysRemaining = subscription?.current_period_end
    ? Math.max(0, Math.ceil((new Date(subscription.current_period_end).getTime() - Date.now()) / (1000 * 60 * 60 * 24)))
    : null;

  return {
    subscription,
    isLoading,
    error,
    hasActiveSubscription,
    isLifetime,
    daysRemaining,
    refreshSubscription,
    waitForSubscription,
  };
};