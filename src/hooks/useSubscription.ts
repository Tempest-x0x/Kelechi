import { useState, useEffect } from 'react';
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
}

export const useSubscription = () => {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSubscription = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        
        if (!session) {
          setIsLoading(false);
          return;
        }

        const { data, error: fetchError } = await supabase
          .from('subscriptions')
          .select('*')
          .eq('user_id', session.user.id)
          .eq('status', 'active')
          .order('created_at', { ascending: false })
          .limit(1)
          .maybeSingle();

        if (fetchError) {
          console.error('Error fetching subscription:', fetchError);
          setError(fetchError.message);
        } else {
          setSubscription(data);
        }
      } catch (err: any) {
        console.error('Subscription error:', err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSubscription();

    // Listen for auth changes
    const { data: { subscription: authSubscription } } = supabase.auth.onAuthStateChange(() => {
      fetchSubscription();
    });

    return () => authSubscription.unsubscribe();
  }, []);

  const hasActiveSubscription = subscription?.status === 'active';

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
  };
};
