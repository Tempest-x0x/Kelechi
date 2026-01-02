import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY');

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Get user from authorization header
    const authHeader = req.headers.get('authorization');
    if (!authHeader) {
      throw new Error('No authorization header');
    }

    const supabase = createClient(SUPABASE_URL!, SUPABASE_ANON_KEY!, {
      global: {
        headers: { Authorization: authHeader },
      },
    });

    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      throw new Error('Unauthorized');
    }

    // Get active subscription for user
    const { data: subscription, error: subError } = await supabase
      .from('subscriptions')
      .select('*')
      .eq('user_id', user.id)
      .eq('status', 'active')
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (subError) {
      console.error('Error fetching subscription:', subError);
      throw new Error('Failed to fetch subscription');
    }

    if (!subscription) {
      return new Response(
        JSON.stringify({
          hasAccess: false,
          plan: null,
          expiresAt: null,
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Check if subscription is still valid (for non-lifetime plans)
    const isLifetime = subscription.plan_type === 'lifetime';
    const isExpired = !isLifetime && 
      subscription.current_period_end && 
      new Date(subscription.current_period_end) < new Date();

    if (isExpired) {
      // Update subscription status to expired
      await supabase
        .from('subscriptions')
        .update({ status: 'expired' })
        .eq('id', subscription.id);

      return new Response(
        JSON.stringify({
          hasAccess: false,
          plan: subscription.plan_type,
          expiresAt: subscription.current_period_end,
          expired: true,
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    return new Response(
      JSON.stringify({
        hasAccess: true,
        plan: subscription.plan_type,
        expiresAt: subscription.current_period_end,
        isLifetime,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error: any) {
    console.error('Error in verify-subscription:', error);
    return new Response(
      JSON.stringify({ error: error.message, hasAccess: false }),
      {
        status: error.message === 'Unauthorized' ? 401 : 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
