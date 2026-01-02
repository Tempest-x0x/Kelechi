import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const PAYSTACK_SECRET_KEY = Deno.env.get('PAYSTACK_SECRET_KEY');
const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

// Plan configurations
const PLANS = {
  retail: {
    amount: 7600000, // ₦76,000 in kobo
    name: 'Retail',
    interval: 'monthly',
  },
  funded: {
    amount: 31000000, // ₦310,000 in kobo
    name: 'Funded Trader',
    interval: 'monthly',
  },
  lifetime: {
    amount: 155000000, // ₦1,550,000 in kobo
    name: 'Lifetime',
    interval: null, // One-time payment
  },
};

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    if (!PAYSTACK_SECRET_KEY) {
      throw new Error('Paystack secret key not configured');
    }

    const { plan, email, userId } = await req.json();

    console.log(`Initializing payment for plan: ${plan}, email: ${email}, userId: ${userId}`);

    if (!plan || !email || !userId) {
      throw new Error('Missing required fields: plan, email, userId');
    }

    const planConfig = PLANS[plan as keyof typeof PLANS];
    if (!planConfig) {
      throw new Error(`Invalid plan: ${plan}`);
    }

    // Generate unique reference
    const reference = `ft_${plan}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Initialize transaction with Paystack
    const paystackResponse = await fetch('https://api.paystack.co/transaction/initialize', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${PAYSTACK_SECRET_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        amount: planConfig.amount,
        reference,
        callback_url: `${req.headers.get('origin') || 'https://forextell.ai'}/dashboard`,
        metadata: {
          plan_type: plan,
          user_id: userId,
          plan_name: planConfig.name,
        },
      }),
    });

    const paystackData = await paystackResponse.json();

    console.log('Paystack response:', paystackData);

    if (!paystackData.status) {
      throw new Error(paystackData.message || 'Failed to initialize payment');
    }

    // Create pending subscription in database
    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);

    const { error: insertError } = await supabase
      .from('subscriptions')
      .insert({
        user_id: userId,
        plan_type: plan,
        status: 'pending',
        paystack_reference: reference,
        amount: planConfig.amount,
        current_period_start: new Date().toISOString(),
        current_period_end: planConfig.interval === 'monthly' 
          ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()
          : null, // Lifetime has no end
      });

    if (insertError) {
      console.error('Error creating subscription record:', insertError);
      // Don't throw - payment can still proceed
    }

    return new Response(
      JSON.stringify({
        authorization_url: paystackData.data.authorization_url,
        reference: paystackData.data.reference,
        access_code: paystackData.data.access_code,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error: any) {
    console.error('Error in paystack-initialize:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
