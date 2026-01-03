import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const PAYSTACK_SECRET_KEY = Deno.env.get('PAYSTACK_SECRET_KEY');
const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface CancelRequest {
  subscriptionCode: string;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { subscriptionCode }: CancelRequest = await req.json();

    if (!subscriptionCode) {
      return new Response(
        JSON.stringify({ error: 'Subscription code is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Cancelling subscription: ${subscriptionCode}`);

    // First, get the subscription details from Paystack to find the email token
    const getResponse = await fetch(
      `https://api.paystack.co/subscription/${subscriptionCode}`,
      {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );

    const subscriptionData = await getResponse.json();
    
    if (!getResponse.ok) {
      console.error('Failed to get subscription:', subscriptionData);
      throw new Error(subscriptionData.message || 'Failed to get subscription details');
    }

    const emailToken = subscriptionData.data?.email_token;

    if (!emailToken) {
      console.error('No email token found in subscription data');
      throw new Error('Unable to cancel subscription - missing email token');
    }

    // Cancel the subscription on Paystack
    const cancelResponse = await fetch('https://api.paystack.co/subscription/disable', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        code: subscriptionCode,
        token: emailToken,
      }),
    });

    const cancelResult = await cancelResponse.json();

    if (!cancelResponse.ok) {
      console.error('Failed to cancel on Paystack:', cancelResult);
      throw new Error(cancelResult.message || 'Failed to cancel subscription on Paystack');
    }

    console.log('Subscription cancelled on Paystack:', cancelResult);

    // Update local database
    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);
    
    const { error: updateError } = await supabase
      .from('subscriptions')
      .update({
        status: 'cancelled',
        updated_at: new Date().toISOString(),
      })
      .eq('paystack_subscription_code', subscriptionCode);

    if (updateError) {
      console.error('Failed to update local subscription:', updateError);
      // Don't throw here - the Paystack cancellation was successful
    }

    return new Response(
      JSON.stringify({ success: true, message: 'Subscription cancelled successfully' }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error: any) {
    console.error('Cancel subscription error:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});