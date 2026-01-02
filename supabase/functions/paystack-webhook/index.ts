import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { createHmac } from "https://deno.land/std@0.168.0/node/crypto.ts";

const PAYSTACK_SECRET_KEY = Deno.env.get('PAYSTACK_SECRET_KEY');
const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

serve(async (req) => {
  try {
    // Verify webhook signature
    const signature = req.headers.get('x-paystack-signature');
    const body = await req.text();

    if (!PAYSTACK_SECRET_KEY) {
      console.error('Paystack secret key not configured');
      return new Response('Configuration error', { status: 500 });
    }

    // Verify signature
    const hash = createHmac('sha512', PAYSTACK_SECRET_KEY)
      .update(body)
      .digest('hex');

    if (hash !== signature) {
      console.error('Invalid webhook signature');
      return new Response('Invalid signature', { status: 401 });
    }

    const event = JSON.parse(body);
    console.log('Paystack webhook event:', event.event, event.data?.reference);

    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);

    switch (event.event) {
      case 'charge.success': {
        const { reference, customer, metadata, amount } = event.data;
        
        console.log(`Processing successful charge: ${reference}`);

        // Update subscription to active
        const { error: updateError } = await supabase
          .from('subscriptions')
          .update({
            status: 'active',
            paystack_customer_code: customer?.customer_code,
            updated_at: new Date().toISOString(),
          })
          .eq('paystack_reference', reference);

        if (updateError) {
          console.error('Error updating subscription:', updateError);
          
          // Try to create subscription if it doesn't exist
          if (metadata?.user_id && metadata?.plan_type) {
            const isLifetime = metadata.plan_type === 'lifetime';
            
            const { error: insertError } = await supabase
              .from('subscriptions')
              .insert({
                user_id: metadata.user_id,
                plan_type: metadata.plan_type,
                status: 'active',
                paystack_reference: reference,
                paystack_customer_code: customer?.customer_code,
                amount: amount,
                current_period_start: new Date().toISOString(),
                current_period_end: isLifetime 
                  ? null 
                  : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
              });

            if (insertError) {
              console.error('Error creating subscription:', insertError);
            }
          }
        }

        console.log(`Subscription activated for reference: ${reference}`);
        break;
      }

      case 'subscription.create': {
        const { subscription_code, customer, plan, next_payment_date } = event.data;
        
        console.log(`Subscription created: ${subscription_code}`);

        // Update subscription with Paystack subscription code
        const { error } = await supabase
          .from('subscriptions')
          .update({
            paystack_subscription_code: subscription_code,
            paystack_customer_code: customer?.customer_code,
            current_period_end: next_payment_date,
            updated_at: new Date().toISOString(),
          })
          .eq('paystack_customer_code', customer?.customer_code)
          .eq('status', 'active');

        if (error) {
          console.error('Error updating subscription with code:', error);
        }
        break;
      }

      case 'subscription.disable':
      case 'subscription.not_renew': {
        const { subscription_code, customer } = event.data;
        
        console.log(`Subscription cancelled: ${subscription_code}`);

        const { error } = await supabase
          .from('subscriptions')
          .update({
            status: 'cancelled',
            updated_at: new Date().toISOString(),
          })
          .eq('paystack_subscription_code', subscription_code);

        if (error) {
          console.error('Error cancelling subscription:', error);
        }
        break;
      }

      case 'invoice.payment_failed': {
        const { subscription, customer } = event.data;
        
        console.log(`Payment failed for subscription: ${subscription?.subscription_code}`);

        if (subscription?.subscription_code) {
          const { error } = await supabase
            .from('subscriptions')
            .update({
              status: 'expired',
              updated_at: new Date().toISOString(),
            })
            .eq('paystack_subscription_code', subscription.subscription_code);

          if (error) {
            console.error('Error marking subscription as expired:', error);
          }
        }
        break;
      }

      default:
        console.log(`Unhandled event type: ${event.event}`);
    }

    return new Response('OK', { status: 200 });
  } catch (error: any) {
    console.error('Webhook error:', error);
    return new Response(error.message, { status: 500 });
  }
});
