import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY");
const SUPABASE_URL = Deno.env.get("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

async function sendEmail(to: string, subject: string, html: string) {
  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "ForexTell AI <onboarding@resend.dev>",
      to: [to],
      subject,
      html,
    }),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Resend API error: ${error}`);
  }
  
  return response.json();
}

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface ReminderRequest {
  type: '3_day' | '1_day' | 'expired';
}

const getPlanDisplayName = (planType: string): string => {
  switch (planType) {
    case 'retail':
      return 'Retail';
    case 'funded':
      return 'Funded Trader';
    case 'lifetime':
      return 'Lifetime';
    default:
      return planType;
  }
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);
    const { type }: ReminderRequest = await req.json();
    
    const now = new Date();
    let targetDate: Date;
    let emailSubject: string;
    let emailHeadline: string;
    let emailMessage: string;
    
    switch (type) {
      case '3_day':
        targetDate = new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000);
        emailSubject = 'Your ForexTell AI subscription expires in 3 days';
        emailHeadline = 'Your Subscription Expires Soon';
        emailMessage = 'Your ForexTell AI subscription will expire in <strong>3 days</strong>. Renew now to keep your access to our EUR/USD decision engine.';
        break;
      case '1_day':
        targetDate = new Date(now.getTime() + 1 * 24 * 60 * 60 * 1000);
        emailSubject = 'Your ForexTell AI subscription expires tomorrow!';
        emailHeadline = 'Last Day to Renew!';
        emailMessage = 'Your ForexTell AI subscription will expire <strong>tomorrow</strong>. Don\'t miss out on your daily trading bias!';
        break;
      case 'expired':
        targetDate = now;
        emailSubject = 'Your ForexTell AI subscription has expired';
        emailHeadline = 'Your Subscription Has Expired';
        emailMessage = 'Your ForexTell AI subscription has expired. Renew now to continue receiving your daily EUR/USD trading bias.';
        break;
      default:
        throw new Error('Invalid reminder type');
    }

    // Calculate date range for finding subscriptions
    const startOfDay = new Date(targetDate);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(targetDate);
    endOfDay.setHours(23, 59, 59, 999);

    // Find subscriptions expiring on target date
    const { data: subscriptions, error: fetchError } = await supabase
      .from('subscriptions')
      .select(`
        id,
        user_id,
        plan_type,
        current_period_end,
        profiles!inner(email, full_name)
      `)
      .eq('status', 'active')
      .neq('plan_type', 'lifetime')
      .gte('current_period_end', startOfDay.toISOString())
      .lte('current_period_end', endOfDay.toISOString());

    if (fetchError) {
      console.error('Error fetching subscriptions:', fetchError);
      throw fetchError;
    }

    console.log(`Found ${subscriptions?.length || 0} subscriptions expiring for ${type} reminder`);

    const emailResults = [];

    for (const sub of subscriptions || []) {
      const profile = sub.profiles as any;
      const userEmail = profile?.email;
      
      if (!userEmail) {
        console.log(`No email found for user ${sub.user_id}`);
        continue;
      }

      const planName = getPlanDisplayName(sub.plan_type);
      const displayName = profile?.full_name || 'Trader';

      const emailHtml = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc;">
          <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 16px; overflow: hidden; margin-top: 40px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%); padding: 40px 32px; text-align: center;">
              <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 700;">${emailHeadline}</h1>
            </div>
            
            <!-- Body -->
            <div style="padding: 40px 32px;">
              <p style="font-size: 16px; color: #1e293b; line-height: 1.6; margin: 0 0 24px 0;">
                Hi ${displayName},
              </p>
              
              <p style="font-size: 16px; color: #475569; line-height: 1.6; margin: 0 0 24px 0;">
                ${emailMessage}
              </p>
              
              <div style="background-color: #fef3c7; border-radius: 12px; padding: 20px; margin-bottom: 24px; border-left: 4px solid #f59e0b;">
                <p style="font-size: 14px; color: #92400e; margin: 0; line-height: 1.5;">
                  <strong>Current Plan:</strong> ${planName}<br>
                  <strong>Expires:</strong> ${new Date(sub.current_period_end).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              
              <!-- CTA Button -->
              <div style="text-align: center; margin: 32px 0;">
                <a href="https://forextellai.lovable.app/#pricing" style="display: inline-block; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); color: #ffffff; text-decoration: none; padding: 16px 32px; border-radius: 50px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);">
                  Renew Subscription →
                </a>
              </div>
              
              <p style="font-size: 14px; color: #64748b; line-height: 1.6; margin: 24px 0 0 0;">
                Questions? Just reply to this email.
              </p>
            </div>
            
            <!-- Footer -->
            <div style="background-color: #f8fafc; padding: 24px 32px; text-align: center; border-top: 1px solid #e2e8f0;">
              <p style="font-size: 12px; color: #94a3b8; margin: 0;">
                © ${new Date().getFullYear()} ForexTell AI. Not financial advice.
              </p>
            </div>
          </div>
        </body>
        </html>
      `;

      try {
        const emailResponse = await sendEmail(userEmail, emailSubject, emailHtml);
        console.log(`Reminder email sent to ${userEmail}:`, emailResponse);
        emailResults.push({ email: userEmail, success: true });
      } catch (emailError: any) {
        console.error(`Failed to send email to ${userEmail}:`, emailError);
        emailResults.push({ email: userEmail, success: false, error: emailError.message });
      }
    }

    return new Response(
      JSON.stringify({ 
        success: true, 
        type,
        emailsSent: emailResults.filter(r => r.success).length,
        emailsFailed: emailResults.filter(r => !r.success).length,
        results: emailResults 
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  } catch (error: any) {
    console.error("Error in send-subscription-reminder:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
});