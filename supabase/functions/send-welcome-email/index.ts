import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY");

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

interface WelcomeEmailRequest {
  email: string;
  name?: string;
  planType: string;
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
    const { email, name, planType }: WelcomeEmailRequest = await req.json();
    
    console.log(`Sending welcome email to ${email} for ${planType} plan`);

    const planName = getPlanDisplayName(planType);
    const displayName = name || 'Trader';

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
          <div style="background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); padding: 40px 32px; text-align: center;">
            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 700;">Welcome to ForexTell AI</h1>
            <p style="color: rgba(255, 255, 255, 0.9); margin-top: 8px; font-size: 16px;">Your ${planName} plan is now active</p>
          </div>
          
          <!-- Body -->
          <div style="padding: 40px 32px;">
            <p style="font-size: 16px; color: #1e293b; line-height: 1.6; margin: 0 0 24px 0;">
              Hi ${displayName},
            </p>
            
            <p style="font-size: 16px; color: #475569; line-height: 1.6; margin: 0 0 24px 0;">
              Thank you for subscribing to ForexTell AI! Your <strong>${planName}</strong> plan is now active, and you have full access to our EUR/USD decision engine.
            </p>
            
            <!-- Quick Start -->
            <div style="background-color: #f8fafc; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
              <h2 style="font-size: 18px; color: #1e293b; margin: 0 0 16px 0; font-weight: 600;">Quick Start Guide</h2>
              
              <div style="margin-bottom: 16px;">
                <div style="display: flex; align-items: flex-start; margin-bottom: 12px;">
                  <span style="background-color: #2563eb; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; margin-right: 12px; flex-shrink: 0;">1</span>
                  <span style="color: #475569; font-size: 14px; line-height: 1.5;"><strong>Check your morning bias</strong> - Get your directional bias before market opens</span>
                </div>
                <div style="display: flex; align-items: flex-start; margin-bottom: 12px;">
                  <span style="background-color: #2563eb; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; margin-right: 12px; flex-shrink: 0;">2</span>
                  <span style="color: #475569; font-size: 14px; line-height: 1.5;"><strong>Note your levels</strong> - Entry, invalidation, and targets</span>
                </div>
                <div style="display: flex; align-items: flex-start;">
                  <span style="background-color: #2563eb; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; margin-right: 12px; flex-shrink: 0;">3</span>
                  <span style="color: #475569; font-size: 14px; line-height: 1.5;"><strong>Trade with rules</strong> - Let invalidation protect your capital</span>
                </div>
              </div>
            </div>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 32px 0;">
              <a href="https://forextellai.lovable.app/dashboard" style="display: inline-block; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); color: #ffffff; text-decoration: none; padding: 16px 32px; border-radius: 50px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);">
                Go to Dashboard →
              </a>
            </div>
            
            <p style="font-size: 14px; color: #64748b; line-height: 1.6; margin: 24px 0 0 0;">
              If you have any questions, just reply to this email. We're here to help!
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

    const emailResponse = await sendEmail(
      email,
      `Welcome to ForexTell AI - ${planName} Plan Activated! 🎉`,
      emailHtml
    );

    console.log("Welcome email sent successfully:", emailResponse);

    return new Response(JSON.stringify({ success: true, data: emailResponse }), {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  } catch (error: any) {
    console.error("Error sending welcome email:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
});