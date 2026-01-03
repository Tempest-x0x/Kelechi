import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Check, Loader2, Star } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

interface PricingCardProps {
  name: string;
  priceUSD: number;
  priceNGN: number;
  period: 'month' | 'one-time';
  features: string[];
  planType: 'retail' | 'funded' | 'lifetime';
  popular?: boolean;
}

export const PricingCard = ({
  name,
  priceUSD,
  priceNGN,
  period,
  features,
  planType,
  popular = false,
}: PricingCardProps) => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleSubscribe = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    
    if (!session) {
      toast.info('Please log in or create an account first');
      navigate('/auth');
      return;
    }

    setIsLoading(true);

    try {
      const { data, error } = await supabase.functions.invoke('paystack-initialize', {
        body: {
          plan: planType,
          email: session.user.email,
          userId: session.user.id,
        },
      });

      if (error) throw error;

      if (data?.authorization_url) {
        window.location.href = data.authorization_url;
      } else {
        throw new Error('Failed to get payment URL');
      }
    } catch (err: any) {
      console.error('Payment initialization error:', err);
      toast.error(err.message || 'Failed to initialize payment');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`relative premium-card p-8 flex flex-col ${
      popular 
        ? 'border-primary shadow-primary-glow ring-1 ring-primary/20' 
        : ''
    }`}>
      {popular && (
        <div className="absolute -top-3.5 left-1/2 -translate-x-1/2">
          <Badge className="bg-primary text-primary-foreground px-4 py-1.5 rounded-full shadow-lg">
            <Star className="h-3.5 w-3.5 mr-1.5 fill-current" />
            Most Popular
          </Badge>
        </div>
      )}
      
      <div className="text-center mb-8">
        <h3 className="text-lg font-semibold mb-4">{name}</h3>
        <div className="mb-2">
          <span className="text-5xl font-bold">${priceUSD}</span>
          <span className="text-muted-foreground ml-1">/{period === 'month' ? 'mo' : 'once'}</span>
        </div>
        <p className="text-sm text-muted-foreground">
          ₦{priceNGN.toLocaleString()}
        </p>
      </div>

      <div className="space-y-4 flex-1 mb-8">
        {features.map((feature, index) => (
          <div key={index} className="flex items-start gap-3">
            <div className="p-0.5 rounded-full bg-[hsl(var(--bullish))]/10 mt-0.5">
              <Check className="h-3.5 w-3.5 text-[hsl(var(--bullish))]" />
            </div>
            <span className="text-sm leading-relaxed">{feature}</span>
          </div>
        ))}
      </div>

      <Button 
        className={`w-full h-12 rounded-full text-base font-medium transition-all ${
          popular 
            ? 'shadow-primary-glow hover:shadow-lg' 
            : ''
        }`}
        variant={popular ? 'default' : 'outline'}
        onClick={handleSubscribe}
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Processing...
          </>
        ) : (
          'Get Started'
        )}
      </Button>
    </div>
  );
};