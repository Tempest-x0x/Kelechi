import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Loader2, Star } from 'lucide-react';
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
    // Check if user is logged in
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
    <Card className={`relative ${popular ? 'border-primary shadow-lg shadow-primary/20' : ''}`}>
      {popular && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2">
          <Badge className="bg-primary text-primary-foreground">
            <Star className="h-3 w-3 mr-1" />
            Most Popular
          </Badge>
        </div>
      )}
      
      <CardHeader className="text-center pb-2">
        <CardTitle className="text-xl">{name}</CardTitle>
        <div className="mt-4">
          <span className="text-4xl font-bold">${priceUSD}</span>
          <span className="text-muted-foreground">/{period === 'month' ? 'mo' : 'once'}</span>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          ₦{priceNGN.toLocaleString()}
        </p>
      </CardHeader>

      <CardContent className="space-y-3">
        {features.map((feature, index) => (
          <div key={index} className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-bullish mt-0.5 shrink-0" />
            <span className="text-sm">{feature}</span>
          </div>
        ))}
      </CardContent>

      <CardFooter>
        <Button 
          className="w-full" 
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
      </CardFooter>
    </Card>
  );
};
