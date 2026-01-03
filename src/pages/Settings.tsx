import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ThemeProvider, useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  ArrowLeft, 
  Calendar,
  CreditCard,
  Crown,
  Loader2,
  Moon,
  Sun,
  Activity,
  AlertTriangle
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useSubscription } from '@/hooks/useSubscription';
import { toast } from 'sonner';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

const ThemeToggle = () => {
  const { theme, setTheme } = useTheme();
  
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      className="rounded-full"
    >
      <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
    </Button>
  );
};

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

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'active':
      return 'bg-[hsl(var(--bullish))]/10 text-[hsl(var(--bullish))] border-[hsl(var(--bullish))]/20';
    case 'cancelled':
      return 'bg-amber-500/10 text-amber-600 border-amber-500/20';
    case 'expired':
      return 'bg-[hsl(var(--bearish))]/10 text-[hsl(var(--bearish))] border-[hsl(var(--bearish))]/20';
    default:
      return 'bg-muted text-muted-foreground';
  }
};

const Settings = () => {
  const navigate = useNavigate();
  const { subscription, isLoading, refreshSubscription } = useSubscription();
  const [user, setUser] = useState<any>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/auth');
        return;
      }
      setUser(session.user);
    });

    const { data: { subscription: authSub } } = supabase.auth.onAuthStateChange((_, session) => {
      if (!session) {
        navigate('/auth');
      }
    });

    return () => authSub.unsubscribe();
  }, [navigate]);

  const handleCancelSubscription = async () => {
    if (!subscription?.paystack_subscription_code) {
      toast.error('No active subscription to cancel');
      return;
    }

    setIsCancelling(true);
    try {
      const { data, error } = await supabase.functions.invoke('cancel-subscription', {
        body: {
          subscriptionCode: subscription.paystack_subscription_code,
        },
      });

      if (error) throw error;

      toast.success('Subscription cancelled. You will retain access until the end of your billing period.');
      refreshSubscription();
    } catch (err: any) {
      console.error('Cancel subscription error:', err);
      toast.error(err.message || 'Failed to cancel subscription');
    } finally {
      setIsCancelling(false);
    }
  };

  if (isLoading) {
    return (
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
        <div className="min-h-screen bg-background flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <header className="border-b border-border bg-background/80 backdrop-blur-lg sticky top-0 z-50">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="h-16 lg:h-20 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={() => navigate('/dashboard')}
                  className="rounded-full"
                >
                  <ArrowLeft className="h-5 w-5" />
                </Button>
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary rounded-xl">
                    <Activity className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <div>
                    <h1 className="font-semibold text-lg leading-none">Account Settings</h1>
                    <p className="text-xs text-muted-foreground hidden sm:block">Manage your subscription</p>
                  </div>
                </div>
              </div>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
          <div className="max-w-2xl mx-auto space-y-8">
            {/* Profile Section */}
            <section className="premium-card p-6 lg:p-8">
              <h2 className="text-lg font-semibold mb-4">Profile</h2>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-muted-foreground">Email</label>
                  <p className="font-medium">{user?.email}</p>
                </div>
              </div>
            </section>

            {/* Subscription Section */}
            <section className="premium-card p-6 lg:p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold">Subscription</h2>
                {subscription && (
                  <Badge className={`${getStatusColor(subscription.status)} border`}>
                    {subscription.status.charAt(0).toUpperCase() + subscription.status.slice(1)}
                  </Badge>
                )}
              </div>

              {subscription ? (
                <div className="space-y-6">
                  {/* Current Plan */}
                  <div className="flex items-center gap-4 p-4 bg-muted/30 rounded-xl">
                    <div className="p-3 bg-primary/10 rounded-xl">
                      <Crown className="h-6 w-6 text-primary" />
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-lg">{getPlanDisplayName(subscription.plan_type)} Plan</p>
                      <p className="text-sm text-muted-foreground">
                        {subscription.plan_type === 'lifetime' ? 'Lifetime access' : 'Monthly subscription'}
                      </p>
                    </div>
                  </div>

                  {/* Billing Details */}
                  {subscription.plan_type !== 'lifetime' && subscription.current_period_end && (
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 text-sm">
                        <Calendar className="h-4 w-4 text-muted-foreground" />
                        <span className="text-muted-foreground">Next billing date:</span>
                        <span className="font-medium">
                          {new Date(subscription.current_period_end).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </span>
                      </div>
                      
                      {subscription.amount && (
                        <div className="flex items-center gap-3 text-sm">
                          <CreditCard className="h-4 w-4 text-muted-foreground" />
                          <span className="text-muted-foreground">Amount:</span>
                          <span className="font-medium">
                            ₦{(subscription.amount / 100).toLocaleString()}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="pt-4 border-t border-border space-y-3">
                    {/* Upgrade Button (for retail users) */}
                    {subscription.plan_type === 'retail' && subscription.status === 'active' && (
                      <Button 
                        onClick={() => navigate('/#pricing')}
                        className="w-full rounded-full"
                      >
                        Upgrade to Funded Trader
                      </Button>
                    )}

                    {/* Cancel Button */}
                    {subscription.status === 'active' && subscription.plan_type !== 'lifetime' && subscription.paystack_subscription_code && (
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="outline" className="w-full rounded-full text-muted-foreground">
                            Cancel Subscription
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle className="flex items-center gap-2">
                              <AlertTriangle className="h-5 w-5 text-amber-500" />
                              Cancel Subscription?
                            </AlertDialogTitle>
                            <AlertDialogDescription className="space-y-3">
                              <p>
                                Are you sure you want to cancel your {getPlanDisplayName(subscription.plan_type)} subscription?
                              </p>
                              <p>
                                You will continue to have access until{' '}
                                <strong>
                                  {new Date(subscription.current_period_end!).toLocaleDateString('en-US', {
                                    month: 'long',
                                    day: 'numeric',
                                    year: 'numeric'
                                  })}
                                </strong>.
                              </p>
                              <p className="text-sm">
                                After this date, you will lose access to the trading dashboard and daily bias.
                              </p>
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="rounded-full">Keep Subscription</AlertDialogCancel>
                            <AlertDialogAction 
                              onClick={handleCancelSubscription}
                              disabled={isCancelling}
                              className="rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              {isCancelling ? (
                                <>
                                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                  Cancelling...
                                </>
                              ) : (
                                'Yes, Cancel'
                              )}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    )}

                    {/* Cancelled status message */}
                    {subscription.status === 'cancelled' && subscription.current_period_end && (
                      <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
                        <p className="text-sm text-amber-700 dark:text-amber-400">
                          Your subscription has been cancelled. You will retain access until{' '}
                          <strong>
                            {new Date(subscription.current_period_end).toLocaleDateString('en-US', {
                              month: 'long',
                              day: 'numeric',
                              year: 'numeric'
                            })}
                          </strong>.
                        </p>
                        <Button 
                          onClick={() => navigate('/#pricing')}
                          className="mt-4 w-full rounded-full"
                          size="sm"
                        >
                          Resubscribe
                        </Button>
                      </div>
                    )}

                    {/* Expired status */}
                    {subscription.status === 'expired' && (
                      <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-xl">
                        <p className="text-sm text-destructive">
                          Your subscription has expired. Resubscribe to regain access to ForexTell AI.
                        </p>
                        <Button 
                          onClick={() => navigate('/#pricing')}
                          className="mt-4 w-full rounded-full"
                          size="sm"
                        >
                          View Plans
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-4">You don't have an active subscription.</p>
                  <Button onClick={() => navigate('/#pricing')} className="rounded-full">
                    View Plans
                  </Button>
                </div>
              )}
            </section>

            {/* Sign Out */}
            <section className="premium-card p-6 lg:p-8">
              <h2 className="text-lg font-semibold mb-4">Account</h2>
              <Button 
                variant="outline" 
                className="w-full rounded-full"
                onClick={async () => {
                  await supabase.auth.signOut();
                  navigate('/');
                }}
              >
                Sign Out
              </Button>
            </section>
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
};

export default Settings;