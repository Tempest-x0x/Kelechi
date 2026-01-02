import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Zap, Lock } from 'lucide-react';

export const UpgradeModal = () => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(true);

  const features = [
    'Daily directional bias',
    'Entry & invalidation levels',
    'Target levels',
    'Pattern detection',
    'Historical track record',
  ];

  const handleViewPricing = () => {
    setOpen(false);
    navigate('/#pricing');
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-2 mb-2">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Lock className="h-5 w-5 text-primary" />
            </div>
            <Badge variant="secondary">
              <Zap className="h-3 w-3 mr-1" />
              Premium Feature
            </Badge>
          </div>
          <DialogTitle className="text-xl">Upgrade to Access Full Dashboard</DialogTitle>
          <DialogDescription>
            You're viewing the dashboard in preview mode. Subscribe to unlock all features.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 my-4">
          {features.map((feature, index) => (
            <div key={index} className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-bullish" />
              <span className="text-sm">{feature}</span>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-3">
          <Button onClick={handleViewPricing} className="w-full">
            View Pricing Plans
          </Button>
          <Button variant="ghost" onClick={() => setOpen(false)} className="w-full">
            Continue Preview
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Starting at ₦76,000/month • Cancel anytime
        </p>
      </DialogContent>
    </Dialog>
  );
};
