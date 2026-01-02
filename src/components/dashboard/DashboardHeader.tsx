import { memo } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Moon, Sun, LogOut, User, CreditCard } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useTheme } from 'next-themes';
import { User as SupabaseUser } from '@supabase/supabase-js';

interface Subscription {
  plan_type: string;
  status: string;
}

interface DashboardHeaderProps {
  user: SupabaseUser | null;
  subscription: Subscription | null;
  onLogout: () => void;
}

export const DashboardHeader = memo(function DashboardHeader({
  user,
  subscription,
  onLogout,
}: DashboardHeaderProps) {
  const { theme, setTheme } = useTheme();

  const getPlanBadge = () => {
    if (!subscription || subscription.status !== 'active') {
      return (
        <Badge variant="secondary" className="text-xs">
          Free
        </Badge>
      );
    }

    const planLabels: Record<string, string> = {
      retail: 'Retail',
      funded: 'Funded',
      lifetime: 'Lifetime',
    };

    return (
      <Badge className="bg-primary text-primary-foreground text-xs">
        {planLabels[subscription.plan_type] || subscription.plan_type}
      </Badge>
    );
  };

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-lg">
              <Activity className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-semibold text-lg leading-none">ForexTell AI</h1>
              <p className="text-xs text-muted-foreground">EUR/USD Decision Engine</p>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-secondary rounded-full text-sm">
            <span className="w-2 h-2 bg-bullish rounded-full animate-pulse" />
            <span className="text-muted-foreground">Market Open</span>
          </div>

          {getPlanBadge()}

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <User className="h-4 w-4" />
                <span className="hidden sm:inline max-w-[120px] truncate">
                  {user?.email?.split('@')[0]}
                </span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <div className="flex flex-col">
                  <span className="font-medium">Account</span>
                  <span className="text-xs text-muted-foreground truncate">
                    {user?.email}
                  </span>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link to="/track-record" className="cursor-pointer">
                  <Activity className="mr-2 h-4 w-4" />
                  Track Record
                </Link>
              </DropdownMenuItem>
              {(!subscription || subscription.status !== 'active') && (
                <DropdownMenuItem asChild>
                  <Link to="/#pricing" className="cursor-pointer">
                    <CreditCard className="mr-2 h-4 w-4" />
                    Upgrade Plan
                  </Link>
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onLogout} className="cursor-pointer text-destructive">
                <LogOut className="mr-2 h-4 w-4" />
                Log Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
});
