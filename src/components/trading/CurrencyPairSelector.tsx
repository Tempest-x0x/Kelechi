import { Check, ChevronDown, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { CurrencyPair } from '@/hooks/useCurrencyPairs';
import { cn } from '@/lib/utils';

interface CurrencyPairSelectorProps {
  pairs: CurrencyPair[];
  selectedSymbol: string;
  onSelect: (symbol: string) => void;
  isLoading?: boolean;
}

export function CurrencyPairSelector({
  pairs,
  selectedSymbol,
  onSelect,
  isLoading = false
}: CurrencyPairSelectorProps) {
  const selectedPair = pairs.find(p => p.symbol === selectedSymbol);

  if (isLoading) {
    return (
      <Button variant="outline" disabled className="min-w-[140px]">
        <Loader2 className="h-4 w-4 animate-spin mr-2" />
        Loading...
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="min-w-[140px] justify-between gap-2">
          <span className="font-mono font-semibold">
            {selectedPair?.symbol || selectedSymbol}
          </span>
          <ChevronDown className="h-4 w-4 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[200px] max-h-[300px] overflow-y-auto">
        {pairs.map((pair) => (
          <DropdownMenuItem
            key={pair.id}
            onClick={() => onSelect(pair.symbol)}
            className={cn(
              "cursor-pointer flex items-center justify-between",
              pair.symbol === selectedSymbol && "bg-accent"
            )}
          >
            <div className="flex flex-col">
              <span className="font-mono font-medium">{pair.symbol}</span>
              <span className="text-xs text-muted-foreground">{pair.display_name}</span>
            </div>
            {pair.symbol === selectedSymbol && (
              <Check className="h-4 w-4 text-primary" />
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
