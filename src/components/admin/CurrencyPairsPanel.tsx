import { useState } from 'react';
import { Plus, Check, X, Upload, Loader2, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useCurrencyPairs, CurrencyPair } from '@/hooks/useCurrencyPairs';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

export default function CurrencyPairsPanel() {
  const { pairs, isLoading, refetch, togglePairActive, addPair } = useCurrencyPairs();
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newPair, setNewPair] = useState({
    symbol: '',
    display_name: '',
    pip_value: 0.0001
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [importingStats, setImportingStats] = useState<string | null>(null);

  const handleToggleActive = async (pair: CurrencyPair) => {
    try {
      await togglePairActive(pair.id, !pair.is_active);
      toast.success(`${pair.symbol} ${!pair.is_active ? 'activated' : 'deactivated'}`);
    } catch (err) {
      toast.error('Failed to update pair status');
    }
  };

  const handleAddPair = async () => {
    if (!newPair.symbol || !newPair.display_name) {
      toast.error('Please fill in all required fields');
      return;
    }

    setIsSubmitting(true);
    try {
      await addPair({
        symbol: newPair.symbol.toUpperCase(),
        display_name: newPair.display_name,
        pip_value: newPair.pip_value,
        is_active: true,
        has_pattern_stats: false
      });
      toast.success(`${newPair.symbol} added successfully`);
      setIsAddDialogOpen(false);
      setNewPair({ symbol: '', display_name: '', pip_value: 0.0001 });
    } catch (err: any) {
      toast.error(err?.message || 'Failed to add pair');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleImportStats = async (pair: CurrencyPair) => {
    setImportingStats(pair.id);
    try {
      const { data, error } = await supabase.functions.invoke('import-pattern-stats', {
        body: { symbol: pair.symbol }
      });

      if (error) throw error;
      if (!data.success) throw new Error(data.error || 'Import failed');

      toast.success(`Imported ${data.imported} patterns for ${pair.symbol}`);
      await refetch();
    } catch (err: any) {
      toast.error(err?.message || 'Failed to import pattern statistics');
    } finally {
      setImportingStats(null);
    }
  };

  const handleFetchPriceData = async (pair: CurrencyPair) => {
    try {
      toast.info(`Fetching price data for ${pair.symbol}...`);
      const { error } = await supabase.functions.invoke('fetch-forex-data', {
        body: { symbol: pair.symbol, timeframe: '1h', outputsize: 200 }
      });

      if (error) throw error;
      toast.success(`Price data updated for ${pair.symbol}`);
    } catch (err: any) {
      toast.error(err?.message || 'Failed to fetch price data');
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Currency Pairs</CardTitle>
            <CardDescription>
              Manage which currency pairs are available for scanning and analysis
            </CardDescription>
          </div>
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Pair
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Currency Pair</DialogTitle>
                <DialogDescription>
                  Add a new currency pair for scanning. Make sure you have pattern statistics data for accurate signals.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    placeholder="e.g., EUR/GBP"
                    value={newPair.symbol}
                    onChange={(e) => setNewPair({ ...newPair, symbol: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="display_name">Display Name</Label>
                  <Input
                    id="display_name"
                    placeholder="e.g., Euro / British Pound"
                    value={newPair.display_name}
                    onChange={(e) => setNewPair({ ...newPair, display_name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="pip_value">Pip Value</Label>
                  <Input
                    id="pip_value"
                    type="number"
                    step="0.0001"
                    value={newPair.pip_value}
                    onChange={(e) => setNewPair({ ...newPair, pip_value: parseFloat(e.target.value) })}
                  />
                  <p className="text-xs text-muted-foreground">
                    Use 0.0001 for most pairs, 0.01 for JPY pairs and Gold
                  </p>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleAddPair} disabled={isSubmitting}>
                  {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Add Pair
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Display Name</TableHead>
              <TableHead>Pip Value</TableHead>
              <TableHead>Pattern Stats</TableHead>
              <TableHead>Active</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pairs.map((pair) => (
              <TableRow key={pair.id}>
                <TableCell className="font-mono font-medium">{pair.symbol}</TableCell>
                <TableCell className="text-muted-foreground">{pair.display_name}</TableCell>
                <TableCell className="font-mono">{pair.pip_value}</TableCell>
                <TableCell>
                  {pair.has_pattern_stats ? (
                    <Badge variant="default" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">
                      <Check className="h-3 w-3 mr-1" />
                      Available
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-muted-foreground">
                      <X className="h-3 w-3 mr-1" />
                      Missing
                    </Badge>
                  )}
                </TableCell>
                <TableCell>
                  <Switch
                    checked={pair.is_active}
                    onCheckedChange={() => handleToggleActive(pair)}
                  />
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleFetchPriceData(pair)}
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Fetch Data
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleImportStats(pair)}
                      disabled={importingStats === pair.id}
                    >
                      {importingStats === pair.id ? (
                        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      ) : (
                        <Upload className="h-3 w-3 mr-1" />
                      )}
                      Import Stats
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
