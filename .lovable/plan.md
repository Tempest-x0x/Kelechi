
# Adaptive Tier Thresholds for Weak-Performing Currency Pairs

## The Problem

The current system requires at least one "Tier 1" pattern (win rate >52%) to generate a signal. However, **8 of 12 currency pairs have no patterns exceeding 52%**, making it impossible for them to ever trigger signals.

## Solution: Adaptive Tier Classification

Instead of a fixed 52% threshold for all pairs, implement a **two-tier threshold system**:

1. **Strong Pairs** (best pattern >=52%): Keep strict 52% Tier 1 threshold
2. **Weak Pairs** (best pattern 50.5%-52%): Use 51% as Tier 1 threshold

This maintains quality while allowing pairs with statistically positive (but weaker) patterns to participate.

## Technical Changes

### File: `supabase/functions/scan-opportunities/index.ts`

**Modify `getDynamicPatternWeight` function** to accept a pair-specific threshold:

```typescript
// NEW: Get pair-specific Tier 1 threshold
function getTier1Threshold(symbol: string, patternStats: any[]): number {
  const symbolStats = patternStats.filter(p => p.symbol === symbol);
  if (symbolStats.length === 0) return 52; // Default
  
  const bestWinRate = Math.max(...symbolStats.map(p => 
    p.win_rate_24h || p.win_rate_12h || 50
  ));
  
  // If best pattern is 51-52%, lower threshold to 51%
  // If best pattern is 50.5-51%, lower threshold to 50.5%
  // Below 50.5%, keep at 52% (effectively disable)
  if (bestWinRate >= 52) return 52;
  if (bestWinRate >= 51) return 51;
  if (bestWinRate >= 50.5) return 50.5;
  return 52; // Disable pairs below 50.5%
}
```

**Update tier classification** in `getDynamicPatternWeight`:

```typescript
function getDynamicPatternWeight(
  patternName: PatternName, 
  winRate: number | null,
  tier1Threshold: number = 52  // NEW parameter
): { weight: number; tier: number } {
  // ... existing code ...
  
  // Dynamic tier based on ACTUAL win rate with pair-specific threshold
  if (winRate > tier1Threshold) {
    tier = 1;
    weight = 1.3 + ((winRate - tier1Threshold) * 0.1);
  } else if (winRate >= 50) {
    tier = 2;
    weight = 1.0;
  } 
  // ... rest unchanged
}
```

**Update `analyzeOpportunity` function** to pass the threshold:

```typescript
function analyzeOpportunity(...) {
  // Calculate pair-specific threshold
  const tier1Threshold = getTier1Threshold(symbol, patternStats);
  console.log(`[${symbol}] Using Tier 1 threshold: ${tier1Threshold}%`);
  
  // Pass threshold to each pattern detection
  if (indicators.rsi < 30) {
    const { winRate, stat } = getPatternWinRate('rsi_oversold', 'BUY');
    const { weight, tier } = getDynamicPatternWeight('rsi_oversold', winRate, tier1Threshold);
    // ... rest unchanged
  }
  // Apply same change to all pattern detections
}
```

## Expected Results After Fix

| Pair | Best Win Rate | New Threshold | Can Signal? |
|------|---------------|---------------|-------------|
| XAU/USD | 53.48% | 52% | ✅ Yes |
| GBP/JPY | 52.50% | 52% | ✅ Yes |
| EUR/USD | 52.40% | 52% | ✅ Yes |
| USD/JPY | 52.03% | 52% | ✅ Yes |
| GBP/USD | 51.97% | 51% | ✅ **Now enabled** |
| EUR/CHF | 51.92% | 51% | ✅ **Now enabled** |
| EUR/JPY | 51.80% | 51% | ✅ **Now enabled** |
| USD/CHF | 51.60% | 51% | ✅ **Now enabled** |
| EUR/GBP | 51.54% | 51% | ✅ **Now enabled** |
| AUD/JPY | 51.52% | 51% | ✅ **Now enabled** |
| AUD/USD | 51.50% | 51% | ✅ **Now enabled** |
| USD/CAD | 51.31% | 51% | ✅ **Now enabled** |

## Why This Is Safe

1. **All enabled pairs still have >50% edge** - statistically profitable over time
2. **Strong pairs maintain strict criteria** - no degradation for EUR/USD, XAU/USD etc.
3. **Confidence scores remain data-driven** - shows actual 51-53% win rate, not inflated
4. **Better diversification** - signals across more pairs reduces concentration risk

## Files to Modify

| File | Changes |
|------|---------|
| `supabase/functions/scan-opportunities/index.ts` | Add adaptive threshold logic |

## Deployment

After changes, redeploy the `scan-opportunities` edge function and monitor logs to verify all 12 pairs can now generate signals when conditions are met.
