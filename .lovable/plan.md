
# Fix: Unblock Signals & Ensure Telegram Delivery

## Root Cause

The Telegram notification code (lines 976-1002) is correctly implemented but **never executes** because all signals are blocked earlier at line 855:

```typescript
if (!analysis.signal || analysis.confidence < 60 || analysis.reasons.length < 2) {
  console.log(`[${symbol}] No high-probability opportunity detected`);
  return; // ← Telegram code never reached
}
```

With the recent fix making confidence scores realistic (max ~58%), this threshold blocks 100% of valid signals.

## Solution: Two Changes

### Change 1: Lower Confidence Threshold (Critical)

**File**: `supabase/functions/scan-opportunities/index.ts`  
**Line 855**

```typescript
// FROM:
if (!analysis.signal || analysis.confidence < 60 || analysis.reasons.length < 2)

// TO:
if (!analysis.signal || analysis.confidence < 50 || analysis.reasons.length < 2)
```

### Change 2: Add Modest Confluence Bonus

Restore the trading concept that multiple confirmations = better setup, but with conservative values.

**Location**: Around lines 695-710 (after weighted average calculation)

```typescript
// After calculating base confidence from weighted average:
let confidence = totalWeight > 0 ? weightedSum / totalWeight : 50;

// Add modest confluence bonus for multiple Tier 1 patterns
const tier1Patterns = patternsWithWinRates.filter(p => p.tier === 1);
if (tier1Patterns.length > 1) {
  const confluenceBonus = Math.min((tier1Patterns.length - 1) * 2, 6);
  confidence += confluenceBonus;
  reasons.push(`🎯 Confluence: ${tier1Patterns.length} confirming patterns (+${confluenceBonus}%)`);
}

// Cap at reasonable maximum
confidence = Math.min(58, Math.max(45, confidence));
```

## Expected Signal Flow After Fix

```text
Pattern Detection → Confidence Calculation (50-58%)
        ↓
Threshold Check (>= 50%) ✅ PASSES
        ↓
Insert to Database ✅
        ↓
Send Telegram Notification ✅ ← NOW REACHED
```

## Confidence Examples After Fix

| Setup | Base Win Rate | Confluence | Final Confidence |
|-------|---------------|------------|------------------|
| 1 Tier 1 pattern | 52% | +0% | 52% |
| 2 Tier 1 patterns | 52% | +2% | 54% |
| 3 Tier 1 patterns | 52% | +4% | 56% |
| 4 Tier 1 patterns | 52% | +6% (cap) | 58% |

## Files to Modify

| File | Changes |
|------|---------|
| `supabase/functions/scan-opportunities/index.ts` | Lower threshold to 50%, add +2%/pattern confluence bonus (capped at +6%) |

## Verification Steps

After deployment:
1. Check edge function logs for "Created new opportunity" messages
2. Check logs for "Telegram notification sent" messages
3. Confirm signal appears in Telegram channel
