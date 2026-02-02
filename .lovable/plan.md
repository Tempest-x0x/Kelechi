# Confidence & R:R Fix - COMPLETED ✅

## Changes Made (2026-02-02)

### 1. Fixed Confidence Calculation (scan-opportunities)
**Before:** Arbitrary formula: `50 + (tier1_count * 12) + combo_bonuses` → resulted in 95% confidence
**After:** Data-driven: weighted average of actual win rates from detected patterns → realistic 45-58% confidence

### 2. Reduced R:R from 1:3 to 1:2.2 (both functions)
**Before:** TP1 = 3.0x ATR, TP2 = 4.5x ATR
**After:** TP1 = 2.2x ATR, TP2 = 3.0x ATR

### Files Modified
- `supabase/functions/scan-opportunities/index.ts` - Confidence calc + R:R
- `supabase/functions/generate-prediction/index.ts` - R:R only

### Expected Results
1. **Realistic confidence scores**: 52% instead of 95%
2. **Achievable TP targets**: 2.2x ATR vs 3x ATR
3. **Better win rate**: Targets aligned with pattern statistics methodology
