# Instructions to Run Analysis and Generate CSV Files

## To Run the Original Improved Analysis

Run the `comprehensive_stock_analysis.py` script which uses the original improved exit logic:

```bash
cd analysis
python comprehensive_stock_analysis.py
```

Or if using venv:

```bash
cd analysis
..\venv\Scripts\python.exe comprehensive_stock_analysis.py
```

## What This Script Does

1. **Analyzes all 5 stocks**: GNPX, MLTX, VLN, INBS, ANPA
2. **Time Range**: From 4 AM ET onwards
3. **Uses Original Improved Exit Logic** (from comprehensive_stock_analysis.py):
   - 20-minute minimum hold time
   - Dynamic trailing stops (7% for 0-30 min, 10% for 30+ min)
   - Strong reversal (3+ signals required)
   - Profit target only after 30+ min AND 20%+ profit

## Output

The script will generate CSV files in the `analysis/` directory:
- `GNPX_detailed_trades.csv`
- `MLTX_detailed_trades.csv`
- `VLN_detailed_trades.csv`
- `INBS_detailed_trades.csv`
- `ANPA_detailed_trades.csv`

Each CSV contains:
- Entry_Time, Entry_Price
- Exit_Time, Exit_Price
- Pattern, Score, Confidence
- Exit_Reason
- Hold_Time_Min
- PnL_Pct, PnL_Dollar

## Expected Results (from EXIT_LOGIC_IMPROVEMENT_COMPARISON.md)

Based on the improved exit logic, expected results:
- **GNPX**: 2 trades, 100% win rate, 8.49% total P&L
- **MLTX**: 4 trades, 50% win rate, 4.86% total P&L
- **VLN**: 4 trades, 100% win rate, 19.74% total P&L
- **INBS**: 5 trades, 40% win rate, -6.35% total P&L
- **ANPA**: 9 trades, 55.6% win rate, 48.12% total P&L

**Overall**: 24 trades, 69.2% win rate, 74.86% total P&L

## Comparison

After running, compare the generated CSV files with the expected results to verify the implementation matches the improved exit logic.
