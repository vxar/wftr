# Trade Simulator

A generic, reusable trade simulator for testing trading strategies with historical data from Webull API.

## Features

- **Configurable Parameters**: Easily adjust ticker, detection time, capital, and trading parameters
- **Webull API Integration**: Downloads and persists historical minute-level data
- **Minute-by-Minute Analysis**: Simulates real-time trading starting from detection time
- **Multiple Trading Strategies**: Includes surge, breakout, RSI bounce, MACD crossover, and MA crossover strategies
- **Risk Management**: Configurable stop loss, take profit, and minimum hold time
- **Comprehensive Reporting**: Detailed P&L analysis, win rates, and performance metrics
- **Data Persistence**: Saves historical data and results for future analysis

## Installation

1. Install required dependencies:
```bash
pip install pandas numpy requests
```

2. Ensure the existing Webull API code is available in `src/data/`:
   - `src/data/webull_data_api.py`
   - `src/data/WebullUtil.py`

The simulator uses the existing Webull API implementation from the trading bot codebase, not the public webull Python library.

## Usage

### Command Line Interface

Run the simulator from command line with various parameters:

```bash
py run_simulator.py --ticker AAPL --detection-time "2024-01-16 09:30:00" --capital 2500
```

### Available Parameters

- `--ticker, -t`: Stock ticker symbol (required)
- `--detection-time, -d`: Detection time in format 'YYYY-MM-DD HH:MM:SS' (required)
- `--capital, -c`: Initial capital (default: 2500.0)
- `--max-positions, -m`: Maximum positions (default: 1)
- `--stop-loss, -s`: Stop loss percentage (default: 0.04 = 4%)
- `--take-profit, -p`: Take profit percentage (default: 0.08 = 8%)
- `--commission, -x`: Commission per trade (default: 0.005 = 0.5%)
- `--min-hold, -o`: Minimum hold time in minutes (default: 10)
- `--data-folder, -f`: Data folder for storage (default: simulation_data)
- `--save-results, -r`: Save results to JSON file
- `--verbose, -v`: Enable verbose logging

### Example Commands

1. **Basic simulation**:
```bash
py run_simulator.py --ticker TSLA --detection-time "2024-01-16 10:15:00"
```

2. **Custom parameters**:
```bash
py run_simulator.py --ticker NVDA --detection-time "2024-01-16 09:45:00" --capital 5000 --stop-loss 0.03 --take-profit 0.06
```

3. **Save results**:
```bash
py run_simulator.py --ticker AAPL --detection-time "2024-01-16 09:30:00" --save-results
```

### Python API Usage

```python
from src.simulation.pure_trade_simulator import PureTradeSimulator as TradeSimulator, SimulationConfig

# Create configuration
config = SimulationConfig(
    ticker="AAPL",
    detection_time="2024-01-16 09:30:00",
    initial_capital=2500.0,
    max_positions=1,
    stop_loss_pct=0.04,
    take_profit_pct=0.08
)

# Run simulation
simulator = TradeSimulator(config)
result = simulator.run_simulation()

# Print and save results
simulator.print_results(result)
simulator.save_results(result)
```

## Trading Strategies

The simulator implements multiple entry strategies:

1. **Surge**: Price movement > 2% with volume spike > 2x average
2. **Breakout**: Price breaks above Bollinger Band with high volume
3. **RSI Bounce**: RSI moves from oversold (< 30) to neutral with volume
4. **MACD Cross**: MACD line crosses above signal line with volume
5. **MA Cross**: 5-period MA crosses above 15-period MA with volume

### Exit Conditions

- **Stop Loss**: Price drops below configured percentage
- **Take Profit**: Price rises above configured percentage
- **RSI Overbought**: RSI > 70 (after minimum hold time)
- **Low Volume**: Volume drops below 50% of average
- **MACD Cross Down**: MACD crosses below signal line
- **End of Simulation**: Final price if still in position

## Performance Metrics

The simulator provides comprehensive performance analysis:

- Total P&L and percentage return
- Win rate and trade counts
- Maximum drawdown
- Sharpe ratio
- Individual trade details with entry/exit reasons
- Holding time analysis

## Data Management

- Historical data is automatically downloaded using the existing Webull API codebase
- Data is cached in CSV files for future use
- Results are saved in JSON format
- Data folder structure: `simulation_data/TICKER_YYYYMMDD.csv`

## Configuration

### Default Settings

- Initial Capital: $2,500
- Maximum Positions: 1
- Stop Loss: 4%
- Take Profit: 8%
- Commission: 0.5%
- Minimum Hold Time: 10 minutes

### Customization

All parameters can be adjusted via command line arguments or configuration object:

```python
config = SimulationConfig(
    ticker="YOUR_TICKER",
    detection_time="YYYY-MM-DD HH:MM:SS",
    initial_capital=10000.0,  # Custom capital
    stop_loss_pct=0.05,       # 5% stop loss
    take_profit_pct=0.10,     # 10% take profit
    commission_per_trade=0.003,  # 0.3% commission
    min_hold_minutes=5        # 5 minute minimum hold
)
```

## Output Examples

### Console Output
```
Starting trade simulator for AAPL
Detection time: 2024-01-16 09:30:00
Initial capital: $2,500.00
Stop loss: 4.0%
Take profit: 8.0%
Commission: 0.5%
Min hold time: 10 minutes
------------------------------------------------------------
TRADE SIMULATION RESULTS FOR AAPL
============================================================
Detection Time: 2024-01-16 09:30:00
Initial Capital: $2,500.00
Final Capital: $2,647.50
Total P&L: $147.50 (5.90%)
Total Trades: 3
Win Rate: 66.67%
Winning Trades: 2
Losing Trades: 1
Max Drawdown: 2.1%
Sharpe Ratio: 1.24

TRADE DETAILS:
--------------------------------------------------------------------------------
#   Strategy    Entry    Exit     P&L        P&L%    Hold     Reason       
--------------------------------------------------------------------------------
1   surge       $178.45  $182.30  $38.40     8.6%    15       take_profit  
2   breakout    $182.50  $179.20  -$16.50    -3.6%   8        stop_loss    
3   rsi_bounce  $179.80  $185.20  $30.20     6.8%    22       take_profit  
```

## Error Handling

The simulator includes comprehensive error handling:

- Invalid parameter validation
- Webull API connection issues
- Data availability checks
- Calculation error prevention

## Troubleshooting

### Common Issues

1. **Webull API Error**: Ensure the existing Webull API modules (`webull_data_api.py` and `WebullUtil.py`) are in the `src/data/` directory
2. **No Data Available**: Check if the ticker is valid and detection time is within market hours
3. **Invalid Time Format**: Use 'YYYY-MM-DD HH:MM:SS' format for detection time
4. **Permission Errors**: Ensure write permissions for the data folder
5. **Import Errors**: Make sure all required dependencies are installed and the src directory is in Python path

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
py run_simulator.py --ticker AAPL --detection-time "2024-01-16 09:30:00" --verbose
```

## Integration with Trading Bot

The simulator can be used to:

- Test new trading strategies before deployment
- Optimize parameters for specific tickers
- Backtest historical performance
- Validate risk management rules
- Compare different strategy combinations

## File Structure

```
src/simulation/
├── trade_simulator.py      # Main simulator class
├── __init__.py            # Package initialization
simulation_data/           # Data storage folder
├── TICKER_YYYYMMDD.csv   # Historical data files
├── simulation_results_*.json  # Results files
run_simulator.py          # Command line interface
README_SIMULATOR.md       # This documentation
```

## Contributing

To extend the simulator:

1. Add new entry/exit strategies in `detect_entry_signals()` and `detect_exit_signals()`
2. Modify risk management parameters in `SimulationConfig`
3. Add new performance metrics in `calculate_results()`
4. Extend reporting functionality in `print_results()`

## License

This simulator is part of the autonomous trading bot project and follows the same licensing terms.
