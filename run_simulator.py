#!/usr/bin/env python3
"""
Flexible Pure Trade Simulator - Test multiple stocks
"""

from src.simulation.pure_trade_simulator import PureTradeSimulator, SimulationConfig
from src.config.settings import settings
import logging
import sys
import pandas as pd

def run_simulation(ticker, detection_time, initial_capital=None, **kwargs):
    """Run simulation for a single stock"""
    # Use centralized settings if not provided
    if initial_capital is None:
        initial_capital = settings.capital.initial_capital
    
    # Enhanced configuration using same logic as realtime bot
    config_params = {
        'ticker': ticker,
        'detection_time': detection_time,
        'initial_capital': initial_capital,
        'max_positions': settings.trading.max_positions,
        'commission_per_trade': 0.005,  # 0.5% commission
        'data_folder': 'simulation_data',
        # Note: stop_loss_pct and take_profit_pct are now handled by IntelligentPositionManager
        # The position manager uses enhanced SURGE/SWING/BREAKOUT configurations
        # These parameters are kept for backward compatibility but not used in core logic
        'stop_loss_pct': 0.12,  # 12% (matches SURGE config)
        'take_profit_pct': 0.25,  # 25% (matches SURGE final target)
        'min_hold_minutes': 5  # Reduced to allow faster exits for strong movers
    }
    
    # Override with any provided kwargs
    config_params.update(kwargs)
    
    config = SimulationConfig(**config_params)
    
    print(f"\nRunning Simulation for {ticker}")
    print(f"Detection Time: {detection_time}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("-" * 50)
    
    try:
        simulator = PureTradeSimulator(config)
        result = simulator.run_simulation()
        
        # Display results
        print(f"\n{ticker} RESULTS")
        print("=" * 30)
        print(f"Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"P&L: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2%})")
        print(f"Max DD: {result.max_drawdown:.2%}")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        
        # Display entries and exits table
        if not simulator.entries_exits_df.empty:
            print(f"\n{ticker} ENTRIES & EXITS")
            print("=" * 80)
            # Format the dataframe for better display
            display_df = simulator.entries_exits_df.copy()
            # Format price column
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
            # Set column widths for better formatting
            pd.set_option('display.max_colwidth', 20)
            print(display_df.to_string(index=False))
            pd.reset_option('display.max_colwidth')
        else:
            print(f"\n{ticker} ENTRIES & EXITS")
            print("=" * 30)
            print("No entries or exits recorded.")
        
        return result
        
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")
        return None

def main():
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    
    # Default stocks to test
    default_stocks = [
        ('IVF', '2026-01-16 19:30:00'),
        ('AAPL', '2026-01-16 19:30:00'),
        ('ICON', '2026-01-16 19:30:00')
    ]
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Pure Trade Simulator - Flexible Testing')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--detection-time', type=str, help='Detection time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--capital', type=float, default=None, help='Initial capital (uses settings if not provided)')
    parser.add_argument('--max-positions', type=int, default=None, help='Max positions (uses settings if not provided)')
    parser.add_argument('--commission', type=float, default=0.005, help='Commission per trade (0.005 = 0.5 percent)')
    parser.add_argument('--stop-loss', type=float, default=0.12, help='Stop loss percentage (0.12 = 12 percent, matches SURGE config)')
    parser.add_argument('--take-profit', type=float, default=0.25, help='Take profit percentage (0.25 = 25 percent, matches SURGE final target)')
    parser.add_argument('--min-hold', type=int, default=5, help='Minimum hold minutes (reduced for strong movers)')
    
    args = parser.parse_args()
    
    # Check if help or no arguments
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python run_simulator_flexible.py                    # Test default stocks")
        print("  python run_simulator_flexible.py --ticker IVF                # Test specific stock")
        print("  python run_simulator_flexible.py --ticker IVF --detection-time '2024-01-16 16:12:00'     # Test stock with custom date")
        print("  python run_simulator_flexible.py --help                # Show all options")
        stocks = default_stocks
    else:
        # Custom stock with provided arguments
        if args.ticker:
            config_params = {
                'detection_time': args.detection_time or '2026-01-16 19:30:00',
                'initial_capital': args.capital or settings.capital.initial_capital,
                'max_positions': args.max_positions or settings.trading.max_positions,
                'commission_per_trade': args.commission,
                'stop_loss_pct': args.stop_loss,
                'take_profit_pct': args.take_profit,
                'min_hold_minutes': args.min_hold
            }
            stocks = [(args.ticker.upper(), args.detection_time or '2026-01-16 19:30:00', config_params)]
        else:
            stocks = default_stocks
    
    print("ENHANCED PURE TRADE SIMULATOR")
    print("=" * 50)
    print("Architecture: Pure wrapper - ALL logic in IntelligentPositionManager")
    print("Enhanced Features:")
    print("  - Dynamic surge detection with recovery checks")
    print("  - Position-specific exit logic (SURGE/SWING/BREAKOUT)")
    print("  - Momentum-based profit target adjustments")
    print("  - Exit delays for strong movers")
    print("=" * 50)
    
    # Run simulations
    results = []
    for stock_data in stocks:
        if len(stock_data) == 3:  # Custom config with params
            ticker, detection_time, config_params = stock_data
            result = run_simulation(ticker, **config_params)
        else:  # Default config
            ticker, detection_time = stock_data
            result = run_simulation(ticker, detection_time)
        
        if result:
            results.append((ticker, result))
    
    # Summary
    if len(results) > 1:
        print("\n📊 SUMMARY")
        print("=" * 50)
        total_trades = sum(r.total_trades for _, r in results)
        total_pnl = sum(r.total_pnl for _, r in results)
        winning_stocks = sum(1 for _, r in results if r.total_pnl > 0)
        
        print(f"Total Stocks: {len(results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Stocks: {winning_stocks}/{len(results)}")
        print(f"Total P&L: ${total_pnl:.2f}")
        
        print("\nBest Performers:")
        results.sort(key=lambda x: x[1].total_pnl, reverse=True)
        for ticker, result in results[:3]:
            print(f"  {ticker}: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2%})")
    
    print("\nAll simulations completed!")

if __name__ == "__main__":
    main()
