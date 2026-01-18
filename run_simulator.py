#!/usr/bin/env python3
"""
Flexible Pure Trade Simulator - Test multiple stocks
"""

from src.simulation.pure_trade_simulator import PureTradeSimulator, SimulationConfig
import logging
import sys

def run_simulation(ticker, detection_time, initial_capital=2500.0, **kwargs):
    """Run simulation for a single stock"""
    # Default configuration
    config_params = {
        'ticker': ticker,
        'detection_time': detection_time,
        'initial_capital': initial_capital,
        'max_positions': 1,
        'commission_per_trade': 0.005,  # 0.5% commission
        'data_folder': 'simulation_data',
        'stop_loss_pct': 0.06,  # 6% stop loss
        'take_profit_pct': 0.08,  # 8% take profit
        'min_hold_minutes': 10  # Minimum hold time
    }
    
    # Override with any provided kwargs
    config_params.update(kwargs)
    
    config = SimulationConfig(**config_params)
    
    print(f"\n🚀 Running Simulation for {ticker}")
    print(f"📊 Detection Time: {detection_time}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print("-" * 50)
    
    try:
        simulator = PureTradeSimulator(config)
        result = simulator.run_simulation()
        
        # Display results
        print(f"\n📈 {ticker} RESULTS")
        print("=" * 30)
        print(f"Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"P&L: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2%})")
        print(f"Max DD: {result.max_drawdown:.2%}")
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        
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
    parser.add_argument('--capital', type=float, default=2500.0, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=1, help='Max positions')
    parser.add_argument('--commission', type=float, default=0.005, help='Commission per trade (0.005 = 0.5 percent)')
    parser.add_argument('--stop-loss', type=float, default=0.06, help='Stop loss percentage (0.06 = 6 percent)')
    parser.add_argument('--take-profit', type=float, default=0.08, help='Take profit percentage (0.08 = 8 percent)')
    parser.add_argument('--min-hold', type=int, default=10, help='Minimum hold minutes')
    
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
                'initial_capital': args.capital,
                'max_positions': args.max_positions,
                'commission_per_trade': args.commission,
                'stop_loss_pct': args.stop_loss,
                'take_profit_pct': args.take_profit,
                'min_hold_minutes': args.min_hold
            }
            stocks = [(args.ticker.upper(), args.detection_time or '2026-01-16 19:30:00', config_params)]
        else:
            stocks = default_stocks
    
    print("🎯 PURE TRADE SIMULATOR")
    print("=" * 50)
    print("Architecture: Pure wrapper - ALL logic in position manager")
    print("No trading logic in simulator - only data feeding")
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
        
        print("\n🏆 Best Performers:")
        results.sort(key=lambda x: x[1].total_pnl, reverse=True)
        for ticker, result in results[:3]:
            print(f"  {ticker}: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2%})")
    
    print("\n✅ All simulations completed!")

if __name__ == "__main__":
    main()
