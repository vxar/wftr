#!/usr/bin/env python3
"""
Trade Simulator Runner

This script provides a command-line interface for running the trade simulator
with different configurations and parameters.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from src.simulation.pure_trade_simulator import PureTradeSimulator as TradeSimulator, SimulationConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run trade simulator for testing trading strategies"
    )
    
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        help="Stock ticker symbol (e.g., AAPL, TSLA)"
    )
    
    parser.add_argument(
        "--detection-time", "-d",
        required=True,
        help="Detection time in format 'YYYY-MM-DD HH:MM:SS'"
    )
    
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=2500.0,
        help="Initial capital (default: 2500.0)"
    )
    
    parser.add_argument(
        "--max-positions", "-m",
        type=int,
        default=1,
        help="Maximum number of positions (default: 1)"
    )
    
    parser.add_argument(
        "--stop-loss", "-s",
        type=float,
        default=0.06,
        help="Stop loss percentage (default: 0.06 = 6%%)"
    )
    
    parser.add_argument(
        "--take-profit", "-p",
        type=float,
        default=0.08,
        help="Take profit percentage (default: 0.08 = 8%%)"
    )
    
    parser.add_argument(
        "--commission", "-x",
        type=float,
        default=0.005,
        help="Commission per trade (default: 0.005 = 0.5%%)"
    )
    
    parser.add_argument(
        "--min-hold", "-o",
        type=int,
        default=10,
        help="Minimum hold time in minutes (default: 10)"
    )
    
    parser.add_argument(
        "--data-folder", "-f",
        default="simulation_data",
        help="Data folder for storing historical data (default: simulation_data)"
    )
    
    parser.add_argument(
        "--save-results", "-r",
        action="store_true",
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    try:
        datetime.strptime(args.detection_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("Error: Detection time must be in format 'YYYY-MM-DD HH:MM:SS'")
        return False
    
    if args.capital <= 0:
        print("Error: Initial capital must be positive")
        return False
    
    if args.max_positions < 1:
        print("Error: Maximum positions must be at least 1")
        return False
    
    if not (0 < args.stop_loss < 1):
        print("Error: Stop loss must be between 0 and 1")
        return False
    
    if not (0 < args.take_profit < 1):
        print("Error: Take profit must be between 0 and 1")
        return False
    
    if not (0 <= args.commission < 1):
        print("Error: Commission must be between 0 and 1")
        return False
    
    if args.min_hold < 0:
        print("Error: Minimum hold time must be non-negative")
        return False
    
    return True


def main():
    """Main function"""
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Create configuration
    config = SimulationConfig(
        ticker=args.ticker.upper(),
        detection_time=args.detection_time,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        commission_per_trade=args.commission,
        data_folder=args.data_folder,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_hold_minutes=args.min_hold
    )
    
    print(f"Starting trade simulator for {config.ticker}")
    print(f"Detection time: {config.detection_time}")
    print(f"Initial capital: ${config.initial_capital:,.2f}")
    print(f"Stop loss: {config.stop_loss_pct:.1%}")
    print(f"Take profit: {config.take_profit_pct:.1%}")
    print(f"Commission: {config.commission_per_trade:.1%}")
    print(f"Min hold time: {config.min_hold_minutes} minutes")
    print("-" * 60)
    
    # Create and run simulator
    simulator = TradeSimulator(config)
    
    try:
        result = simulator.run_simulation()
        simulator.print_results(result)
        
        if args.save_results:
            simulator.save_results(result)
        
        # Exit with appropriate code based on results
        if result.total_pnl > 0:
            print(f"\n✅ Simulation completed with profit: ${result.total_pnl:,.2f}")
            sys.exit(0)
        elif result.total_pnl < 0:
            print(f"\n❌ Simulation completed with loss: ${result.total_pnl:,.2f}")
            sys.exit(1)
        else:
            print(f"\n⚪ Simulation completed with no trades")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
