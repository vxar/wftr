"""
Comprehensive Backtesting Framework
Advanced backtesting system with realistic market simulation and performance analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import pytz
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Backtesting modes"""
    HISTORICAL = "historical"  # Use historical data
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    WALK_FORWARD = "walk_forward"  # Walk-forward analysis
    STRESS_TEST = "stress_test"  # Stress testing scenarios

class CommissionModel(Enum):
    """Commission models"""
    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    TIERED = "tiered"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_model: CommissionModel
    commission_rate: float
    slippage_bps: int  # Basis points
    position_size_method: str  # 'fixed', 'percentage', 'volatility_based'
    max_positions: int
    risk_per_trade: float
    stop_loss_method: str  # 'fixed', 'atr', 'volatility'
    take_profit_method: str  # 'fixed', 'atr', 'volatility'
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly'
    market_impact_model: bool  # Include market impact
    short_selling: bool  # Allow short selling
    leverage: float  # Account leverage

@dataclass
class TradeRecord:
    """Individual trade record"""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    entry_value: float
    exit_value: float
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    position_type: str
    entry_signal: str
    exit_signal: str
    hold_time: timedelta
    max_drawdown: float
    max_runup: float

@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: timedelta
    trades: List[TradeRecord]
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    performance_metrics: Dict

class ComprehensiveBacktester:
    """
    Advanced backtesting framework with realistic market simulation
    """
    
    def __init__(self, data_api=None):
        """
        Args:
            data_api: Data API for fetching historical data
        """
        self.data_api = data_api
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Backtesting state
        self.current_capital = 0.0
        self.current_positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Performance tracking
        self.daily_returns = []
        self.drawdowns = []
        self.running_capital = []
        
        # Market data cache
        self.market_data_cache = {}
        
    def run_backtest(self, 
                    config: BacktestConfig,
                    strategy_function: Callable,
                    data_sources: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            config: Backtesting configuration
            strategy_function: Strategy function that returns trading signals
            data_sources: Dictionary of historical data for all tickers
            
        Returns:
            Complete backtest results
        """
        try:
            logger.info(f"Starting backtest: {config.start_date.date()} to {config.end_date.date()}")
            
            # Initialize backtest
            self._initialize_backtest(config)
            
            # Generate trading dates
            trading_dates = self._generate_trading_dates(config.start_date, config.end_date)
            
            # Run backtest day by day
            for date in trading_dates:
                self._process_trading_day(date, strategy_function, data_sources, config)
            
            # Finalize backtest
            result = self._finalize_backtest(config)
            
            logger.info(f"Backtest completed: Total return {result.total_return:.2%}, Sharpe ratio {result.sharpe_ratio:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _initialize_backtest(self, config: BacktestConfig):
        """Initialize backtest state"""
        self.current_capital = config.initial_capital
        self.current_positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.drawdowns = []
        self.running_capital = []
        
        # Record initial state
        self.equity_curve.append({
            'date': config.start_date,
            'capital': self.current_capital,
            'positions_value': 0.0,
            'total_value': self.current_capital,
            'cash': self.current_capital
        })
    
    def _generate_trading_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate list of trading dates"""
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday-Friday
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        return dates
    
    def _process_trading_day(self, 
                           date: datetime,
                           strategy_function: Callable,
                           data_sources: Dict[str, pd.DataFrame],
                           config: BacktestConfig):
        """Process a single trading day"""
        try:
            # Get market data for this date
            daily_data = self._get_daily_data(date, data_sources)
            
            if not daily_data:
                return
            
            # Update existing positions
            self._update_positions(date, daily_data, config)
            
            # Generate trading signals
            signals = strategy_function(date, daily_data, self.current_positions, config)
            
            # Execute trades
            self._execute_signals(date, signals, daily_data, config)
            
            # Calculate daily performance
            self._calculate_daily_performance(date, config)
            
        except Exception as e:
            logger.error(f"Error processing trading day {date.date()}: {e}")
    
    def _get_daily_data(self, date: datetime, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Get market data for a specific date"""
        daily_data = {}
        
        for ticker, df in data_sources.items():
            # Filter data for this date
            date_data = df[df['timestamp'].dt.date == date.date()]
            
            if not date_data.empty:
                # Get the last data point of the day
                last_bar = date_data.iloc[-1]
                
                daily_data[ticker] = {
                    'open': last_bar['open'],
                    'high': last_bar['high'],
                    'low': last_bar['low'],
                    'close': last_bar['close'],
                    'volume': last_bar['volume'],
                    'timestamp': last_bar['timestamp']
                }
        
        return daily_data
    
    def _update_positions(self, date: datetime, daily_data: Dict[str, Dict], config: BacktestConfig):
        """Update existing positions with current prices"""
        positions_to_close = []
        
        for ticker, position in self.current_positions.items():
            if ticker in daily_data:
                current_price = daily_data[ticker]['close']
                
                # Update position value
                position['current_price'] = current_price
                position['current_value'] = position['shares'] * current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
                position['unrealized_pnl_pct'] = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                # Check exit conditions
                exit_signal = self._check_exit_conditions(position, daily_data[ticker], config)
                
                if exit_signal:
                    positions_to_close.append((ticker, exit_signal))
        
        # Close positions with exit signals
        for ticker, exit_signal in positions_to_close:
            self._close_position(ticker, date, daily_data[ticker]['close'], exit_signal, config)
    
    def _check_exit_conditions(self, position: Dict, market_data: Dict, config: BacktestConfig) -> Optional[str]:
        """Check if position should be closed"""
        try:
            current_price = market_data['close']
            entry_price = position['entry_price']
            unrealized_pct = (current_price - entry_price) / entry_price * 100
            
            # Stop loss
            if config.stop_loss_method == 'fixed':
                stop_loss_pct = 3.0  # Default 3% stop loss
            elif config.stop_loss_method == 'atr':
                # Would need ATR calculation
                stop_loss_pct = 2.5
            else:
                stop_loss_pct = 2.0
            
            if unrealized_pct <= -stop_loss_pct:
                return 'stop_loss'
            
            # Take profit
            if config.take_profit_method == 'fixed':
                take_profit_pct = 8.0  # Default 8% target
            elif config.take_profit_method == 'atr':
                take_profit_pct = 6.0
            else:
                take_profit_pct = 10.0
            
            if unrealized_pct >= take_profit_pct:
                return 'take_profit'
            
            # Time-based exit (simplified)
            entry_time = position['entry_time']
            current_time = datetime.now(self.et_timezone)
            hold_days = (current_time - entry_time).days
            
            if hold_days > 5:  # Exit after 5 days
                return 'time_exit'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def _execute_signals(self, 
                        date: datetime,
                        signals: List[Dict],
                        daily_data: Dict[str, Dict],
                        config: BacktestConfig):
        """Execute trading signals"""
        for signal in signals:
            try:
                ticker = signal.get('ticker')
                action = signal.get('action')  # 'buy' or 'sell'
                
                if ticker not in daily_data:
                    continue
                
                current_price = daily_data[ticker]['close']
                
                if action == 'buy' and ticker not in self.current_positions:
                    if len(self.current_positions) < config.max_positions:
                        self._open_position(ticker, date, current_price, signal, config)
                
                elif action == 'sell' and ticker in self.current_positions:
                    self._close_position(ticker, date, current_price, 'sell_signal', config)
                    
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                continue
    
    def _open_position(self, 
                      ticker: str,
                      date: datetime,
                      price: float,
                      signal: Dict,
                      config: BacktestConfig):
        """Open a new position"""
        try:
            # Calculate position size
            if config.position_size_method == 'fixed':
                position_value = config.initial_capital * 0.1  # 10% per position
            elif config.position_size_method == 'percentage':
                position_value = self.current_capital * 0.05  # 5% of current capital
            elif config.position_size_method == 'volatility_based':
                # Would need volatility calculation
                position_value = self.current_capital * 0.03
            else:
                position_value = config.initial_capital * 0.05
            
            # Check available capital
            if position_value > self.current_capital:
                position_value = self.current_capital * 0.9  # Use 90% of available capital
            
            # Calculate shares
            shares = position_value / price
            
            # Apply slippage
            slippage_cost = price * config.slippage_bps / 10000
            adjusted_price = price + slippage_cost
            
            # Calculate commission
            commission = self._calculate_commission(shares, adjusted_price, config)
            
            # Total cost
            total_cost = (shares * adjusted_price) + commission
            
            if total_cost > self.current_capital:
                return  # Not enough capital
            
            # Create position
            position = {
                'ticker': ticker,
                'entry_time': date,
                'entry_price': adjusted_price,
                'shares': shares,
                'entry_value': total_cost,
                'current_price': adjusted_price,
                'current_value': total_cost,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'position_type': signal.get('position_type', 'long'),
                'entry_signal': signal.get('signal_type', 'unknown'),
                'commission': commission,
                'slippage': slippage_cost * shares
            }
            
            self.current_positions[ticker] = position
            self.current_capital -= total_cost
            
            logger.debug(f"Opened position: {ticker} - {shares:.2f} shares @ ${adjusted_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error opening position {ticker}: {e}")
    
    def _close_position(self, 
                       ticker: str,
                       date: datetime,
                       price: float,
                       exit_signal: str,
                       config: BacktestConfig):
        """Close an existing position"""
        try:
            if ticker not in self.current_positions:
                return
            
            position = self.current_positions[ticker]
            
            # Apply slippage
            slippage_cost = price * config.slippage_bps / 10000
            adjusted_price = price - slippage_cost  # Selling, so slippage reduces price
            
            # Calculate commission
            commission = self._calculate_commission(position['shares'], adjusted_price, config)
            
            # Calculate proceeds
            gross_proceeds = position['shares'] * adjusted_price
            net_proceeds = gross_proceeds - commission
            
            # Calculate P&L
            pnl = net_proceeds - position['entry_value']
            pnl_pct = pnl / position['entry_value'] * 100
            
            # Create trade record
            trade_record = TradeRecord(
                ticker=ticker,
                entry_time=position['entry_time'],
                exit_time=date,
                entry_price=position['entry_price'],
                exit_price=adjusted_price,
                shares=position['shares'],
                entry_value=position['entry_value'],
                exit_value=net_proceeds,
                commission=position['commission'] + commission,
                slippage=position['slippage'] + (slippage_cost * position['shares']),
                pnl=pnl,
                pnl_pct=pnl_pct,
                position_type=position['position_type'],
                entry_signal=position['entry_signal'],
                exit_signal=exit_signal,
                hold_time=date - position['entry_time'],
                max_drawdown=0.0,  # Would need intraday data
                max_runup=0.0  # Would need intraday data
            )
            
            self.trades.append(trade_record)
            
            # Update capital
            self.current_capital += net_proceeds
            
            # Remove position
            del self.current_positions[ticker]
            
            logger.debug(f"Closed position: {ticker} - P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error closing position {ticker}: {e}")
    
    def _calculate_commission(self, shares: float, price: float, config: BacktestConfig) -> float:
        """Calculate trading commission"""
        trade_value = shares * price
        
        if config.commission_model == CommissionModel.PER_SHARE:
            return shares * config.commission_rate
        elif config.commission_model == CommissionModel.PERCENTAGE:
            return trade_value * config.commission_rate / 100
        elif config.commission_model == CommissionModel.FIXED:
            return config.commission_rate
        elif config.commission_model == CommissionModel.TIERED:
            # Simplified tiered commission
            if trade_value < 1000:
                return trade_value * 0.005  # 0.5%
            elif trade_value < 10000:
                return trade_value * 0.003  # 0.3%
            else:
                return trade_value * 0.002  # 0.2%
        else:
            return 0.0
    
    def _calculate_daily_performance(self, date: datetime, config: BacktestConfig):
        """Calculate daily performance metrics"""
        try:
            # Calculate total portfolio value
            positions_value = sum(pos['current_value'] for pos in self.current_positions.values())
            total_value = self.current_capital + positions_value
            
            # Calculate daily return
            if len(self.equity_curve) > 0:
                prev_total_value = self.equity_curve[-1]['total_value']
                daily_return = (total_value - prev_total_value) / prev_total_value
            else:
                daily_return = 0.0
            
            # Update equity curve
            self.equity_curve.append({
                'date': date,
                'capital': self.current_capital,
                'positions_value': positions_value,
                'total_value': total_value,
                'cash': self.current_capital,
                'daily_return': daily_return
            })
            
            self.daily_returns.append(daily_return)
            self.running_capital.append(total_value)
            
            # Calculate drawdown
            peak = max(self.running_capital) if self.running_capital else total_value
            drawdown = (peak - total_value) / peak if peak > 0 else 0.0
            self.drawdowns.append(drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating daily performance: {e}")
    
    def _finalize_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Finalize backtest and calculate results"""
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # Calculate performance metrics
            final_capital = self.equity_curve[-1]['total_value']
            total_return = (final_capital - config.initial_capital) / config.initial_capital
            
            # Annualized return
            days = (config.end_date - config.start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Volatility
            returns = pd.Series(self.daily_returns)
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Max drawdown
            max_drawdown = max(self.drawdowns) if self.drawdowns else 0
            
            # Max drawdown duration (simplified)
            max_drawdown_duration = timedelta(days=30)  # Placeholder
            
            # Trade statistics
            total_trades = len(self.trades)
            winning_trades = sum(1 for t in self.trades if t.pnl > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            wins = [t.pnl for t in self.trades if t.pnl > 0]
            losses = [t.pnl for t in self.trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Average trade duration
            if self.trades:
                avg_duration = sum(t.hold_time for t in self.trades) / len(self.trades)
            else:
                avg_duration = timedelta(0)
            
            # Monthly returns
            monthly_returns = equity_df['total_value'].resample('M').last().pct_change().dropna()
            
            # Additional performance metrics
            performance_metrics = {
                'calmar_ratio': annualized_return / max_drawdown if max_drawdown > 0 else 0,
                'var_95': returns.quantile(0.05) if len(returns) > 0 else 0,
                'skewness': returns.skew() if len(returns) > 2 else 0,
                'kurtosis': returns.kurtosis() if len(returns) > 3 else 0,
                'best_trade': max([t.pnl for t in self.trades]) if self.trades else 0,
                'worst_trade': min([t.pnl for t in self.trades]) if self.trades else 0,
                'avg_trade': np.mean([t.pnl for t in self.trades]) if self.trades else 0,
                'commission_total': sum(t.commission for t in self.trades),
                'slippage_total': sum(t.slippage for t in self.trades)
            }
            
            return BacktestResult(
                config=config,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                avg_trade_duration=avg_duration,
                trades=self.trades,
                equity_curve=equity_df,
                monthly_returns=monthly_returns,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error finalizing backtest: {e}")
            raise
    
    def run_monte_carlo(self, 
                       base_result: BacktestResult,
                       num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation on backtest results"""
        try:
            logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")
            
            # Extract daily returns
            daily_returns = base_result.equity_curve['daily_return'].dropna()
            
            # Run simulations
            simulation_results = []
            
            for i in range(num_simulations):
                # Randomly sample daily returns
                simulated_returns = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
                
                # Calculate cumulative returns
                cumulative_returns = (1 + simulated_returns).cumprod()
                final_return = cumulative_returns[-1] - 1
                
                # Calculate metrics
                volatility = simulated_returns.std() * np.sqrt(252)
                sharpe = (final_return * (365 / len(daily_returns)) - 0.02) / volatility if volatility > 0 else 0
                
                # Calculate drawdown
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (peak - cumulative_returns) / peak
                max_dd = drawdown.max()
                
                simulation_results.append({
                    'total_return': final_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd
                })
            
            # Calculate statistics
            results_df = pd.DataFrame(simulation_results)
            
            monte_carlo_stats = {
                'return_mean': results_df['total_return'].mean(),
                'return_std': results_df['total_return'].std(),
                'return_5th_percentile': results_df['total_return'].quantile(0.05),
                'return_95th_percentile': results_df['total_return'].quantile(0.95),
                'sharpe_mean': results_df['sharpe_ratio'].mean(),
                'sharpe_std': results_df['sharpe_ratio'].std(),
                'max_drawdown_mean': results_df['max_drawdown'].mean(),
                'max_drawdown_worst': results_df['max_drawdown'].max(),
                'probability_of_loss': (results_df['total_return'] < 0).mean(),
                'probability_of_10pct_loss': (results_df['total_return'] < -0.10).mean()
            }
            
            logger.info("Monte Carlo simulation completed")
            return monte_carlo_stats
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def generate_report(self, result: BacktestResult, output_path: str = None) -> str:
        """Generate comprehensive backtest report"""
        try:
            report = []
            report.append("# COMPREHENSIVE BACKTEST REPORT\n")
            
            # Configuration
            report.append("## Configuration")
            report.append(f"- Period: {result.start_date.date()} to {result.end_date.date()}")
            report.append(f"- Initial Capital: ${result.initial_capital:,.2f}")
            report.append(f"- Commission Model: {result.config.commission_model.value}")
            report.append(f"- Slippage: {result.config.slippage_bps} bps")
            report.append(f"- Max Positions: {result.config.max_positions}")
            report.append("")
            
            # Performance Summary
            report.append("## Performance Summary")
            report.append(f"- Total Return: {result.total_return:.2%}")
            report.append(f"- Annualized Return: {result.annualized_return:.2%}")
            report.append(f"- Volatility: {result.volatility:.2%}")
            report.append(f"- Sharpe Ratio: {result.sharpe_ratio:.2f}")
            report.append(f"- Sortino Ratio: {result.sortino_ratio:.2f}")
            report.append(f"- Maximum Drawdown: {result.max_drawdown:.2%}")
            report.append(f"- Calmar Ratio: {result.performance_metrics.get('calmar_ratio', 0):.2f}")
            report.append("")
            
            # Trade Statistics
            report.append("## Trade Statistics")
            report.append(f"- Total Trades: {result.total_trades}")
            report.append(f"- Winning Trades: {result.winning_trades}")
            report.append(f"- Losing Trades: {result.losing_trades}")
            report.append(f"- Win Rate: {result.win_rate:.2%}")
            report.append(f"- Average Win: ${result.avg_win:,.2f}")
            report.append(f"- Average Loss: ${result.avg_loss:,.2f}")
            report.append(f"- Profit Factor: {result.profit_factor:.2f}")
            report.append(f"- Average Trade Duration: {result.avg_trade_duration}")
            report.append(f"- Best Trade: ${result.performance_metrics.get('best_trade', 0):,.2f}")
            report.append(f"- Worst Trade: ${result.performance_metrics.get('worst_trade', 0):,.2f}")
            report.append("")
            
            # Risk Metrics
            report.append("## Risk Metrics")
            report.append(f"- 95% VaR: {result.performance_metrics.get('var_95', 0):.2%}")
            report.append(f"- Skewness: {result.performance_metrics.get('skewness', 0):.3f}")
            report.append(f"- Kurtosis: {result.performance_metrics.get('kurtosis', 0):.3f}")
            report.append(f"- Total Commission: ${result.performance_metrics.get('commission_total', 0):,.2f}")
            report.append(f"- Total Slippage: ${result.performance_metrics.get('slippage_total', 0):,.2f}")
            report.append("")
            
            # Monthly Returns
            report.append("## Monthly Returns")
            for date, ret in result.monthly_returns.items():
                report.append(f"- {date.strftime('%Y-%m')}: {ret:.2%}")
            report.append("")
            
            report_text = "\n".join(report)
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Backtest report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtest Results', fontsize=16)
            
            # Equity Curve
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve['total_value'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # Drawdown
            axes[0, 1].fill_between(result.equity_curve.index, 0, -np.array(self.drawdowns) * 100, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True)
            
            # Daily Returns Distribution
            returns = pd.Series(self.daily_returns)
            axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
            
            # Monthly Returns
            monthly_returns = result.monthly_returns * 100
            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
            axes[1, 1].bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Results plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def compare_strategies(self, 
                          results: List[BacktestResult],
                          strategy_names: List[str]) -> pd.DataFrame:
        """Compare multiple backtest results"""
        try:
            comparison_data = []
            
            for i, result in enumerate(results):
                comparison_data.append({
                    'Strategy': strategy_names[i],
                    'Total Return': result.total_return,
                    'Annualized Return': result.annualized_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Sortino Ratio': result.sortino_ratio,
                    'Max Drawdown': result.max_drawdown,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Total Trades': result.total_trades
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.set_index('Strategy', inplace=True)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()
