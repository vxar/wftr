"""
Daily Trade Analysis System
Automatically analyzes daily trading performance at 8pm and generates detailed reports
"""
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pytz
from dataclasses import dataclass, asdict

from ..database.trading_database import TradingDatabase

logger = logging.getLogger(__name__)

@dataclass
class DailyAnalysisReport:
    """Daily trade analysis report structure"""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_hold_time: float
    best_pattern: str
    worst_pattern: str
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trade_frequency: Dict[str, int]
    hourly_performance: Dict[str, float]
    pattern_performance: Dict[str, Dict]
    rejection_analysis: Dict
    recommendations: List[str]

class DailyTradeAnalyzer:
    """Analyzes daily trading performance and generates comprehensive reports"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db = TradingDatabase(db_path)
        self.et_timezone = pytz.timezone('America/New_York')
        self.reports_path = Path("data/daily_reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Schedule daily analysis at 8pm ET
        self._schedule_analysis()
    
    def _schedule_analysis(self):
        """Schedule daily analysis at 8pm ET"""
        try:
            # Schedule for 8pm ET every day
            schedule.every().day.at("20:00").do(self.run_daily_analysis)
            logger.info("Daily trade analysis scheduled for 8:00 PM ET")
            
            # Start scheduler in background
            import threading
            def run_scheduler():
                while True:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
            
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
        except Exception as e:
            logger.error(f"Error scheduling daily analysis: {e}")
    
    def run_daily_analysis(self, target_date: Optional[str] = None) -> DailyAnalysisReport:
        """
        Run comprehensive daily trade analysis
        
        Args:
            target_date: Date to analyze (YYYY-MM-DD), defaults to yesterday
            
        Returns:
            DailyAnalysisReport with comprehensive metrics
        """
        try:
            # Determine analysis date
            if target_date:
                analysis_date = target_date
            else:
                # Default to yesterday (since we run at 8pm for current day)
                yesterday = datetime.now(self.et_timezone) - timedelta(days=1)
                analysis_date = yesterday.strftime('%Y-%m-%d')
            
            logger.info(f"Running daily trade analysis for {analysis_date}")
            
            # Get all trades for the day
            trades = self.db.get_all_trades()
            daily_trades = [
                trade for trade in trades 
                if trade.get('exit_time', '').startswith(analysis_date)
            ]
            
            if not daily_trades:
                logger.warning(f"No trades found for {analysis_date}")
                return self._create_empty_report(analysis_date)
            
            # Get rejected entries for the day
            rejected_entries = self.db.get_rejected_entries(date=analysis_date)
            
            # Generate comprehensive analysis
            report = self._generate_analysis_report(analysis_date, daily_trades, rejected_entries)
            
            # Save report
            self._save_report(report)
            
            # Log summary
            self._log_summary(report)
            
            logger.info(f"Daily analysis completed for {analysis_date}")
            return report
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")
            return self._create_empty_report(target_date or datetime.now().strftime('%Y-%m-%d'))
    
    def _generate_analysis_report(self, date: str, trades: List[Dict], rejected_entries: List[Dict]) -> DailyAnalysisReport:
        """Generate comprehensive analysis report"""
        try:
            # Basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('pnl_dollars', 0) > 0)
            losing_trades = sum(1 for t in trades if t.get('pnl_dollars', 0) < 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = sum(t.get('pnl_dollars', 0) for t in trades)
            total_pnl_pct = sum(t.get('pnl_pct', 0) for t in trades) / total_trades if total_trades > 0 else 0
            
            wins = [t.get('pnl_dollars', 0) for t in trades if t.get('pnl_dollars', 0) > 0]
            losses = [t.get('pnl_dollars', 0) for t in trades if t.get('pnl_dollars', 0) < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Hold time analysis
            hold_times = []
            for trade in trades:
                try:
                    entry_time = datetime.fromisoformat(trade.get('entry_time', '').replace('Z', '+00:00'))
                    exit_time = datetime.fromisoformat(trade.get('exit_time', '').replace('Z', '+00:00'))
                    hold_minutes = (exit_time - entry_time).total_seconds() / 60
                    hold_times.append(hold_minutes)
                except:
                    continue
            avg_hold_time = np.mean(hold_times) if hold_times else 0
            
            # Pattern performance
            pattern_performance = self._analyze_pattern_performance(trades)
            best_pattern = max(pattern_performance.items(), key=lambda x: x[1].get('total_pnl', float('-inf')))[0] if pattern_performance else "N/A"
            worst_pattern = min(pattern_performance.items(), key=lambda x: x[1].get('total_pnl', float('inf')))[0] if pattern_performance else "N/A"
            
            # Risk metrics
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf') if wins else 0
            sharpe_ratio = self._calculate_sharpe_ratio(trades)
            max_drawdown = self._calculate_max_drawdown(trades)
            
            # Trade frequency by hour
            hourly_performance = self._analyze_hourly_performance(trades)
            
            # Rejection analysis
            rejection_analysis = self._analyze_rejections(rejected_entries)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(trades, rejected_entries, pattern_performance)
            
            return DailyAnalysisReport(
                date=date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_hold_time=avg_hold_time,
                best_pattern=best_pattern,
                worst_pattern=worst_pattern,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trade_frequency=self._get_trade_frequency(trades),
                hourly_performance=hourly_performance,
                pattern_performance=pattern_performance,
                rejection_analysis=rejection_analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return self._create_empty_report(date)
    
    def _analyze_pattern_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by entry pattern"""
        pattern_stats = {}
        
        for trade in trades:
            pattern = trade.get('entry_pattern', 'Unknown')
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'pnl_list': []
                }
            
            pattern_stats[pattern]['trades'] += 1
            pnl = trade.get('pnl_dollars', 0)
            pattern_stats[pattern]['total_pnl'] += pnl
            pattern_stats[pattern]['pnl_list'].append(pnl)
            
            if pnl > 0:
                pattern_stats[pattern]['wins'] += 1
        
        # Calculate additional metrics
        for pattern, stats in pattern_stats.items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
                stats['std_dev'] = np.std(stats['pnl_list']) if len(stats['pnl_list']) > 1 else 0
        
        return pattern_stats
    
    def _analyze_hourly_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Analyze performance by hour of day"""
        hourly_pnl = {}
        
        for trade in trades:
            try:
                exit_time = datetime.fromisoformat(trade.get('exit_time', '').replace('Z', '+00:00'))
                hour = exit_time.strftime('%H:00')
                
                if hour not in hourly_pnl:
                    hourly_pnl[hour] = 0
                
                hourly_pnl[hour] += trade.get('pnl_dollars', 0)
            except:
                continue
        
        return hourly_pnl
    
    def _analyze_rejections(self, rejected_entries: List[Dict]) -> Dict:
        """Analyze rejected entries"""
        if not rejected_entries:
            return {'total_rejections': 0, 'reasons': {}, 'missed_opportunities': 0}
        
        # Group by rejection reason
        reasons = {}
        for entry in rejected_entries:
            reason = entry.get('reason', 'Unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            'total_rejections': len(rejected_entries),
            'reasons': reasons,
            'most_common_reason': max(reasons.items(), key=lambda x: x[1])[0] if reasons else 'N/A'
        }
    
    def _calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for daily returns"""
        if len(trades) < 2:
            return 0
        
        returns = [trade.get('pnl_pct', 0) / 100 for trade in trades]
        
        if not returns or np.std(returns) == 0:
            return 0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', ''))
        
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in sorted_trades:
            cumulative_pnl += trade.get('pnl_dollars', 0)
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _get_trade_frequency(self, trades: List[Dict]) -> Dict[str, int]:
        """Get trade frequency by different time periods"""
        frequency = {
            'morning': 0,    # 9:30 - 12:00
            'afternoon': 0,  # 12:00 - 16:00
            'late': 0        # 16:00 - 20:00
        }
        
        for trade in trades:
            try:
                exit_time = datetime.fromisoformat(trade.get('exit_time', '').replace('Z', '+00:00'))
                hour = exit_time.hour
                
                if 9 <= hour < 12:
                    frequency['morning'] += 1
                elif 12 <= hour < 16:
                    frequency['afternoon'] += 1
                elif 16 <= hour < 20:
                    frequency['late'] += 1
            except:
                continue
        
        return frequency
    
    def _generate_recommendations(self, trades: List[Dict], rejected_entries: List[Dict], pattern_performance: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Win rate analysis
        if len(trades) > 0:
            win_rate = sum(1 for t in trades if t.get('pnl_dollars', 0) > 0) / len(trades) * 100
            
            if win_rate < 40:
                recommendations.append("Low win rate detected. Consider tightening entry criteria or improving pattern recognition.")
            
            if win_rate > 70:
                recommendations.append("Excellent win rate! Consider increasing position sizes to maximize returns.")
        
        # Pattern analysis
        if pattern_performance:
            worst_patterns = [(p, stats) for p, stats in pattern_performance.items() if stats.get('win_rate', 0) < 30]
            if worst_patterns:
                worst_pattern_name = max(worst_patterns, key=lambda x: x[1].get('trades', 0))[0]
                recommendations.append(f"Pattern '{worst_pattern_name}' shows poor performance. Consider disabling or refining it.")
        
        # Rejection analysis
        if len(rejected_entries) > len(trades) * 2:
            recommendations.append("High rejection rate detected. Entry criteria may be too strict.")
        
        # Hold time analysis
        hold_times = []
        for trade in trades:
            try:
                entry_time = datetime.fromisoformat(trade.get('entry_time', '').replace('Z', '+00:00'))
                exit_time = datetime.fromisoformat(trade.get('exit_time', '').replace('Z', '+00:00'))
                hold_times.append((exit_time - entry_time).total_seconds() / 60)
            except:
                continue
        
        if hold_times:
            avg_hold = np.mean(hold_times)
            if avg_hold < 5:
                recommendations.append("Very short average hold time. Consider longer holding periods or scalping strategy optimization.")
            elif avg_hold > 120:
                recommendations.append("Long hold times detected. Consider more active profit taking or stop loss adjustment.")
        
        # Profit factor analysis
        wins = [t.get('pnl_dollars', 0) for t in trades if t.get('pnl_dollars', 0) > 0]
        losses = [t.get('pnl_dollars', 0) for t in trades if t.get('pnl_dollars', 0) < 0]
        
        if wins and losses:
            profit_factor = abs(sum(wins) / sum(losses))
            if profit_factor < 1.0:
                recommendations.append("Profit factor below 1.0. Average losses exceed average wins. Review stop loss strategy.")
            elif profit_factor > 3.0:
                recommendations.append("Excellent profit factor. Strategy is capturing wins effectively.")
        
        return recommendations
    
    def _save_report(self, report: DailyAnalysisReport):
        """Save report to file"""
        try:
            # Save as JSON
            report_file = self.reports_path / f"daily_report_{report.date}.json"
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            # Save latest report
            latest_file = self.reports_path / "latest_daily_report.json"
            with open(latest_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            logger.info(f"Daily report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def _log_summary(self, report: DailyAnalysisReport):
        """Log analysis summary"""
        logger.info(f"{'='*60}")
        logger.info(f"DAILY TRADE ANALYSIS - {report.date}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trades: {report.total_trades}")
        logger.info(f"Win Rate: {report.win_rate:.1f}% ({report.winning_trades}/{report.total_trades})")
        logger.info(f"Total P&L: ${report.total_pnl:+,.2f}")
        logger.info(f"Average Win: ${report.avg_win:.2f}")
        logger.info(f"Average Loss: ${report.avg_loss:.2f}")
        logger.info(f"Profit Factor: {report.profit_factor:.2f}")
        logger.info(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: ${report.max_drawdown:.2f}")
        logger.info(f"Best Pattern: {report.best_pattern}")
        logger.info(f"Worst Pattern: {report.worst_pattern}")
        
        if report.recommendations:
            logger.info(f"Recommendations: {len(report.recommendations)}")
            for i, rec in enumerate(report.recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info(f"{'='*60}")
    
    def _create_empty_report(self, date: str) -> DailyAnalysisReport:
        """Create empty report for days with no trades"""
        return DailyAnalysisReport(
            date=date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_pnl_pct=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            avg_hold_time=0,
            best_pattern="N/A",
            worst_pattern="N/A",
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown=0,
            trade_frequency={},
            hourly_performance={},
            pattern_performance={},
            rejection_analysis={},
            recommendations=["No trades executed today. Consider market conditions or strategy adjustments."]
        )
    
    def get_latest_report(self) -> Optional[DailyAnalysisReport]:
        """Get the latest daily analysis report"""
        try:
            latest_file = self.reports_path / "latest_daily_report.json"
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                return DailyAnalysisReport(**data)
            return None
        except Exception as e:
            logger.error(f"Error loading latest report: {e}")
            return None
    
    def get_report_history(self, days: int = 30) -> List[DailyAnalysisReport]:
        """Get daily analysis reports for the last N days"""
        try:
            reports = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for file_path in self.reports_path.glob("daily_report_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    report_date = datetime.strptime(data.get('date', '2024-01-01'), '%Y-%m-%d')
                    if report_date >= cutoff_date:
                        reports.append(DailyAnalysisReport(**data))
                except:
                    continue
            
            return sorted(reports, key=lambda x: x.date, reverse=True)
        except Exception as e:
            logger.error(f"Error getting report history: {e}")
            return []
