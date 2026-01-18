"""
Rejected Trade Analyzer and Re-analysis System
Comprehensive system for logging, analyzing, and re-evaluating rejected trades
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import pytz
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class RejectionReason(Enum):
    """Reasons for trade rejection"""
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    LOW_VOLUME = "low_volume"
    HIGH_VOLATILITY = "high_volatility"
    MANIPULATION_DETECTED = "manipulation_detected"
    MARKET_CONDITIONS = "market_conditions"
    POSITION_LIMIT = "position_limit"
    CAPITAL_CONSTRAINTS = "capital_constraints"
    TIME_RESTRICTION = "time_restriction"
    PATTERN_FAILURE = "pattern_failure"
    MULTI_TIMEFRAME_MISMATCH = "multi_timeframe_mismatch"
    RISK_THRESHOLD = "risk_threshold"
    BLACKLISTED = "blacklisted"
    TECHNICAL_FAILURE = "technical_failure"

class ReanalysisStatus(Enum):
    """Status of reanalysis"""
    PENDING = "pending"
    IMPROVED = "improved"
    STILL_REJECTED = "still_rejected"
    CONVERTED_TO_TRADE = "converted_to_trade"
    EXPIRED = "expired"

@dataclass
class RejectedTrade:
    """Rejected trade record"""
    ticker: str
    timestamp: datetime
    rejection_reason: RejectionReason
    rejection_details: str
    
    # Original signal data
    entry_price: float
    signal_strength: float
    pattern_name: str
    volume_ratio: float
    volatility_score: float
    multi_timeframe_confidence: float
    market_condition: str
    
    # Thresholds that caused rejection
    confidence_threshold: float
    volume_threshold: float
    volatility_threshold: float
    
    # What happened after rejection
    subsequent_price_action: Optional[Dict] = None
    would_have_been_profitable: Optional[bool] = None
    missed_profit_pct: Optional[float] = None
    
    # Reanalysis data
    reanalysis_status: ReanalysisStatus = ReanalysisStatus.PENDING
    reanalysis_timestamp: Optional[datetime] = None
    reanalysis_notes: str = ""
    follow_up_count: int = 0

@dataclass
class ReanalysisResult:
    """Result of reanalysis"""
    original_rejection: RejectedTrade
    current_conditions: Dict
    new_signal_strength: float
    recommendation: str  # 'enter_now', 'continue_monitoring', 'permanent_reject'
    confidence_improvement: float
    reasons_for_change: List[str]
    updated_thresholds: Optional[Dict] = None

class RejectedTradeAnalyzer:
    """
    Comprehensive system for analyzing and re-evaluating rejected trades
    """
    
    def __init__(self, 
                 reanalysis_interval_minutes: int = 15,
                 max_follow_up_attempts: int = 3,
                 opportunity_window_hours: int = 4):
        """
        Args:
            reanalysis_interval_minutes: How often to recheck rejected trades
            max_follow_up_attempts: Maximum number of times to recheck a trade
            opportunity_window_hours: How long to keep tracking rejected trades
        """
        self.reanalysis_interval_minutes = reanalysis_interval_minutes
        self.max_follow_up_attempts = max_follow_up_attempts
        self.opportunity_window_hours = opportunity_window_hours
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Storage
        self.rejected_trades: List[RejectedTrade] = []
        self.reanalysis_history: List[ReanalysisResult] = []
        
        # Analysis metrics
        self.rejection_statistics = defaultdict(int)
        self.improvement_statistics = defaultdict(int)
        
        # Learning data
        self.threshold_performance = defaultdict(list)
        self.pattern_rejection_rates = defaultdict(lambda: defaultdict(int))
        
        # File storage
        self.data_path = Path("data/rejected_trades")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_rejected_trades()
    
    def log_rejected_trade(self, 
                          ticker: str,
                          rejection_reason: RejectionReason,
                          rejection_details: str,
                          signal_data: Dict,
                          thresholds: Dict) -> bool:
        """
        Log a rejected trade for future analysis
        
        Args:
            ticker: Stock ticker
            rejection_reason: Primary reason for rejection
            rejection_details: Detailed explanation
            signal_data: Original signal data
            thresholds: Thresholds that caused rejection
            
        Returns:
            True if successfully logged
        """
        try:
            rejected_trade = RejectedTrade(
                ticker=ticker,
                timestamp=datetime.now(self.et_timezone),
                rejection_reason=rejection_reason,
                rejection_details=rejection_details,
                entry_price=signal_data.get('entry_price', 0),
                signal_strength=signal_data.get('signal_strength', 0),
                pattern_name=signal_data.get('pattern_name', ''),
                volume_ratio=signal_data.get('volume_ratio', 1),
                volatility_score=signal_data.get('volatility_score', 0),
                multi_timeframe_confidence=signal_data.get('multi_timeframe_confidence', 0),
                market_condition=signal_data.get('market_condition', 'normal'),
                confidence_threshold=thresholds.get('confidence', 0.7),
                volume_threshold=thresholds.get('volume', 1.5),
                volatility_threshold=thresholds.get('volatility', 0.7)
            )
            
            self.rejected_trades.append(rejected_trade)
            self.rejection_statistics[rejection_reason.value] += 1
            
            # Update pattern rejection rates
            self.pattern_rejection_rates[rejected_trade.pattern_name][rejection_reason.value] += 1
            
            logger.info(f"Logged rejected trade: {ticker} - {rejection_reason.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging rejected trade: {e}")
            return False
    
    def update_subsequent_performance(self, ticker: str, price_history: pd.DataFrame):
        """
        Update rejected trades with subsequent price action
        
        Args:
            ticker: Stock ticker
            price_history: Historical price data since rejection
        """
        try:
            # Find rejected trades for this ticker
            ticker_rejections = [rt for rt in self.rejected_trades if rt.ticker == ticker]
            
            for rejected_trade in ticker_rejections:
                if rejected_trade.subsequent_price_action is not None:
                    continue  # Already updated
                
                # Analyze price action since rejection
                subsequent_data = self._analyze_subsequent_price_action(
                    rejected_trade, price_history
                )
                
                if subsequent_data:
                    rejected_trade.subsequent_price_action = subsequent_data
                    rejected_trade.would_have_been_profitable = subsequent_data['profitable']
                    rejected_trade.missed_profit_pct = subsequent_data['max_profit_pct']
                    
                    # Update threshold performance tracking
                    self._update_threshold_performance(rejected_trade, subsequent_data)
                    
        except Exception as e:
            logger.error(f"Error updating subsequent performance for {ticker}: {e}")
    
    def _analyze_subsequent_price_action(self, 
                                       rejected_trade: RejectedTrade,
                                       price_history: pd.DataFrame) -> Optional[Dict]:
        """Analyze what happened to price after trade was rejected"""
        try:
            if len(price_history) < 2:
                return None
            
            # Get data after rejection time
            rejection_time = rejected_trade.timestamp
            price_history['timestamp'] = pd.to_datetime(price_history['timestamp'])
            
            # Filter data after rejection
            subsequent_data = price_history[price_history['timestamp'] > rejection_time]
            
            if subsequent_data.empty:
                return None
            
            # Calculate price movements
            entry_price = rejected_trade.entry_price
            subsequent_prices = subsequent_data['close'].values
            
            # Maximum profit if entered
            max_price = np.max(subsequent_prices)
            max_profit_pct = ((max_price - entry_price) / entry_price) * 100
            
            # Minimum loss if entered
            min_price = np.min(subsequent_prices)
            max_loss_pct = ((min_price - entry_price) / entry_price) * 100
            
            # Final price after opportunity window
            window_end = rejection_time + timedelta(hours=self.opportunity_window_hours)
            window_data = subsequent_data[subsequent_data['timestamp'] <= window_end]
            
            if not window_data.empty:
                final_price = window_data.iloc[-1]['close']
                final_profit_pct = ((final_price - entry_price) / entry_price) * 100
            else:
                final_price = subsequent_prices[-1]
                final_profit_pct = ((final_price - entry_price) / entry_price) * 100
            
            # Determine if it would have been profitable
            profitable = max_profit_pct > 2.0  # At least 2% profit threshold
            
            # Volatility analysis
            price_changes = np.diff(subsequent_prices)
            volatility = np.std(price_changes) / np.mean(subsequent_prices) if np.mean(subsequent_prices) > 0 else 0
            
            return {
                'max_price': max_price,
                'min_price': min_price,
                'final_price': final_price,
                'max_profit_pct': max_profit_pct,
                'max_loss_pct': max_loss_pct,
                'final_profit_pct': final_profit_pct,
                'profitable': profitable,
                'volatility': volatility,
                'data_points': len(subsequent_data),
                'time_horizon_hours': (subsequent_data.iloc[-1]['timestamp'] - rejection_time).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"Error analyzing subsequent price action: {e}")
            return None
    
    def _update_threshold_performance(self, rejected_trade: RejectedTrade, subsequent_data: Dict):
        """Update threshold performance metrics"""
        try:
            # Track how thresholds performed
            threshold_key = f"{rejected_trade.rejection_reason.value}_{rejected_trade.pattern_name}"
            
            performance_data = {
                'signal_strength': rejected_trade.signal_strength,
                'threshold': rejected_trade.confidence_threshold,
                'missed_profit': subsequent_data['max_profit_pct'],
                'would_have_been_profitable': subsequent_data['profitable'],
                'timestamp': rejected_trade.timestamp
            }
            
            self.threshold_performance[threshold_key].append(performance_data)
            
        except Exception as e:
            logger.error(f"Error updating threshold performance: {e}")
    
    def reanalyze_rejected_trades(self, current_market_data: Dict) -> List[ReanalysisResult]:
        """
        Reanalyze rejected trades with current market conditions
        
        Args:
            current_market_data: Current market data for all tickers
            
        Returns:
            List of reanalysis results
        """
        try:
            results = []
            current_time = datetime.now(self.et_timezone)
            
            # Get trades eligible for reanalysis
            eligible_trades = self._get_eligible_trades(current_time)
            
            for rejected_trade in eligible_trades:
                if rejected_trade.ticker not in current_market_data:
                    continue
                
                current_data = current_market_data[rejected_trade.ticker]
                reanalysis_result = self._perform_reanalysis(rejected_trade, current_data)
                
                if reanalysis_result:
                    results.append(reanalysis_result)
                    self.reanalysis_history.append(reanalysis_result)
                    
                    # Update the original trade
                    rejected_trade.reanalysis_status = ReanalysisStatus.IMPROVED if reanalysis_result.confidence_improvement > 0.1 else ReanalysisStatus.STILL_REJECTED
                    rejected_trade.reanalysis_timestamp = current_time
                    rejected_trade.reanalysis_notes = reanalysis_result.recommendation
                    rejected_trade.follow_up_count += 1
                    
                    # Update statistics
                    if reanalysis_result.confidence_improvement > 0.1:
                        self.improvement_statistics['improved'] += 1
                    else:
                        self.improvement_statistics['still_rejected'] += 1
            
            logger.info(f"Reanalyzed {len(eligible_trades)} rejected trades, {len(results)} showed improvement")
            return results
            
        except Exception as e:
            logger.error(f"Error in reanalysis: {e}")
            return []
    
    def _get_eligible_trades(self, current_time: datetime) -> List[RejectedTrade]:
        """Get trades eligible for reanalysis"""
        eligible = []
        
        for rejected_trade in self.rejected_trades:
            # Check if within opportunity window
            time_since_rejection = current_time - rejected_trade.timestamp
            if time_since_rejection > timedelta(hours=self.opportunity_window_hours):
                continue
            
            # Check if we haven't exceeded follow-up attempts
            if rejected_trade.follow_up_count >= self.max_follow_up_attempts:
                continue
            
            # Check if enough time has passed since last analysis
            if rejected_trade.reanalysis_timestamp:
                time_since_reanalysis = current_time - rejected_trade.reanalysis_timestamp
                if time_since_reanalysis < timedelta(minutes=self.reanalysis_interval_minutes):
                    continue
            
            eligible.append(rejected_trade)
        
        return eligible
    
    def _perform_reanalysis(self, 
                           rejected_trade: RejectedTrade,
                           current_data: Dict) -> Optional[ReanalysisResult]:
        """Perform detailed reanalysis of a rejected trade"""
        try:
            # Calculate new signal strength with current data
            new_signal_strength = self._calculate_current_signal_strength(rejected_trade, current_data)
            
            # Determine improvement
            confidence_improvement = new_signal_strength - rejected_trade.signal_strength
            
            # Analyze reasons for change
            reasons_for_change = []
            
            # Volume improvement
            current_volume_ratio = current_data.get('volume_ratio', 1)
            if current_volume_ratio > rejected_trade.volume_ratio * 1.5:
                reasons_for_change.append(f"Volume increased from {rejected_trade.volume_ratio:.1f}x to {current_volume_ratio:.1f}x")
            
            # Volatility improvement
            current_volatility = current_data.get('volatility_score', 0)
            if current_volatility < rejected_trade.volatility_score * 0.8:
                reasons_for_change.append(f"Volatility decreased from {rejected_trade.volatility_score:.2f} to {current_volatility:.2f}")
            
            # Multi-timeframe improvement
            current_mt_confidence = current_data.get('multi_timeframe_confidence', 0)
            if current_mt_confidence > rejected_trade.multi_timeframe_confidence * 1.2:
                reasons_for_change.append(f"Multi-timeframe confidence improved")
            
            # Price movement analysis
            current_price = current_data.get('price', rejected_trade.entry_price)
            price_change = (current_price - rejected_trade.entry_price) / rejected_trade.entry_price * 100
            
            if abs(price_change) > 2:
                reasons_for_change.append(f"Price moved {price_change:.1f}% since rejection")
            
            # Make recommendation
            recommendation = self._make_reanalysis_recommendation(
                rejected_trade, new_signal_strength, confidence_improvement, reasons_for_change
            )
            
            # Calculate updated thresholds
            updated_thresholds = None
            if confidence_improvement > 0.1:
                updated_thresholds = self._calculate_updated_thresholds(rejected_trade, current_data)
            
            return ReanalysisResult(
                original_rejection=rejected_trade,
                current_conditions=current_data,
                new_signal_strength=new_signal_strength,
                recommendation=recommendation,
                confidence_improvement=confidence_improvement,
                reasons_for_change=reasons_for_change,
                updated_thresholds=updated_thresholds
            )
            
        except Exception as e:
            logger.error(f"Error performing reanalysis: {e}")
            return None
    
    def _calculate_current_signal_strength(self, 
                                         rejected_trade: RejectedTrade,
                                         current_data: Dict) -> float:
        """Calculate current signal strength based on updated data"""
        try:
            # Base signal strength
            base_strength = rejected_trade.signal_strength
            
            # Volume component
            current_volume_ratio = current_data.get('volume_ratio', 1)
            volume_score = min(current_volume_ratio / 3.0, 1.0) * 0.3
            
            # Volatility component (lower is better for entry)
            current_volatility = current_data.get('volatility_score', 0)
            volatility_score = max(0, (1 - current_volatility)) * 0.2
            
            # Multi-timeframe component
            current_mt_confidence = current_data.get('multi_timeframe_confidence', 0)
            mt_score = current_mt_confidence * 0.3
            
            # Price momentum component
            price_momentum = current_data.get('price_momentum', 0)
            momentum_score = min(abs(price_momentum) / 5.0, 1.0) * 0.2
            
            # Combine scores
            new_strength = base_strength * 0.4 + volume_score + volatility_score + mt_score + momentum_score
            
            return min(new_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating current signal strength: {e}")
            return rejected_trade.signal_strength
    
    def _make_reanalysis_recommendation(self, 
                                      rejected_trade: RejectedTrade,
                                      new_signal_strength: float,
                                      confidence_improvement: float,
                                      reasons_for_change: List[str]) -> str:
        """Make recommendation based on reanalysis"""
        try:
            # High confidence improvement
            if confidence_improvement > 0.2 and new_signal_strength > 0.8:
                return "enter_now"
            
            # Moderate improvement
            elif confidence_improvement > 0.1 and new_signal_strength > 0.7:
                return "continue_monitoring"
            
            # Significant positive price movement
            if rejected_trade.subsequent_price_action:
                price_change = rejected_trade.subsequent_price_action.get('final_profit_pct', 0)
                if price_change > 5:
                    return "enter_now"
                elif price_change > 2:
                    return "continue_monitoring"
            
            # No significant improvement
            return "permanent_reject"
            
        except Exception as e:
            logger.error(f"Error making recommendation: {e}")
            return "permanent_reject"
    
    def _calculate_updated_thresholds(self, 
                                   rejected_trade: RejectedTrade,
                                   current_data: Dict) -> Dict:
        """Calculate updated thresholds based on learning"""
        try:
            updated_thresholds = {}
            
            # Get historical performance for this rejection reason
            reason_key = rejected_trade.rejection_reason.value
            pattern_key = rejected_trade.pattern_name
            
            # Analyze if thresholds should be adjusted
            threshold_data = self.threshold_performance.get(f"{reason_key}_{pattern_key}", [])
            
            if len(threshold_data) >= 10:  # Need sufficient data
                # Calculate missed opportunities
                missed_opportunities = [td for td in threshold_data if td['would_have_been_profitable']]
                
                if len(missed_opportunities) / len(threshold_data) > 0.6:  # >60% missed opportunities
                    # Suggest lowering threshold
                    avg_missed_profit = np.mean([td['missed_profit'] for td in missed_opportunities])
                    
                    if reason_key == 'insufficient_confidence':
                        new_confidence = max(rejected_trade.confidence_threshold - 0.05, 0.5)
                        updated_thresholds['confidence'] = new_confidence
                    
                    elif reason_key == 'low_volume':
                        new_volume = max(rejected_trade.volume_threshold - 0.5, 1.0)
                        updated_thresholds['volume'] = new_volume
                    
                    elif reason_key == 'high_volatility':
                        new_volatility = rejected_trade.volatility_threshold + 0.1
                        updated_thresholds['volatility'] = new_volatility
            
            return updated_thresholds
            
        except Exception as e:
            logger.error(f"Error calculating updated thresholds: {e}")
            return {}
    
    def get_rejected_trade_summary(self) -> Dict:
        """Get comprehensive summary of rejected trades"""
        try:
            if not self.rejected_trades:
                return {
                    'total_rejected': 0,
                    'rejection_reasons': {},
                    'pattern_performance': {},
                    'missed_opportunities': {},
                    'threshold_adjustments': {}
                }
            
            # Basic statistics
            total_rejected = len(self.rejected_trades)
            
            # Rejection reasons breakdown
            rejection_breakdown = Counter(rt.rejection_reason.value for rt in self.rejected_trades)
            
            # Pattern performance
            pattern_stats = {}
            for pattern_name in set(rt.pattern_name for rt in self.rejected_trades):
                pattern_trades = [rt for rt in self.rejected_trades if rt.pattern_name == pattern_name]
                pattern_stats[pattern_name] = {
                    'total_rejections': len(pattern_trades),
                    'avg_signal_strength': np.mean([rt.signal_strength for rt in pattern_trades]),
                    'rejection_reasons': Counter(rt.rejection_reason.value for rt in pattern_trades)
                }
            
            # Missed opportunities
            profitable_rejections = [rt for rt in self.rejected_trades 
                                   if rt.would_have_been_profitable and rt.missed_profit_pct]
            
            missed_opportunities = {
                'count': len(profitable_rejections),
                'avg_missed_profit_pct': np.mean([rt.missed_profit_pct for rt in profitable_rejections]) if profitable_rejections else 0,
                'total_missed_profit': sum([rt.missed_profit_pct for rt in profitable_rejections]) if profitable_rejections else 0,
                'max_missed_profit': max([rt.missed_profit_pct for rt in profitable_rejections]) if profitable_rejections else 0
            }
            
            # Reanalysis statistics
            reanalysis_stats = {
                'total_reanalyzed': len(self.reanalysis_history),
                'improved_count': self.improvement_statistics['improved'],
                'still_rejected_count': self.improvement_statistics['still_rejected'],
                'conversion_rate': self.improvement_statistics['improved'] / max(len(self.reanalysis_history), 1)
            }
            
            # Threshold adjustment suggestions
            threshold_suggestions = {}
            for key, performance_data in self.threshold_performance.items():
                if len(performance_data) >= 10:
                    missed_opps = [td for td in performance_data if td['would_have_been_profitable']]
                    if len(missed_opps) / len(performance_data) > 0.6:
                        threshold_suggestions[key] = {
                            'current_threshold': np.mean([td['threshold'] for td in performance_data]),
                            'suggested_adjustment': 'lower',
                            'missed_opportunity_rate': len(missed_opps) / len(performance_data),
                            'avg_missed_profit': np.mean([td['missed_profit'] for td in missed_opps])
                        }
            
            return {
                'total_rejected': total_rejected,
                'rejection_reasons': dict(rejection_breakdown),
                'pattern_performance': pattern_stats,
                'missed_opportunities': missed_opportunities,
                'reanalysis_stats': reanalysis_stats,
                'threshold_adjustments': threshold_suggestions,
                'recent_rejections': [
                    {
                        'ticker': rt.ticker,
                        'timestamp': rt.timestamp.isoformat(),
                        'reason': rt.rejection_reason.value,
                        'pattern': rt.pattern_name,
                        'signal_strength': rt.signal_strength,
                        'missed_profit': rt.missed_profit_pct,
                        'follow_up_count': rt.follow_up_count
                    }
                    for rt in sorted(self.rejected_trades, key=lambda x: x.timestamp, reverse=True)[:20]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting rejected trade summary: {e}")
            return {}
    
    def get_improvement_recommendations(self) -> List[Dict]:
        """Get actionable improvement recommendations"""
        try:
            recommendations = []
            
            # Analyze rejection patterns
            summary = self.get_rejected_trade_summary()
            
            # High missed opportunity rate
            if summary['missed_opportunities']['count'] > 0:
                missed_rate = summary['missed_opportunities']['count'] / max(summary['total_rejected'], 1)
                if missed_rate > 0.3:  # >30% missed opportunities
                    recommendations.append({
                        'type': 'threshold_adjustment',
                        'priority': 'high',
                        'description': f"High missed opportunity rate ({missed_rate:.1%}). Consider lowering entry thresholds.",
                        'suggested_actions': [
                            "Review confidence thresholds for patterns with high missed rates",
                            "Consider volume threshold adjustments for different market conditions",
                            "Implement dynamic volatility thresholds"
                        ]
                    })
            
            # Pattern-specific issues
            for pattern, stats in summary['pattern_performance'].items():
                if stats['avg_signal_strength'] < 0.6 and stats['total_rejections'] > 5:
                    recommendations.append({
                        'type': 'pattern_improvement',
                        'priority': 'medium',
                        'description': f"Pattern {pattern} has low average signal strength ({stats['avg_signal_strength']:.2f})",
                        'suggested_actions': [
                            f"Review {pattern} detection logic",
                            "Consider additional confirmation indicators",
                            "Adjust pattern-specific thresholds"
                        ]
                    })
            
            # Reanalysis effectiveness
            reanalysis_stats = summary.get('reanalysis_stats', {})
            if reanalysis_stats.get('conversion_rate', 0) < 0.1:
                recommendations.append({
                    'type': 'reanalysis_improvement',
                    'priority': 'medium',
                    'description': "Low reanalysis conversion rate. Consider improving reanalysis criteria.",
                    'suggested_actions': [
                        "Extend opportunity window for certain patterns",
                        "Implement more sensitive reanalysis triggers",
                        "Add market condition awareness to reanalysis"
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting improvement recommendations: {e}")
            return []
    
    def export_rejected_trades_data(self, filepath: str):
        """Export rejected trades data for external analysis"""
        try:
            export_data = {
                'rejected_trades': [
                    {
                        'ticker': rt.ticker,
                        'timestamp': rt.timestamp.isoformat(),
                        'rejection_reason': rt.rejection_reason.value,
                        'rejection_details': rt.rejection_details,
                        'entry_price': rt.entry_price,
                        'signal_strength': rt.signal_strength,
                        'pattern_name': rt.pattern_name,
                        'volume_ratio': rt.volume_ratio,
                        'volatility_score': rt.volatility_score,
                        'multi_timeframe_confidence': rt.multi_timeframe_confidence,
                        'market_condition': rt.market_condition,
                        'confidence_threshold': rt.confidence_threshold,
                        'volume_threshold': rt.volume_threshold,
                        'volatility_threshold': rt.volatility_threshold,
                        'would_have_been_profitable': rt.would_have_been_profitable,
                        'missed_profit_pct': rt.missed_profit_pct,
                        'follow_up_count': rt.follow_up_count,
                        'reanalysis_status': rt.reanalysis_status.value
                    }
                    for rt in self.rejected_trades
                ],
                'reanalysis_history': [
                    {
                        'ticker': rr.original_rejection.ticker,
                        'timestamp': rr.original_rejection.timestamp.isoformat(),
                        'new_signal_strength': rr.new_signal_strength,
                        'confidence_improvement': rr.confidence_improvement,
                        'recommendation': rr.recommendation,
                        'reasons_for_change': rr.reasons_for_change
                    }
                    for rr in self.reanalysis_history
                ],
                'summary': self.get_rejected_trade_summary(),
                'recommendations': self.get_improvement_recommendations()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Rejected trades data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting rejected trades data: {e}")
    
    def _load_rejected_trades(self):
        """Load existing rejected trades data"""
        try:
            data_file = self.data_path / "rejected_trades.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Load rejected trades
                for rt_data in data.get('rejected_trades', []):
                    rejected_trade = RejectedTrade(
                        ticker=rt_data['ticker'],
                        timestamp=pd.to_datetime(rt_data['timestamp']),
                        rejection_reason=RejectionReason(rt_data['rejection_reason']),
                        rejection_details=rt_data['rejection_details'],
                        entry_price=rt_data['entry_price'],
                        signal_strength=rt_data['signal_strength'],
                        pattern_name=rt_data['pattern_name'],
                        volume_ratio=rt_data['volume_ratio'],
                        volatility_score=rt_data['volatility_score'],
                        multi_timeframe_confidence=rt_data['multi_timeframe_confidence'],
                        market_condition=rt_data['market_condition'],
                        confidence_threshold=rt_data['confidence_threshold'],
                        volume_threshold=rt_data['volume_threshold'],
                        volatility_threshold=rt_data['volatility_threshold'],
                        would_have_been_profitable=rt_data.get('would_have_been_profitable'),
                        missed_profit_pct=rt_data.get('missed_profit_pct'),
                        follow_up_count=rt_data.get('follow_up_count', 0),
                        reanalysis_status=ReanalysisStatus(rt_data.get('reanalysis_status', 'pending'))
                    )
                    self.rejected_trades.append(rejected_trade)
                
                logger.info(f"Loaded {len(self.rejected_trades)} rejected trades from file")
                
        except Exception as e:
            logger.error(f"Error loading rejected trades: {e}")
    
    def save_rejected_trades(self):
        """Save rejected trades data to file"""
        try:
            self.export_rejected_trades_data(str(self.data_path / "rejected_trades.json"))
        except Exception as e:
            logger.error(f"Error saving rejected trades: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old rejected trades data"""
        try:
            cutoff_date = datetime.now(self.et_timezone) - timedelta(days=days_to_keep)
            
            # Remove old rejected trades
            original_count = len(self.rejected_trades)
            self.rejected_trades = [rt for rt in self.rejected_trades if rt.timestamp > cutoff_date]
            
            # Remove old reanalysis history
            original_reanalysis_count = len(self.reanalysis_history)
            self.reanalysis_history = [rh for rh in self.reanalysis_history 
                                     if rh.original_rejection.timestamp > cutoff_date]
            
            # Clean up threshold performance data
            for key in list(self.threshold_performance.keys()):
                self.threshold_performance[key] = [
                    td for td in self.threshold_performance[key] 
                    if td['timestamp'] > cutoff_date
                ]
                if not self.threshold_performance[key]:
                    del self.threshold_performance[key]
            
            removed_count = original_count - len(self.rejected_trades)
            logger.info(f"Cleaned up {removed_count} old rejected trades (older than {days_to_keep} days)")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
