"""
Adaptive Learning System
Machine learning-based system that learns from past trades to improve future performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
from pathlib import Path

# ML Models (commented out due to compatibility issues)
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning system modes"""
    OFF = "off"
    PASSIVE = "passive"  # Only collect data
    ACTIVE = "active"    # Update models
    AUTO_TUNE = "auto_tune"  # Automatically adjust parameters

@dataclass
class TradeOutcome:
    """Trade outcome for learning"""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str
    entry_pattern: str
    entry_confidence: float
    multi_timeframe_confidence: float
    volume_ratio: float
    volatility_score: float
    market_condition: str
    pnl_pct: float
    pnl_dollars: float
    hold_time_minutes: float
    max_unrealized_pct: float
    partial_profits_taken: List[float]
    exit_reason: str
    success: bool  # True if profitable
    success_score: float  # 0-1, based on profit relative to target

@dataclass
class FeatureSet:
    """Feature set for ML models"""
    features: np.ndarray
    feature_names: List[str]
    target: float  # For regression (profit percentage)
    classification_target: int  # For classification (win/loss)
    timestamp: datetime

class AdaptiveLearningSystem:
    """
    Machine learning system that learns from trading outcomes to improve strategy
    """
    
    def __init__(self, 
                 learning_mode: LearningMode = LearningMode.PASSIVE,
                 model_update_frequency: int = 50,  # Update after N trades
                 min_samples_for_training: int = 100):
        """
        Args:
            learning_mode: Current learning mode
            model_update_frequency: Update models after this many trades
            min_samples_for_training: Minimum samples needed for training
        """
        self.learning_mode = learning_mode
        self.model_update_frequency = model_update_frequency
        self.min_samples_for_training = min_samples_for_training
        
        # Data storage
        self.trade_history: List[TradeOutcome] = []
        self.feature_sets: List[FeatureSet] = []
        
        # ML models (commented out due to compatibility issues)
        # self.entry_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        # self.profit_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        # self.scaler = StandardScaler()
        
        # Model performance tracking
        self.model_performance = {
            'entry_classifier_accuracy': 0.0,
            'profit_regressor_mse': 0.0,
            'last_update_time': None,
            'total_updates': 0
        }
        
        # Adaptive parameters
        self.adaptive_parameters = {
            'confidence_threshold': 0.72,
            'volume_threshold': 1.5,
            'volatility_threshold': 0.7,
            'pattern_weights': {
                'Volume_Breakout_Momentum': 0.85,
                'RSI_Accumulation_Entry': 0.75,
                'Golden_Cross_Volume': 0.78,
                'Slow_Accumulation': 0.80,
                'MACD_Acceleration_Breakout': 0.82,
                'Consolidation_Breakout': 0.83
            }
        }
        
        # Learning metrics
        self.learning_metrics = {
            'total_trades_analyzed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'prediction_accuracy': 0.0,
            'parameter_adjustments': 0
        }
        
        # Storage paths
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def add_trade_outcome(self, trade_data: Dict) -> bool:
        """
        Add trade outcome to learning system
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            True if successfully added
        """
        try:
            # Create TradeOutcome object
            outcome = TradeOutcome(
                ticker=trade_data.get('ticker', ''),
                entry_time=pd.to_datetime(trade_data.get('entry_time')),
                exit_time=pd.to_datetime(trade_data.get('exit_time')),
                entry_price=float(trade_data.get('entry_price', 0)),
                exit_price=float(trade_data.get('exit_price', 0)),
                position_type=trade_data.get('position_type', 'unknown'),
                entry_pattern=trade_data.get('entry_pattern', ''),
                entry_confidence=float(trade_data.get('entry_confidence', 0)),
                multi_timeframe_confidence=float(trade_data.get('multi_timeframe_confidence', 0)),
                volume_ratio=float(trade_data.get('volume_ratio', 1)),
                volatility_score=float(trade_data.get('volatility_score', 0)),
                market_condition=trade_data.get('market_condition', 'normal'),
                pnl_pct=float(trade_data.get('pnl_pct', 0)),
                pnl_dollars=float(trade_data.get('pnl_dollars', 0)),
                hold_time_minutes=float(trade_data.get('hold_time_minutes', 0)),
                max_unrealized_pct=float(trade_data.get('max_unrealized_pct', 0)),
                partial_profits_taken=trade_data.get('partial_profits_taken', []),
                exit_reason=trade_data.get('exit_reason', ''),
                success=trade_data.get('pnl_pct', 0) > 0,
                success_score=self._calculate_success_score(trade_data)
            )
            
            self.trade_history.append(outcome)
            
            # Extract features for ML
            features = self._extract_features(outcome)
            if features:
                self.feature_sets.append(features)
            
            # Update learning metrics
            self.learning_metrics['total_trades_analyzed'] += 1
            
            # Check if models should be updated
            if (self.learning_mode in [LearningMode.ACTIVE, LearningMode.AUTO_TUNE] and
                len(self.trade_history) >= self.min_samples_for_training and
                len(self.trade_history) % self.model_update_frequency == 0):
                self._update_models()
            
            # Auto-tune parameters if enabled
            if self.learning_mode == LearningMode.AUTO_TUNE:
                self._auto_tune_parameters()
            
            logger.info(f"Added trade outcome: {outcome.ticker} - {outcome.pnl_pct:.2f}% ({'profit' if outcome.success else 'loss'})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade outcome: {e}")
            return False
    
    def _calculate_success_score(self, trade_data: Dict) -> float:
        """Calculate success score (0-1) based on profit relative to expectations"""
        pnl_pct = trade_data.get('pnl_pct', 0)
        position_type = trade_data.get('position_type', 'swing')
        
        # Different success criteria for different position types
        target_profits = {
            'scalp': 2.0,
            'swing': 5.0,
            'surge': 10.0,
            'slow_mover': 4.0
        }
        
        target = target_profits.get(position_type, 5.0)
        
        if pnl_pct >= target:
            return 1.0
        elif pnl_pct > 0:
            return min(pnl_pct / target, 1.0)
        else:
            # Losses get scores based on severity
            return max(0, 1 + (pnl_pct / target))  # Negative score for losses
    
    def _extract_features(self, outcome: TradeOutcome) -> Optional[FeatureSet]:
        """Extract features from trade outcome"""
        try:
            feature_names = [
                'entry_confidence',
                'multi_timeframe_confidence',
                'volume_ratio',
                'volatility_score',
                'hour_of_day',
                'day_of_week',
                'market_condition_encoded',
                'position_type_encoded',
                'entry_pattern_encoded',
                'hold_time_normalized',
                'max_unrealized_pct'
            ]
            
            # Encode categorical variables
            market_condition_map = {'normal': 0, 'volatile': 1, 'extreme': 2, 'news_driven': 3}
            position_type_map = {'scalp': 0, 'swing': 1, 'surge': 2, 'slow_mover': 3}
            pattern_map = {
                'Volume_Breakout_Momentum': 0,
                'RSI_Accumulation_Entry': 1,
                'Golden_Cross_Volume': 2,
                'Slow_Accumulation': 3,
                'MACD_Acceleration_Breakout': 4,
                'Consolidation_Breakout': 5
            }
            
            features = np.array([
                outcome.entry_confidence,
                outcome.multi_timeframe_confidence,
                outcome.volume_ratio,
                outcome.volatility_score,
                outcome.entry_time.hour,
                outcome.entry_time.weekday(),
                market_condition_map.get(outcome.market_condition, 0),
                position_type_map.get(outcome.position_type, 0),
                pattern_map.get(outcome.entry_pattern, 0),
                min(outcome.hold_time_minutes / 240, 1.0),  # Normalize to 0-1 (4 hours max)
                min(outcome.max_unrealized_pct / 20, 1.0)  # Normalize to 0-1 (20% max)
            ])
            
            return FeatureSet(
                features=features,
                feature_names=feature_names,
                target=outcome.pnl_pct,
                classification_target=1 if outcome.success else 0,
                timestamp=outcome.exit_time
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _update_models(self):
        """Update ML models with latest trade data"""
        logger.info("ML model update skipped - sklearn compatibility issues")
        return
    
    def _auto_tune_parameters(self):
        """Automatically adjust trading parameters based on learning"""
        try:
            if len(self.trade_history) < 50:
                return
            
            # Analyze recent performance
            recent_trades = self.trade_history[-50:]
            
            # Calculate success rates by pattern
            pattern_performance = {}
            for trade in recent_trades:
                pattern = trade.entry_pattern
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'wins': 0, 'total': 0, 'avg_pnl': 0}
                
                pattern_performance[pattern]['total'] += 1
                if trade.success:
                    pattern_performance[pattern]['wins'] += 1
                pattern_performance[pattern]['avg_pnl'] += trade.pnl_pct
            
            # Calculate win rates and update pattern weights
            for pattern, stats in pattern_performance.items():
                if stats['total'] >= 5:  # Minimum samples for adjustment
                    win_rate = stats['wins'] / stats['total']
                    avg_pnl = stats['avg_pnl'] / stats['total']
                    
                    # Adjust pattern weight based on performance
                    current_weight = self.adaptive_parameters['pattern_weights'].get(pattern, 0.75)
                    
                    if win_rate > 0.6 and avg_pnl > 3.0:
                        # Increase weight for successful patterns
                        new_weight = min(current_weight + 0.05, 0.95)
                    elif win_rate < 0.4 or avg_pnl < -2.0:
                        # Decrease weight for poor performing patterns
                        new_weight = max(current_weight - 0.05, 0.5)
                    else:
                        new_weight = current_weight
                    
                    if abs(new_weight - current_weight) > 0.01:
                        self.adaptive_parameters['pattern_weights'][pattern] = new_weight
                        self.learning_metrics['parameter_adjustments'] += 1
                        logger.info(f"Adjusted {pattern} weight: {current_weight:.3f} -> {new_weight:.3f}")
            
            # Adjust confidence threshold based on overall performance
            overall_win_rate = sum(1 for t in recent_trades if t.success) / len(recent_trades)
            
            if overall_win_rate < 0.4:
                # Too many losses, increase confidence threshold
                new_threshold = min(self.adaptive_parameters['confidence_threshold'] + 0.02, 0.9)
                self.adaptive_parameters['confidence_threshold'] = new_threshold
                logger.info(f"Increased confidence threshold to {new_threshold:.3f} due to low win rate")
            elif overall_win_rate > 0.7:
                # High win rate, can be more aggressive
                new_threshold = max(self.adaptive_parameters['confidence_threshold'] - 0.02, 0.6)
                self.adaptive_parameters['confidence_threshold'] = new_threshold
                logger.info(f"Decreased confidence threshold to {new_threshold:.3f} due to high win rate")
            
        except Exception as e:
            logger.error(f"Error in auto-tune: {e}")
    
    def predict_trade_success(self, trade_features: Dict) -> Tuple[float, float]:
        """
        Predict trade success probability and expected profit
        
        Args:
            trade_features: Dictionary with trade features
            
        Returns:
            Tuple of (success_probability, expected_profit_pct)
        """
        try:
            if self.learning_mode == LearningMode.OFF or len(self.feature_sets) < self.min_samples_for_training:
                return 0.5, 0.0  # Default predictions
            
            # Prepare features
            feature_vector = self._prepare_prediction_features(trade_features)
            if feature_vector is None:
                return 0.5, 0.0
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make predictions
            success_prob = self.entry_classifier.predict_proba(feature_vector_scaled)[0][1]
            expected_profit = self.profit_regressor.predict(feature_vector_scaled)[0]
            
            # Update learning metrics
            self.learning_metrics['successful_predictions'] += 1
            
            return success_prob, expected_profit
            
        except Exception as e:
            logger.error(f"Error predicting trade success: {e}")
            self.learning_metrics['failed_predictions'] += 1
            return 0.5, 0.0
    
    def _prepare_prediction_features(self, trade_features: Dict) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        try:
            # Map categorical variables
            market_condition_map = {'normal': 0, 'volatile': 1, 'extreme': 2, 'news_driven': 3}
            position_type_map = {'scalp': 0, 'swing': 1, 'surge': 2, 'slow_mover': 3}
            pattern_map = {
                'Volume_Breakout_Momentum': 0,
                'RSI_Accumulation_Entry': 1,
                'Golden_Cross_Volume': 2,
                'Slow_Accumulation': 3,
                'MACD_Acceleration_Breakout': 4,
                'Consolidation_Breakout': 5
            }
            
            features = np.array([
                trade_features.get('entry_confidence', 0.7),
                trade_features.get('multi_timeframe_confidence', 0.7),
                trade_features.get('volume_ratio', 1.0),
                trade_features.get('volatility_score', 0.5),
                datetime.now().hour,
                datetime.now().weekday(),
                market_condition_map.get(trade_features.get('market_condition', 'normal'), 0),
                position_type_map.get(trade_features.get('position_type', 'swing'), 0),
                pattern_map.get(trade_features.get('entry_pattern', ''), 0),
                0.5,  # Default hold time (normalized)
                0.5   # Default max unrealized (normalized)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return None
    
    def get_adaptive_parameters(self) -> Dict:
        """Get current adaptive parameters"""
        return self.adaptive_parameters.copy()
    
    def update_adaptive_parameter(self, parameter: str, value: float):
        """Update a specific adaptive parameter"""
        try:
            if parameter in self.adaptive_parameters:
                old_value = self.adaptive_parameters[parameter]
                self.adaptive_parameters[parameter] = value
                logger.info(f"Updated parameter {parameter}: {old_value} -> {value}")
            elif parameter.startswith('pattern_weight_'):
                pattern_name = parameter.replace('pattern_weight_', '')
                if pattern_name in self.adaptive_parameters['pattern_weights']:
                    old_value = self.adaptive_parameters['pattern_weights'][pattern_name]
                    self.adaptive_parameters['pattern_weights'][pattern_name] = value
                    logger.info(f"Updated pattern weight {pattern_name}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown parameter: {parameter}")
                
        except Exception as e:
            logger.error(f"Error updating parameter: {e}")
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning summary"""
        try:
            # Calculate recent performance
            recent_trades = self.trade_history[-100:] if len(self.trade_history) >= 100 else self.trade_history
            
            if recent_trades:
                recent_win_rate = sum(1 for t in recent_trades if t.success) / len(recent_trades)
                recent_avg_pnl = np.mean([t.pnl_pct for t in recent_trades])
                recent_avg_hold_time = np.mean([t.hold_time_minutes for t in recent_trades])
            else:
                recent_win_rate = 0
                recent_avg_pnl = 0
                recent_avg_hold_time = 0
            
            # Pattern performance analysis
            pattern_stats = {}
            for pattern in self.adaptive_parameters['pattern_weights'].keys():
                pattern_trades = [t for t in recent_trades if t.entry_pattern == pattern]
                if pattern_trades:
                    wins = sum(1 for t in pattern_trades if t.success)
                    pattern_stats[pattern] = {
                        'trades': len(pattern_trades),
                        'wins': wins,
                        'win_rate': wins / len(pattern_trades),
                        'avg_pnl': np.mean([t.pnl_pct for t in pattern_trades]),
                        'weight': self.adaptive_parameters['pattern_weights'][pattern]
                    }
            
            return {
                'learning_mode': self.learning_mode.value,
                'total_trades_analyzed': len(self.trade_history),
                'feature_sets_available': len(self.feature_sets),
                'model_performance': self.model_performance,
                'learning_metrics': self.learning_metrics,
                'recent_performance': {
                    'win_rate': recent_win_rate,
                    'avg_pnl_pct': recent_avg_pnl,
                    'avg_hold_time_minutes': recent_avg_hold_time,
                    'sample_size': len(recent_trades)
                },
                'adaptive_parameters': self.adaptive_parameters,
                'pattern_performance': pattern_stats,
                'prediction_accuracy': self.learning_metrics['prediction_accuracy']
            }
            
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {}
    
    def export_learning_data(self, filepath: str):
        """Export learning data for analysis"""
        try:
            export_data = {
                'trade_history': [
                    {
                        'ticker': t.ticker,
                        'entry_time': t.entry_time.isoformat(),
                        'exit_time': t.exit_time.isoformat(),
                        'pnl_pct': t.pnl_pct,
                        'success': t.success,
                        'entry_pattern': t.entry_pattern,
                        'position_type': t.position_type,
                        'volume_ratio': t.volume_ratio,
                        'volatility_score': t.volatility_score,
                        'market_condition': t.market_condition
                    }
                    for t in self.trade_history
                ],
                'adaptive_parameters': self.adaptive_parameters,
                'model_performance': self.model_performance,
                'learning_metrics': self.learning_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Learning data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
    
    def _save_models(self):
        """Save ML models to disk"""
        try:
            # Save models
            with open(self.models_path / 'entry_classifier.pkl', 'wb') as f:
                pickle.dump(self.entry_classifier, f)
            
            with open(self.models_path / 'profit_regressor.pkl', 'wb') as f:
                pickle.dump(self.profit_regressor, f)
            
            with open(self.models_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save parameters
            with open(self.models_path / 'adaptive_parameters.json', 'w') as f:
                json.dump(self.adaptive_parameters, f, indent=2)
            
            logger.info("Models and parameters saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load ML models from disk"""
        try:
            # Load models
            classifier_path = self.models_path / 'entry_classifier.pkl'
            if classifier_path.exists():
                with open(classifier_path, 'rb') as f:
                    self.entry_classifier = pickle.load(f)
                logger.info("Loaded entry classifier model")
            
            regressor_path = self.models_path / 'profit_regressor.pkl'
            if regressor_path.exists():
                with open(regressor_path, 'rb') as f:
                    self.profit_regressor = pickle.load(f)
                logger.info("Loaded profit regressor model")
            
            scaler_path = self.models_path / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded feature scaler")
            
            # Load parameters
            params_path = self.models_path / 'adaptive_parameters.json'
            if params_path.exists():
                with open(params_path, 'r') as f:
                    loaded_params = json.load(f)
                self.adaptive_parameters.update(loaded_params)
                logger.info("Loaded adaptive parameters")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def reset_learning(self):
        """Reset learning system"""
        try:
            self.trade_history.clear()
            self.feature_sets.clear()
            
            # Reset models to default
            self.entry_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.profit_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            # Reset performance metrics
            self.model_performance = {
                'entry_classifier_accuracy': 0.0,
                'profit_regressor_mse': 0.0,
                'last_update_time': None,
                'total_updates': 0
            }
            
            self.learning_metrics = {
                'total_trades_analyzed': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'prediction_accuracy': 0.0,
                'parameter_adjustments': 0
            }
            
            logger.info("Learning system reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting learning system: {e}")
    
    def set_learning_mode(self, mode: LearningMode):
        """Set learning mode"""
        old_mode = self.learning_mode
        self.learning_mode = mode
        logger.info(f"Learning mode changed: {old_mode.value} -> {mode.value}")
        
        if mode == LearningMode.OFF:
            logger.info("Learning system disabled - no data collection or model updates")
        elif mode == LearningMode.PASSIVE:
            logger.info("Learning system in passive mode - collecting data only")
        elif mode == LearningMode.ACTIVE:
            logger.info("Learning system in active mode - updating models")
        elif mode == LearningMode.AUTO_TUNE:
            logger.info("Learning system in auto-tune mode - automatically adjusting parameters")
