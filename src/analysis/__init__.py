"""
Analysis Package
Contains pattern detection, premarket analysis, and stock discovery modules
"""
# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'PatternDetector':
        from analysis.pattern_detector import PatternDetector
        return PatternDetector
    elif name == 'PatternSignal':
        from analysis.pattern_detector import PatternSignal
        return PatternSignal
    elif name == 'PreMarketAnalyzer':
        from analysis.premarket_analyzer import PreMarketAnalyzer
        return PreMarketAnalyzer
    elif name == 'StockDiscovery':
        from analysis.stock_discovery import StockDiscovery
        return StockDiscovery
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PatternDetector',
    'PatternSignal',
    'PreMarketAnalyzer',
    'StockDiscovery'
]
