#!/usr/bin/env python3
"""
Standalone Feature Analysis Script for Trading Prediction Model

This script provides a comprehensive analysis of features without requiring
all the complex dependencies from model_trainer.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    def __init__(self, base_directory='/home/parrochia/go/src/github.com/trading-prediction-with-machine-learning/brent'):
        self.base_directory = base_directory
        self.issues = []
        self.feature_stats = {}
        
        # Complete list of expected features based on selected_features from model_trainer.py
        self.expected_features = [
            # Time features
            'minute', 'day', 'hour', 'day_of_week', 'day_of_year',
            'sin_day', 'cos_day', 'sin_hour', 'cos_hour',
            'market_open_hour', 'stock_open_hour', 'is_summer',
            'period_of_day',
            
            # Basic OHLCV
            'Open', 'High', 'Low', 'Close', 'Volume',
            
            # Volume indicators and moving averages
            'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20',
            'Volume_Ratio_SMA5', 'Volume_Ratio_SMA10', 'Volume_Ratio_SMA20',
            'Volume_Change_1', 'Volume_Change_5',
            
            # Technical indicators - RSI family
            'RSI_14', 'RSI_SMA_7', 'RSI_Trend', 'RSI', 'SMA_RSI',
            'Vol_Weighted_RSI', 'Vol_Weighted_RSI_SMA',
            'Stoch_RSI',
            
            # Technical indicators - Trend and momentum
            'MACD', 'MACD_Signal', 'ATR_14', 'ADX', 'Williams_R',
            'SMA_10', 'EMA_10',
            
            # Bollinger Bands
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            
            # Keltner Channels
            'Keltner_High', 'Keltner_Low', 'Keltner_Width',
            
            # SuperTrend
            'SuperTrend_Trend', 'SuperTrend_Long', 'SuperTrend_Short',
            
            # CCI indicators
            'CCI_5', 'CCI_10', 'CCI_20', 'CCI_40', 'CCI_80',
            
            # Volume analysis
            'PV_Ratio', 'PV_Change', 'OBV', 'ADL', 'MFM', 'CMF_20', 'PVT',
            'Volume_Oscillator', 'VWAP_10', 'VWMA_10', 'VWMA_20',
            'Vol_Weighted_Up', 'Vol_Weighted_Down', 'Vol_Weighted_Down_Avg',
            
            # Force Index and Money Flow
            'Force_Index_1', 'Force_Index_13',
            'Typical_Price', 'Raw_Money_Flow', 'Typical_Price_Prev',
            'Money_Flow_Positive',
            
            # Candlestick features
            'candle_trend', 'candle_range', 'corps_candle', 
            'meche_haute', 'meche_basse', 'ratio_corps',
            'upper_wick', 'lower_wick',
            
            # Volatility and returns
            'hourly_return', 'hourly_volatility',
            'volatility_by_period', 'volatility_6h', 'volatility_12h',
            'volatility_period_0', 'volatility_period_1', 
            'volatility_period_2', 'volatility_period_3',
            
            # Log returns and momentum
            'log_return_5m', 'log_return_1h', 'log_return_4h',
            'momentum_1h', 'momentum_4h',
            
            # Multi-timeframe price features
            '1h_price_change_pct', '4h_price_change_pct', '1d_price_change_pct',
            '1h_range', '1h_position', '4h_range', '4h_position',
            '1d_range', '1d_position',
            
            # Multi-timeframe volume features
            '1h_volume_ratio', '4h_volume_ratio', '1d_volume_ratio',
            
            # Multi-timeframe trend features
            'close_over_1h_SMA', 'close_over_4h_SMA', 'close_over_1d_SMA',
            '1h_trend', '4h_trend', '1d_trend',
            'bullish_alignment', 'bearish_alignment', 'mixed_trend_signals',
            
            # Target variable
            'future_direction_2'
        ]
        
        # Feature categories for analysis
        self.feature_categories = {
            'Time Features': [
                'minute', 'day', 'hour', 'day_of_week', 'day_of_year',
                'sin_day', 'cos_day', 'sin_hour', 'cos_hour',
                'market_open_hour', 'stock_open_hour', 'is_summer', 'period_of_day'
            ],
            'Basic OHLCV': [
                'Open', 'High', 'Low', 'Close', 'Volume'
            ],
            'Volume Indicators': [
                'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20',
                'Volume_Ratio_SMA5', 'Volume_Ratio_SMA10', 'Volume_Ratio_SMA20',
                'Volume_Change_1', 'Volume_Change_5', 'PV_Ratio', 'PV_Change',
                'OBV', 'ADL', 'MFM', 'CMF_20', 'PVT', 'Volume_Oscillator',
                'VWAP_10', 'VWMA_10', 'VWMA_20',
                'Vol_Weighted_Up', 'Vol_Weighted_Down', 'Vol_Weighted_Down_Avg'
            ],
            'RSI Family': [
                'RSI_14', 'RSI_SMA_7', 'RSI_Trend', 'RSI', 'SMA_RSI',
                'Vol_Weighted_RSI', 'Vol_Weighted_RSI_SMA', 'Stoch_RSI'
            ],
            'Technical Indicators': [
                'MACD', 'MACD_Signal', 'ATR_14', 'ADX', 'Williams_R',
                'SMA_10', 'EMA_10'
            ],
            'Bollinger Bands': [
                'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width'
            ],
            'Keltner Channels': [
                'Keltner_High', 'Keltner_Low', 'Keltner_Width'
            ],
            'SuperTrend': [
                'SuperTrend_Trend', 'SuperTrend_Long', 'SuperTrend_Short'
            ],
            'CCI Indicators': [
                'CCI_5', 'CCI_10', 'CCI_20', 'CCI_40', 'CCI_80'
            ],
            'Force Index & Money Flow': [
                'Force_Index_1', 'Force_Index_13',
                'Typical_Price', 'Raw_Money_Flow', 'Typical_Price_Prev',
                'Money_Flow_Positive'
            ],
            'Candlestick Features': [
                'candle_trend', 'candle_range', 'corps_candle',
                'meche_haute', 'meche_basse', 'ratio_corps',
                'upper_wick', 'lower_wick'
            ],
            'Volatility & Returns': [
                'hourly_return', 'hourly_volatility',
                'volatility_by_period', 'volatility_6h', 'volatility_12h',
                'volatility_period_0', 'volatility_period_1',
                'volatility_period_2', 'volatility_period_3',
                'log_return_5m', 'log_return_1h', 'log_return_4h',
                'momentum_1h', 'momentum_4h'
            ],
            'Multi-timeframe Price': [
                '1h_price_change_pct', '4h_price_change_pct', '1d_price_change_pct',
                '1h_range', '1h_position', '4h_range', '4h_position',
                '1d_range', '1d_position'
            ],
            'Multi-timeframe Volume': [
                '1h_volume_ratio', '4h_volume_ratio', '1d_volume_ratio'
            ],
            'Multi-timeframe Trend': [
                'close_over_1h_SMA', 'close_over_4h_SMA', 'close_over_1d_SMA',
                '1h_trend', '4h_trend', '1d_trend',
                'bullish_alignment', 'bearish_alignment', 'mixed_trend_signals'
            ],
            'Target Variable': [
                'future_direction_2'
            ]
        }
        
    def log_issue(self, category, feature, issue, severity='WARNING'):
        """Log an issue found during verification"""
        self.issues.append({
            'category': category,
            'feature': feature,
            'issue': issue,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[{severity}] {category} - {feature}: {issue}")
    
    def load_sample_data(self, max_files=5):
        """Load sample data for analysis"""
        try:
            data_dir = os.path.join(self.base_directory, '5min')
            if not os.path.exists(data_dir):
                self.log_issue('DATA_LOADING', 'DIRECTORY',
                              f"Data directory not found: {data_dir}", 'ERROR')
                return None
            
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if not json_files:
                self.log_issue('DATA_LOADING', 'FILES',
                              f"No JSON files found in {data_dir}", 'ERROR')
                return None
            
            # Load a subset of files for testing
            files_to_load = json_files[:max_files]
            print(f"Loading {len(files_to_load)} files for analysis...")
            
            all_data = []
            for file in files_to_load:
                file_path = os.path.join(data_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict):
                            all_data.append(data)
                except Exception as e:
                    self.log_issue('DATA_LOADING', file,
                                  f"Error loading file: {e}", 'WARNING')
            
            if not all_data:
                self.log_issue('DATA_LOADING', 'CONVERSION',
                              "No valid data loaded from files", 'ERROR')
                return None
            
            df = pd.DataFrame(all_data)
            print(f"Loaded {len(df)} rows of raw data")
            
            # Basic preprocessing
            if 'FromDate' in df.columns:
                df['FromDate'] = pd.to_datetime(df['FromDate'])
            
            return df
            
        except Exception as e:
            self.log_issue('DATA_LOADING', 'GENERAL',
                          f"Error in load_sample_data: {e}", 'ERROR')
            return None
    
    def analyze_raw_data_structure(self, df):
        """Analyze the structure of raw data"""
        print(f"\n📊 RAW DATA ANALYSIS:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        print(f"   • Data types:")
        for col in df.columns:
            print(f"     - {col}: {df[col].dtype}")
        
        # Check for basic OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_basic = [col for col in required_cols if col not in df.columns]
        if missing_basic:
            self.log_issue('MISSING_BASIC_DATA', 'OHLCV',
                          f"Missing basic OHLCV columns: {missing_basic}", 'ERROR')
        
        # Check data quality of basic columns
        if all(col in df.columns for col in required_cols):
            for col in required_cols:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    self.log_issue('DATA_QUALITY', col,
                                  f"Found {null_count} null values in {col}", 'WARNING')
                
                if col != 'Volume':  # Volume can be zero
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        self.log_issue('DATA_QUALITY', col,
                                      f"Found {zero_count} zero values in {col}", 'WARNING')
        
        return df
    
    def calculate_basic_features(self, df):
        """Calculate basic features for analysis"""
        try:
            print("\n🔧 CALCULATING BASIC FEATURES...")
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                self.log_issue('FEATURE_CALCULATION', 'BASIC',
                              "Cannot calculate features - missing OHLCV data", 'ERROR')
                return df
            
            # Time features
            if 'FromDate' in df.columns:
                df['hour'] = df['FromDate'].dt.hour
                df['minute'] = df['FromDate'].dt.minute
                df['day'] = df['FromDate'].dt.day
                df['day_of_week'] = df['FromDate'].dt.dayofweek
                df['day_of_year'] = df['FromDate'].dt.dayofyear
                
                # Cyclical encoding
                df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
                df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
                df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Basic price features
            df['hourly_return'] = df['Close'].pct_change()
            df['log_return_5m'] = np.log(df['Close'] / df['Close'].shift(1))
            df['momentum_5m'] = df['Close'] - df['Close'].shift(1)
            
            # Candlestick features
            df['candle_range'] = df['High'] - df['Low']
            df['corps_candle'] = abs(df['Close'] - df['Open'])
            df['upper_wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['lower_wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['ratio_corps'] = df['corps_candle'] / df['candle_range'].replace(0, np.nan)
            
            # Trend direction
            df['candle_trend'] = (df['Close'] > df['Open']).astype(int)
            
            # Volume features
            df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            
            # Volume ratios
            df['Volume_Ratio_SMA5'] = df['Volume'] / df['Volume_SMA_5']
            df['Volume_Ratio_SMA10'] = df['Volume'] / df['Volume_SMA_10']
            
            # Simple RSI calculation
            df['RSI_14'] = self.calculate_simple_rsi(df['Close'], 14)
            df['RSI_SMA_7'] = df['RSI_14'].rolling(window=7).mean()
            
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['EMA_10'] = df['Close'].ewm(span=10).mean()
            
            # Volatility
            df['hourly_volatility'] = df['hourly_return'].rolling(window=24).std()
            
            # Target variable (simple future direction)
            df['future_direction_2'] = (df['Close'].shift(-2) > df['Close']).astype(int)
            
            print(f"   ✓ Calculated basic features. Dataset now has {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log_issue('FEATURE_CALCULATION', 'BASIC',
                          f"Error calculating basic features: {e}", 'ERROR')
            return df
    
    def calculate_simple_rsi(self, prices, window=14):
        """Simple RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_feature_quality(self, df):
        """Analyze the quality of calculated features"""
        print(f"\n🔍 FEATURE QUALITY ANALYSIS:")
        
        for col in df.columns:
            if col in ['FromDate', 'timestamp']:
                continue
                
            # Calculate statistics
            values = df[col]
            stats = {
                'count': len(values),
                'null_count': values.isna().sum(),
                'null_percentage': (values.isna().sum() / len(values)) * 100,
                'unique_count': values.nunique(),
                'dtype': str(values.dtype)
            }
            
            # Check for infinite values
            if values.dtype in ['float64', 'int64']:
                stats['infinite_count'] = np.isinf(values).sum()
                
                non_null_values = values.dropna()
                if len(non_null_values) > 0:
                    stats.update({
                        'mean': non_null_values.mean(),
                        'std': non_null_values.std(),
                        'min': non_null_values.min(),
                        'max': non_null_values.max(),
                        'median': non_null_values.median(),
                    })
                    
                    # Check for suspicious values
                    if stats['infinite_count'] > 0:
                        self.log_issue('INFINITE_VALUES', col,
                                      f"Found {stats['infinite_count']} infinite values", 'ERROR')
                    
                    if stats['null_percentage'] > 50:
                        self.log_issue('HIGH_NULL_VALUES', col,
                                      f"High null percentage: {stats['null_percentage']:.2f}%", 'ERROR')
                    elif stats['null_percentage'] > 10:
                        self.log_issue('MODERATE_NULL_VALUES', col,
                                      f"Moderate null percentage: {stats['null_percentage']:.2f}%", 'WARNING')
            
            self.feature_stats[col] = stats
        
        # Check feature coverage
        self.check_feature_coverage(df)
    
    def check_feature_coverage(self, df):
        """Check which expected features are present"""
        present_features = set(df.columns)
        expected_features = set(self.expected_features)
        
        missing_features = expected_features - present_features
        unexpected_features = present_features - expected_features - {'FromDate', 'timestamp'}
        
        if missing_features:
            self.log_issue('MISSING_FEATURES', 'COVERAGE',
                          f"Missing {len(missing_features)} expected features", 'WARNING')
        
        if unexpected_features:
            print(f"   ℹ️  Found {len(unexpected_features)} unexpected features")
        
        # Analyze by category
        print(f"\n📋 FEATURE CATEGORY COVERAGE:")
        for category, features in self.feature_categories.items():
            present_count = sum(1 for f in features if f in present_features)
            total_count = len(features)
            percentage = (present_count / total_count) * 100
            
            status = "✅" if percentage == 100 else "⚠️" if percentage > 50 else "❌"
            print(f"   {status} {category}: {present_count}/{total_count} ({percentage:.1f}%)")
            
            if percentage < 100:
                missing_in_category = [f for f in features if f not in present_features]
                print(f"      Missing: {missing_in_category[:3]}{'...' if len(missing_in_category) > 3 else ''}")
    
    def check_value_ranges(self, df):
        """Check if feature values are within expected ranges"""
        print(f"\n📏 VALUE RANGE VALIDATION:")
        
        expected_ranges = {
            'RSI_14': (0, 100),
            'sin_day': (-1, 1),
            'cos_day': (-1, 1),
            'sin_hour': (-1, 1),
            'cos_hour': (-1, 1),
            'hour': (0, 23),
            'minute': (0, 59),
            'day_of_week': (0, 6),
            'candle_trend': (0, 1),
            'future_direction_2': (0, 1),
            'ratio_corps': (0, 1)
        }
        
        range_issues = 0
        for feature, (expected_min, expected_max) in expected_ranges.items():
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    actual_min = values.min()
                    actual_max = values.max()
                    
                    if actual_min < expected_min or actual_max > expected_max:
                        self.log_issue('VALUE_RANGE', feature,
                                      f"Outside expected range [{expected_min}, {expected_max}]: "
                                      f"actual [{actual_min:.4f}, {actual_max:.4f}]", 'WARNING')
                        range_issues += 1
        
        if range_issues == 0:
            print("   ✅ All checked features are within expected ranges")
        else:
            print(f"   ⚠️  {range_issues} features have values outside expected ranges")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FEATURE ANALYSIS REPORT")
        print("="*80)
        
        # Dataset overview
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total rows: {len(df):,}")
        print(f"   • Total columns: {len(df.columns):,}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        if 'FromDate' in df.columns:
            print(f"   • Date range: {df['FromDate'].min()} to {df['FromDate'].max()}")
        
        # Issues summary
        print(f"\n⚠️  ISSUES SUMMARY:")
        if self.issues:
            issue_counts = {}
            for issue in self.issues:
                severity = issue['severity']
                issue_counts[severity] = issue_counts.get(severity, 0) + 1
            
            for severity in ['ERROR', 'WARNING', 'INFO']:
                if severity in issue_counts:
                    print(f"   • {severity}: {issue_counts[severity]}")
        else:
            print("   ✅ No issues detected!")
        
        # Feature statistics
        print(f"\n📈 FEATURE STATISTICS:")
        numeric_features = [f for f, stats in self.feature_stats.items() 
                          if stats.get('dtype') in ['float64', 'int64']]
        high_null_features = [f for f, stats in self.feature_stats.items() 
                            if stats.get('null_percentage', 0) > 10]
        infinite_features = [f for f, stats in self.feature_stats.items() 
                           if stats.get('infinite_count', 0) > 0]
        
        print(f"   • Numeric features: {len(numeric_features)}")
        print(f"   • Features with >10% nulls: {len(high_null_features)}")
        print(f"   • Features with infinite values: {len(infinite_features)}")
        
        # Expected vs actual features
        present_count = sum(1 for f in self.expected_features if f in df.columns)
        print(f"   • Expected features present: {present_count}/{len(self.expected_features)} ({present_count/len(self.expected_features)*100:.1f}%)")
        
        # Category-wise summary
        print(f"\n📋 CATEGORY SUMMARY:")
        for category, features in self.feature_categories.items():
            present = sum(1 for f in features if f in df.columns)
            total = len(features)
            print(f"   • {category}: {present}/{total} ({present/total*100:.0f}%)")
        
        # Save detailed report
        self.save_report()
        
        print(f"\n💾 Detailed report saved to: feature_analysis_report.json")
        print("="*80)
    
    def save_report(self):
        """Save detailed analysis report"""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_expected_features': len(self.expected_features),
                'total_issues': len(self.issues),
                'issue_breakdown': {}
            },
            'feature_categories': self.feature_categories,
            'expected_features': self.expected_features,
            'feature_statistics': self.feature_stats,
            'issues': self.issues
        }
        
        # Calculate issue breakdown
        for issue in self.issues:
            severity = issue['severity']
            report['summary']['issue_breakdown'][severity] = \
                report['summary']['issue_breakdown'].get(severity, 0) + 1
        
        with open('feature_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def run_analysis(self):
        """Run the complete feature analysis"""
        print("="*80)
        print("TRADING PREDICTION MODEL - FEATURE ANALYSIS")
        print("="*80)
        
        # Load data
        print("\n1️⃣  LOADING SAMPLE DATA...")
        df = self.load_sample_data()
        if df is None:
            print("❌ Failed to load data. Analysis terminated.")
            return None
        
        # Analyze raw data structure
        print("\n2️⃣  ANALYZING RAW DATA STRUCTURE...")
        df = self.analyze_raw_data_structure(df)
        
        # Calculate basic features
        print("\n3️⃣  CALCULATING FEATURES...")
        df = self.calculate_basic_features(df)
        
        # Analyze feature quality
        print("\n4️⃣  ANALYZING FEATURE QUALITY...")
        self.analyze_feature_quality(df)
        
        # Check value ranges
        print("\n5️⃣  VALIDATING VALUE RANGES...")
        self.check_value_ranges(df)
        
        # Generate report
        print("\n6️⃣  GENERATING REPORT...")
        self.generate_summary_report(df)
        
        print("\n🎯 ANALYSIS COMPLETE!")
        return df

def main():
    """Main execution function"""
    analyzer = FeatureAnalyzer()
    df = analyzer.run_analysis()
    
    if df is not None:
        print(f"\n✅ Analysis completed successfully!")
        print(f"   • Dataset shape: {df.shape}")
        print(f"   • Issues found: {len(analyzer.issues)}")
        print(f"   • Report saved: feature_analysis_report.json")
        
        return df, analyzer.issues, analyzer.feature_stats
    else:
        print("\n❌ Analysis failed!")
        return None, analyzer.issues, analyzer.feature_stats

if __name__ == "__main__":
    df, issues, stats = main()