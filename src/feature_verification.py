#!/usr/bin/env python3
"""
Comprehensive Feature Verification Script for Trading Prediction Model

This script analyzes all features used in the trading prediction model and validates:
- Correct calculation logic
- NaN/infinite values
- Data quality issues  
- Expected value ranges
- Feature categories coverage

Feature Categories:
1. Basic OHLCV features and transformations
2. Technical indicators (RSI, MACD, Bollinger Bands, SuperTrend, etc.)
3. Candlestick patterns
4. Multi-timeframe features
5. Derived/calculated features
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path to import model_trainer functions
sys.path.append('/home/parrochia/go/src/github.com/trading-prediction-with-machine-learning/src')

try:
    from model_trainer import (
        load_json_files_from_directory,
        preprocess_features,
        add_time_columns,
        calculate_rsi,
        volume_weighted_rsi_sma,
        add_future_direction,
        add_direction,
        add_candle_features,
        add_doji,
        add_candle_trend_relation,
        add_engulfing,
        add_wick_features,
        add_body_ratio,
        calculate_macd,
        add_volume_indicators,
        add_vwap_10,
        get_market_opening,
        get_period_of_day_with_timezone,
        add_multi_timeframe_features_corrected,
        compute_cci
    )
    print("✓ Successfully imported model_trainer functions")
except ImportError as e:
    print(f"✗ Error importing model_trainer functions: {e}")
    sys.exit(1)

# Import technical analysis libraries
try:
    import ta
    import pandas_ta as pta
    print("✓ Technical analysis libraries loaded")
except ImportError as e:
    print(f"✗ Error importing technical analysis libraries: {e}")
    sys.exit(1)

class FeatureVerifier:
    def __init__(self, base_directory='/home/parrochia/go/src/github.com/trading-prediction-with-machine-learning/brent'):
        self.base_directory = base_directory
        self.issues = []
        self.feature_stats = {}
        self.expected_features = [
            # Time features
            'day', 'hour', 'minute', 'day_of_week', 'day_of_year',
            'sin_day', 'cos_day', 'sin_hour', 'cos_hour',
            'market_open_hour', 'stock_open_hour', 'is_summer',
            'period_of_day',
            
            # Basic OHLCV
            'Open', 'High', 'Low', 'Close', 'Volume',
            
            # Volume indicators
            'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20',
            'Volume_Ratio_SMA5', 'Volume_Ratio_SMA10', 'Volume_Ratio_SMA20',
            'Volume_Change_1', 'Volume_Change_5',
            'PV_Ratio', 'PV_Change', 'OBV', 'ADL', 'MFM', 'CMF_20', 'PVT',
            'Volume_Oscillator', 'VWAP_10', 'VWMA_10', 'VWMA_20',
            
            # Technical indicators
            'RSI_14', 'RSI_SMA_7', 'RSI_Trend', 'Stoch_RSI',
            'MACD', 'MACD_Signal', 'ATR_14',
            'SMA_10', 'EMA_10',
            'ADX', 'Williams_R',
            
            # Bollinger Bands
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            
            # Keltner Channels
            'Keltner_High', 'Keltner_Low', 'Keltner_Width',
            
            # SuperTrend
            'SuperTrend_Trend', 'SuperTrend_Long', 'SuperTrend_Short',
            
            # CCI indicators
            'CCI_5', 'CCI_10', 'CCI_20', 'CCI_40', 'CCI_80',
            
            # Candlestick features
            'candle_trend', 'candle_range', 'corps_candle', 
            'meche_haute', 'meche_basse', 'ratio_corps',
            'upper_wick', 'lower_wick',
            
            # Doji patterns
            'doji', 'doji_type', 'doji_strength', 'perfect_doji',
            
            # Engulfing patterns
            'bullish_engulfing', 'bearish_engulfing', 'engulfing_strength',
            
            # Volatility and returns
            'hourly_return', 'hourly_volatility',
            'volatility_by_period', 'volatility_6h', 'volatility_12h',
            'volatility_period_0', 'volatility_period_1', 
            'volatility_period_2', 'volatility_period_3',
            
            # Multi-timeframe returns and momentum
            'log_return_5m', 'log_return_1h', 'log_return_4h',
            'momentum_5m', 'momentum_1h', 'momentum_4h',
            
            # Volume weighted features
            'Vol_Weighted_Up', 'Vol_Weighted_Down', 'Vol_Weighted_Down_Avg',
            'Vol_Weighted_RSI', 'Vol_Weighted_RSI_SMA',
            
            # Force Index
            'Force_Index_1', 'Force_Index_13',
            
            # Money Flow
            'Typical_Price', 'Raw_Money_Flow', 'Typical_Price_Prev',
            'Money_Flow_Positive',
            
            # Multi-timeframe features
            '1h_price_change_pct', '4h_price_change_pct', '1d_price_change_pct',
            '1h_range', '1h_position', '4h_range', '4h_position',
            '1d_range', '1d_position',
            '1h_volume_ratio', '4h_volume_ratio', '1d_volume_ratio',
            'close_over_1h_SMA', 'close_over_4h_SMA', 'close_over_1d_SMA',
            '1h_trend', '4h_trend', '1d_trend',
            'bullish_alignment', 'bearish_alignment', 'mixed_trend_signals',
            
            # Target variable
            'future_direction_2'
        ]
        
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
    
    def check_for_infinite_values(self, df, feature):
        """Check for infinite values in a feature"""
        if feature not in df.columns:
            return
            
        inf_count = np.isinf(df[feature]).sum()
        if inf_count > 0:
            self.log_issue('INFINITE_VALUES', feature, 
                          f"Found {inf_count} infinite values", 'ERROR')
    
    def check_for_nan_values(self, df, feature):
        """Check for NaN values in a feature"""
        if feature not in df.columns:
            return
            
        nan_count = df[feature].isna().sum()
        nan_percentage = (nan_count / len(df)) * 100
        
        if nan_percentage > 50:
            self.log_issue('HIGH_NAN_VALUES', feature,
                          f"High NaN percentage: {nan_percentage:.2f}%", 'ERROR')
        elif nan_percentage > 10:
            self.log_issue('MODERATE_NAN_VALUES', feature,
                          f"Moderate NaN percentage: {nan_percentage:.2f}%", 'WARNING')
    
    def check_value_ranges(self, df, feature):
        """Check if feature values are within expected ranges"""
        if feature not in df.columns:
            return
            
        values = df[feature].dropna()
        if len(values) == 0:
            return
            
        min_val = values.min()
        max_val = values.max()
        
        # Define expected ranges for specific features
        expected_ranges = {
            'RSI_14': (0, 100),
            'Stoch_RSI': (0, 1),
            'Williams_R': (-100, 0),
            'sin_day': (-1, 1),
            'cos_day': (-1, 1),
            'sin_hour': (-1, 1),
            'cos_hour': (-1, 1),
            'hour': (0, 23),
            'minute': (0, 59),
            'day_of_week': (0, 6),
            'period_of_day': (0, 3),
            'is_summer': (0, 1),
            'bullish_engulfing': (0, 1),
            'bearish_engulfing': (0, 1),
            'doji': (0, 1),
            'perfect_doji': (0, 1)
        }
        
        if feature in expected_ranges:
            expected_min, expected_max = expected_ranges[feature]
            if min_val < expected_min or max_val > expected_max:
                self.log_issue('VALUE_RANGE', feature,
                              f"Values outside expected range [{expected_min}, {expected_max}]: "
                              f"actual range [{min_val:.4f}, {max_val:.4f}]", 'WARNING')
    
    def calculate_feature_statistics(self, df):
        """Calculate comprehensive statistics for all features"""
        for feature in df.columns:
            if feature in ['FromDate', 'date', 'prev_date', 'timestamp']:
                continue
                
            values = df[feature]
            stats = {
                'count': len(values),
                'null_count': values.isna().sum(),
                'null_percentage': (values.isna().sum() / len(values)) * 100,
                'infinite_count': np.isinf(values).sum() if values.dtype in ['float64', 'int64'] else 0,
                'unique_count': values.nunique(),
                'dtype': str(values.dtype)
            }
            
            if values.dtype in ['float64', 'int64']:
                non_null_values = values.dropna()
                if len(non_null_values) > 0:
                    stats.update({
                        'mean': non_null_values.mean(),
                        'std': non_null_values.std(),
                        'min': non_null_values.min(),
                        'max': non_null_values.max(),
                        'median': non_null_values.median(),
                        'q25': non_null_values.quantile(0.25),
                        'q75': non_null_values.quantile(0.75)
                    })
            
            self.feature_stats[feature] = stats
    
    def verify_feature_calculations(self, df):
        """Verify specific feature calculation logic"""
        
        # Verify RSI calculation
        if 'RSI_14' in df.columns and len(df) > 14:
            manual_rsi = self.calculate_manual_rsi(df['Close'], 14)
            if len(manual_rsi) > 0 and len(df['RSI_14'].dropna()) > 0:
                correlation = np.corrcoef(manual_rsi.dropna(), df['RSI_14'].dropna())[0, 1]
                if correlation < 0.95:
                    self.log_issue('CALCULATION_ERROR', 'RSI_14',
                                  f"RSI calculation may be incorrect (correlation: {correlation:.4f})", 'ERROR')
        
        # Verify volume ratios
        if all(col in df.columns for col in ['Volume', 'Volume_SMA_5']):
            calculated_ratio = df['Volume'] / df['Volume_SMA_5']
            if 'Volume_Ratio_SMA5' in df.columns:
                if not np.allclose(calculated_ratio.dropna(), df['Volume_Ratio_SMA5'].dropna(), 
                                 rtol=1e-05, atol=1e-08, equal_nan=True):
                    self.log_issue('CALCULATION_ERROR', 'Volume_Ratio_SMA5',
                                  "Volume ratio calculation inconsistency", 'WARNING')
        
        # Verify candlestick calculations
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Check candle_range
            expected_range = df['High'] - df['Low']
            if 'candle_range' in df.columns:
                if not np.allclose(expected_range.dropna(), df['candle_range'].dropna(),
                                 rtol=1e-05, atol=1e-08, equal_nan=True):
                    self.log_issue('CALCULATION_ERROR', 'candle_range',
                                  "Candle range calculation inconsistency", 'WARNING')
            
            # Check corps_candle (body size)
            expected_body = abs(df['Close'] - df['Open'])
            if 'corps_candle' in df.columns:
                if not np.allclose(expected_body.dropna(), df['corps_candle'].dropna(),
                                 rtol=1e-05, atol=1e-08, equal_nan=True):
                    self.log_issue('CALCULATION_ERROR', 'corps_candle',
                                  "Candle body calculation inconsistency", 'WARNING')
        
        # Verify log returns
        if 'log_return_5m' in df.columns and 'Close' in df.columns:
            expected_log_return = np.log(df['Close'] / df['Close'].shift(1))
            if not np.allclose(expected_log_return.dropna(), df['log_return_5m'].dropna(),
                             rtol=1e-05, atol=1e-08, equal_nan=True):
                self.log_issue('CALCULATION_ERROR', 'log_return_5m',
                              "Log return calculation inconsistency", 'WARNING')
    
    def calculate_manual_rsi(self, prices, window=14):
        """Manual RSI calculation for verification"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_feature_coverage(self, df):
        """Check if all expected features are present"""
        missing_features = []
        present_features = list(df.columns)
        
        for expected_feature in self.expected_features:
            if expected_feature not in present_features:
                missing_features.append(expected_feature)
        
        if missing_features:
            self.log_issue('MISSING_FEATURES', 'COVERAGE',
                          f"Missing features: {missing_features}", 'WARNING')
        
        # Check for unexpected features
        unexpected_features = [f for f in present_features 
                             if f not in self.expected_features and 
                             f not in ['FromDate', 'date', 'prev_date', 'timestamp']]
        
        if unexpected_features:
            self.log_issue('UNEXPECTED_FEATURES', 'COVERAGE',
                          f"Unexpected features found: {unexpected_features}", 'INFO')
    
    def verify_multi_timeframe_consistency(self, df):
        """Verify multi-timeframe feature consistency"""
        multi_tf_features = [
            ('1h_price_change_pct', '4h_price_change_pct', '1d_price_change_pct'),
            ('1h_volume_ratio', '4h_volume_ratio', '1d_volume_ratio'),
            ('1h_trend', '4h_trend', '1d_trend')
        ]
        
        for feature_group in multi_tf_features:
            for feature in feature_group:
                if feature in df.columns:
                    # Check for logical consistency (e.g., longer timeframes should be smoother)
                    values = df[feature].dropna()
                    if len(values) > 1:
                        volatility = values.std()
                        # Store for comparison across timeframes
                        setattr(self, f"{feature}_volatility", volatility)
    
    def load_sample_data(self, max_files=10):
        """Load sample data for testing"""
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
            print(f"Loading {len(files_to_load)} files for verification...")
            
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
            return df
            
        except Exception as e:
            self.log_issue('DATA_LOADING', 'GENERAL',
                          f"Error in load_sample_data: {e}", 'ERROR')
            return None
    
    def process_features(self, df):
        """Process all features using the model_trainer pipeline"""
        try:
            print("Starting feature processing pipeline...")
            
            # Preprocess basic features
            df = preprocess_features(df)
            print(f"After preprocessing: {len(df)} rows")
            
            # Load mock multi-timeframe data for testing
            df_1h = self.create_mock_timeframe_data(df, '1h')
            df_4h = self.create_mock_timeframe_data(df, '4h') 
            df_1d = self.create_mock_timeframe_data(df, '1d')
            
            # Add volume indicators
            df = add_volume_indicators(df)
            
            # Add time columns
            df = add_time_columns(df)
            
            # Market opening features
            df[['market_open_hour', 'stock_open_hour', 'is_summer']] = df['FromDate'].apply(
                lambda x: pd.Series(get_market_opening(x))
            )
            
            # RSI calculations
            df['RSI_14'] = calculate_rsi(df)
            df['RSI_SMA_7'] = df['RSI_14'].rolling(window=7).mean()
            df['RSI_SMA_14'] = df['RSI_14'].rolling(window=14).mean()
            
            # Volume weighted RSI
            df = volume_weighted_rsi_sma(df)
            
            # RSI trend features
            df['RSI_EMA_7'] = df['RSI_14'].ewm(span=7, adjust=False).mean()
            df['RSI_EMA_14'] = df['RSI_14'].ewm(span=14, adjust=False).mean()
            df['RSI_Trend'] = df['RSI_14'].diff()
            df['RSI_Trend_Direction'] = df['RSI_Trend'].apply(lambda x: 1 if x > 0 else 0)
            df['RSI_Crossover_SMA'] = (df['RSI_14'] > df['RSI_SMA_7']).astype(int)
            df['RSI_Crossover_EMA'] = (df['RSI_14'] > df['RSI_EMA_7']).astype(int)
            
            # Direction and candle features
            df = add_future_direction(df)
            df = add_direction(df)
            df = add_candle_features(df)
            df = add_doji(df)
            df = add_candle_trend_relation(df)
            df = add_engulfing(df)
            df = add_wick_features(df)
            df = add_body_ratio(df)
            
            # MACD
            df = calculate_macd(df)
            
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            
            # Technical indicators using ta library
            if len(df) > 14:
                try:
                    df['ATR_14'] = ta.volatility.AverageTrueRange(
                        high=df['High'], low=df['Low'], close=df['Close'], window=14
                    ).average_true_range()
                    
                    df['MACD'] = ta.trend.MACD(
                        close=df['Close'], window_slow=26, window_fast=12, window_sign=9
                    ).macd()
                    
                    # Bollinger Bands
                    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                    df['Bollinger_High'] = bb.bollinger_hband()
                    df['Bollinger_Low'] = bb.bollinger_lband()
                    df['Bollinger_Width'] = bb.bollinger_wband()
                    
                    # SuperTrend
                    if len(df) > 20:
                        supertrend_result = pta.overlap.supertrend(
                            high=df['High'], low=df['Low'], close=df['Close'],
                            length=10, multiplier=3
                        )
                        df['SuperTrend_Trend'] = supertrend_result['SUPERT_10_3.0']
                        df['SuperTrend_Direction'] = supertrend_result['SUPERTd_10_3.0']
                        df['SuperTrend_Long'] = supertrend_result['SUPERTl_10_3.0']
                        df['SuperTrend_Short'] = supertrend_result['SUPERTs_10_3.0']
                    
                    # Stochastic RSI
                    df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(
                        close=df['Close'], window=14, smooth1=3, smooth2=3
                    ).stochrsi()
                    
                    # Keltner Channels
                    kc = ta.volatility.KeltnerChannel(
                        high=df['High'], low=df['Low'], close=df['Close'], window=20
                    )
                    df['Keltner_High'] = kc.keltner_channel_hband()
                    df['Keltner_Low'] = kc.keltner_channel_lband()
                    df['Keltner_Width'] = kc.keltner_channel_wband()
                    
                    # ADX
                    df['ADX'] = ta.trend.adx(
                        high=df['High'], low=df['Low'], close=df['Close'], window=14
                    )
                    
                    # Williams %R
                    df['Williams_R'] = ta.momentum.williams_r(
                        high=df['High'], low=df['Low'], close=df['Close'], lbp=14
                    )
                    
                except Exception as e:
                    self.log_issue('FEATURE_PROCESSING', 'TECHNICAL_INDICATORS',
                                  f"Error calculating technical indicators: {e}", 'ERROR')
            
            # CCI indicators
            for period in [5, 10, 20, 40, 80]:
                df[f'CCI_{period}'] = compute_cci(df, period)
            
            # Cyclical features
            df['day_of_year'] = df['FromDate'].dt.dayofyear
            df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Period of day
            df['period_of_day'] = df['FromDate'].apply(get_period_of_day_with_timezone)
            
            # Returns and volatility
            df['hourly_return'] = df['Close'].pct_change()
            df['hourly_volatility'] = df['hourly_return'].rolling(window=24).std()
            
            # Volatility by period
            volatility_by_period = df.groupby('period_of_day')['hourly_return'].std().reset_index()
            volatility_by_period.rename(columns={'hourly_return': 'volatility_by_period'}, inplace=True)
            df = df.merge(volatility_by_period, on='period_of_day', how='left')
            
            # Multi-window volatilities
            df['volatility_6h'] = df['hourly_return'].rolling(window=6).std()
            df['volatility_12h'] = df['hourly_return'].rolling(window=12).std()
            
            # Period-specific volatilities
            for period in range(4):
                period_data = df[df['period_of_day'] == period]['hourly_return']
                df.loc[df['period_of_day'] == period, f'volatility_period_{period}'] = \
                    period_data.rolling(window=6).std()
            
            # Log returns and momentum
            df['log_return_5m'] = np.log(df['Close'] / df['Close'].shift(1))
            df['momentum_5m'] = df['Close'] - df['Close'].shift(1)
            
            # Multi-timeframe features
            try:
                df = add_multi_timeframe_features_corrected(df, df_1h, df_4h, df_1d)
            except Exception as e:
                self.log_issue('FEATURE_PROCESSING', 'MULTI_TIMEFRAME',
                              f"Error adding multi-timeframe features: {e}", 'WARNING')
            
            # VWAP
            df = add_vwap_10(df)
            
            # Fill NaN values
            df = df.fillna(0)
            
            print(f"Feature processing completed. Final dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log_issue('FEATURE_PROCESSING', 'GENERAL',
                          f"Error in process_features: {e}", 'ERROR')
            return df
    
    def create_mock_timeframe_data(self, df_5m, timeframe):
        """Create mock higher timeframe data for testing"""
        try:
            if timeframe == '1h':
                # Resample to hourly
                df_tf = df_5m.set_index('FromDate').resample('1H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == '4h':
                # Resample to 4-hourly
                df_tf = df_5m.set_index('FromDate').resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif timeframe == '1d':
                # Resample to daily
                df_tf = df_5m.set_index('FromDate').resample('1D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                return pd.DataFrame()
            
            df_tf.reset_index(inplace=True)
            return df_tf
            
        except Exception as e:
            self.log_issue('DATA_PROCESSING', f'MOCK_{timeframe.upper()}',
                          f"Error creating mock {timeframe} data: {e}", 'WARNING')
            return pd.DataFrame()
    
    def run_comprehensive_verification(self):
        """Run the complete feature verification process"""
        print("="*80)
        print("COMPREHENSIVE FEATURE VERIFICATION")
        print("="*80)
        
        # Load sample data
        print("\n1. Loading sample data...")
        df = self.load_sample_data()
        if df is None:
            print("❌ Failed to load data. Exiting verification.")
            return
        
        # Process features
        print("\n2. Processing features...")
        df = self.process_features(df)
        if df is None:
            print("❌ Failed to process features. Exiting verification.")
            return
        
        print(f"\n3. Analyzing {len(df.columns)} features across {len(df)} rows...")
        
        # Calculate feature statistics
        self.calculate_feature_statistics(df)
        
        # Check feature coverage
        print("\n4. Checking feature coverage...")
        self.check_feature_coverage(df)
        
        # Verify feature calculations
        print("\n5. Verifying feature calculations...")
        self.verify_feature_calculations(df)
        
        # Check for data quality issues
        print("\n6. Checking data quality...")
        for feature in df.columns:
            if feature not in ['FromDate', 'date', 'prev_date', 'timestamp']:
                self.check_for_nan_values(df, feature)
                self.check_for_infinite_values(df, feature)
                self.check_value_ranges(df, feature)
        
        # Verify multi-timeframe consistency
        print("\n7. Verifying multi-timeframe consistency...")
        self.verify_multi_timeframe_consistency(df)
        
        # Generate report
        self.generate_report(df)
        
        return df
    
    def generate_report(self, df):
        """Generate comprehensive verification report"""
        print("\n" + "="*80)
        print("VERIFICATION REPORT")
        print("="*80)
        
        # Summary statistics
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   • Total rows: {len(df):,}")
        print(f"   • Total features: {len(df.columns):,}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Feature categories
        feature_categories = {
            'Time Features': ['day', 'hour', 'minute', 'day_of_week', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour'],
            'OHLCV Features': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'Technical Indicators': ['RSI_14', 'MACD', 'ATR_14', 'ADX', 'Bollinger_High', 'SuperTrend_Trend'],
            'Volume Indicators': ['Volume_SMA_5', 'OBV', 'ADL', 'CMF_20', 'VWAP_10'],
            'Candlestick Features': ['candle_trend', 'corps_candle', 'doji', 'bullish_engulfing'],
            'Multi-timeframe': ['log_return_1h', '1h_trend', '4h_position', '1d_volume_ratio']
        }
        
        print(f"\n📋 FEATURE CATEGORIES:")
        for category, features in feature_categories.items():
            present = sum(1 for f in features if f in df.columns)
            print(f"   • {category}: {present}/{len(features)} features present")
        
        # Issues summary
        print(f"\n⚠️  ISSUES FOUND: {len(self.issues)}")
        if self.issues:
            issue_counts = {}
            for issue in self.issues:
                severity = issue['severity']
                issue_counts[severity] = issue_counts.get(severity, 0) + 1
            
            for severity, count in sorted(issue_counts.items()):
                print(f"   • {severity}: {count}")
            
            # Top issues
            print(f"\n🔍 TOP ISSUES:")
            for i, issue in enumerate(self.issues[:10]):
                print(f"   {i+1}. [{issue['severity']}] {issue['category']} - {issue['feature']}: {issue['issue']}")
        else:
            print("   ✅ No issues found!")
        
        # Feature quality summary
        print(f"\n📈 DATA QUALITY SUMMARY:")
        high_nan_features = [f for f, stats in self.feature_stats.items() 
                           if stats.get('null_percentage', 0) > 10]
        infinite_features = [f for f, stats in self.feature_stats.items() 
                           if stats.get('infinite_count', 0) > 0]
        
        print(f"   • Features with >10% NaN values: {len(high_nan_features)}")
        print(f"   • Features with infinite values: {len(infinite_features)}")
        
        if high_nan_features:
            print(f"   • High NaN features: {high_nan_features[:5]}{'...' if len(high_nan_features) > 5 else ''}")
        
        if infinite_features:
            print(f"   • Infinite value features: {infinite_features[:5]}{'...' if len(infinite_features) > 5 else ''}")
        
        # Save detailed report
        self.save_detailed_report(df)
        
        print(f"\n💾 Detailed report saved to: feature_verification_report.json")
        print("\n" + "="*80)
    
    def save_detailed_report(self, df):
        """Save detailed verification report to JSON"""
        report = {
            'verification_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_rows': len(df),
                'total_features': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'date_range': {
                    'start': df['FromDate'].min().isoformat() if 'FromDate' in df.columns else None,
                    'end': df['FromDate'].max().isoformat() if 'FromDate' in df.columns else None
                }
            },
            'feature_statistics': self.feature_stats,
            'issues': self.issues,
            'feature_list': list(df.columns),
            'expected_features': self.expected_features,
            'missing_features': [f for f in self.expected_features if f not in df.columns],
            'unexpected_features': [f for f in df.columns 
                                  if f not in self.expected_features and 
                                  f not in ['FromDate', 'date', 'prev_date', 'timestamp']]
        }
        
        with open('feature_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

def main():
    """Main execution function"""
    verifier = FeatureVerifier()
    df = verifier.run_comprehensive_verification()
    
    print("\n🎯 VERIFICATION COMPLETE!")
    print("Check the detailed report in: feature_verification_report.json")
    
    if df is not None:
        return df, verifier.issues, verifier.feature_stats
    else:
        return None, verifier.issues, verifier.feature_stats

if __name__ == "__main__":
    df, issues, stats = main()