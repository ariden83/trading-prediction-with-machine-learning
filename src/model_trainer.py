import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
# from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
# from sklearn.dummy import DummyClassifier
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import ta
import pandas_ta as pta
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import pytz
# import fastparquet
from tensorflow.keras.initializers import HeNormal, GlorotUniform, LecunNormal

print("Parquet engines loaded successfully!")

plt.switch_backend('agg')
sns.set(style='whitegrid')

selectSeed = 581
os.environ['PYTHONHASHSEED'] = str(selectSeed)  # Fixe le hash seed de Python
random.seed(selectSeed)
np.random.seed(selectSeed)
tf.random.set_seed(selectSeed)

# Assurez-vous que TensorFlow fonctionne en mode déterministe (nécessaire pour certaines versions)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Force TensorFlow to use single thread for better determinism
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

selected_features_save = [
    'day', 'hour',
    'minute',
    'Open', 'High', 'Low', 'Close', "Volume",
    'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20',
    'Volume_Ratio_SMA5', 'Volume_Ratio_SMA10', 'Volume_Ratio_SMA20',
    'Volume_Change_1', 'Volume_Change_5',
    'SuperTrend_Trend',
    'SuperTrend_Long', 'SuperTrend_Short',
    'ATR_14',
    'MACD', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
    'ADX', 'Stoch_RSI', 'Williams_R',
    'Keltner_High', 'Keltner_Low', 'Keltner_Width',
    'candle_trend', 'candle_range', 'corps_candle', 'meche_haute', 'meche_basse', 'ratio_corps',
    'RSI_14', 'RSI_SMA_7',
    'market_open_hour', 'stock_open_hour', 'is_summer',
    'sin_day', 'cos_day', 'sin_hour', 'cos_hour',
    'period_of_day', 'hourly_return',
    'volatility_by_period', 'hourly_volatility',
    'volatility_6h', 'volatility_12h',
    'volatility_period_0', 'volatility_period_1', 'volatility_period_2', 'volatility_period_3',
    'day_of_week',
    'log_return_5m', 'log_return_1h', 'log_return_4h',
    'momentum_5m', 'momentum_1h', 'momentum_4h',
    'RSI_Trend', 'SMA_10', 'EMA_10',
    'upper_wick', 'lower_wick',
    #'VWAP',
    'VWMA_10',
    'VWMA_20',

    'PV_Ratio',
    'PV_Change',
    'OBV',
    'ADL',
    'MFM',
    'CMF_20',
    'PVT',
    'Volume_Oscillator',

    'Force_Index_1',
    'Force_Index_13',
    'Typical_Price',
    'Raw_Money_Flow',
    'Typical_Price_Prev',
    'Money_Flow_Positive',

    'CCI_5',
    'CCI_10',
    'CCI_20',
    'CCI_40',
    'CCI_80',
]

selected_features = [
    'minute', 'Open', 'Low', 'Close', 'Volume', 'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20', 'Volume_Ratio_SMA5',
    'Volume_Ratio_SMA10', 'Volume_Change_1', 'Volume_Change_5', 'SuperTrend_Trend', 'SuperTrend_Long', 'SuperTrend_Short',
    'MACD', 'Bollinger_High', 'Bollinger_Width', 'ADX', 'Stoch_RSI', 'Keltner_High', 'Keltner_Low', 'Keltner_Width',
    'candle_trend', 'candle_range', 'corps_candle', 'meche_haute', 'meche_basse', 'ratio_corps', 'RSI_14', 'RSI_SMA_7',
    'market_open_hour', 'stock_open_hour', 'is_summer', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour', 'period_of_day',
    'hourly_return', 'volatility_by_period', 'hourly_volatility', 'volatility_6h', 'volatility_12h', 'volatility_period_0',
    'volatility_period_1', 'volatility_period_2', 'volatility_period_3', 'day_of_week', 'log_return_5m', 'log_return_1h',
    'log_return_4h', 'momentum_1h', 'momentum_4h', 'Vol_Weighted_Up', 'Vol_Weighted_Down', 'Vol_Weighted_Down_Avg',
    'Vol_Weighted_RSI', 'Vol_Weighted_RSI_SMA', 'CCI_5', 'CCI_10', 'CCI_20', 'CCI_40', 'CCI_80', 'RSI', 'SMA_RSI', 'ATR_14',
    'MACD_Signal', 'MACD', 'RSI_Trend', 'EMA_10', 'upper_wick', 'lower_wick', 'VWMA_10', 'VWMA_20', 'PV_Ratio', 'PV_Change',
    'OBV',
    'ADL', 'MFM', 'CMF_20', 'PVT', 'Volume_Oscillator', 'Force_Index_1', 'Force_Index_13', 'Typical_Price', 'Raw_Money_Flow',
    'VWAP_10',
    '1h_price_change_pct',
    '4h_price_change_pct',
    '1d_price_change_pct',
    '1h_range',
    '1h_position',
    '4h_range',
    '4h_position',
    '1d_range',
    '1d_position',

    '1h_volume_ratio',
    '4h_volume_ratio',
    '1d_volume_ratio',

    #'1h_SMA',
    #'4h_SMA',
    #'1d_SMA',

    'close_over_1h_SMA',
    'close_over_4h_SMA',
    'close_over_1d_SMA',

    '1h_trend',
    '4h_trend',
    '1d_trend',

    'bullish_alignment',
    'bearish_alignment',
    'mixed_trend_signals',

    # Support and resistance features
    #'1h_support',
    #'1h_resistance',
    #'4h_support',
    #'4h_resistance',
    #'1d_support',
    #'1d_resistance',

    # Support and resistance proximity features
    #'1h_dist_to_support',
    #'1h_dist_to_resistance',
    #'4h_dist_to_support',
    #'4h_dist_to_resistance',
    #'1d_dist_to_support',
    #'1d_dist_to_resistance',

    # Support and resistance levels strength
    #'1h_support_strength',
    #'1h_resistance_strength',
    #'4h_support_strength',
    #'4h_resistance_strength',
    #'1d_support_strength',
    #'1d_resistance_strength',

    # Position relative to support/resistance range
    #'1h_sup_res_position',
    #'4h_sup_res_position',
    #'1d_sup_res_position',

    # Price at support/resistance
    #'1h_at_support',
    #'1h_at_resistance',
    #'4h_at_support',
    #'4h_at_resistance',
    #'1d_at_support',
    #'1d_at_resistance',

    # Support/resistance breakouts
    # '1h_breaking_support',
    # '1h_breaking_resistance',
    #  '4h_breaking_support',
    # '4h_breaking_resistance',
    # '1d_breaking_support',
    # '1d_breaking_resistance',

    # Multi-timeframe support/resistance features
    # 'multi_timeframe_support',
    # 'multi_timeframe_resistance',
    # 'strong_support_zone',
    # 'strong_resistance_zone',
    # 'in_sr_range',
    # 'breaking_sr_range',

    # Range width of support/resistance
    # '1h_sr_range_width',
    # '4h_sr_range_width',
    # '1d_sr_range_width',

    # Support/resistance count in proximity
    # '1h_sup_res_count',
    # '4h_sup_res_count',
    # '1d_sup_res_count',

    # 'double_top_5m',
    # 'double_bottom_5m',
    # 'double_top_1h',
    # 'double_bottom_1h',
    # 'double_top_4h',
    #  'double_bottom_4h',
    # 'double_top_1d',
    # 'double_bottom_1d',
    # 'has_breakout'
]


def compute_cci(df, period=20):
    """
    Calcule le Commodity Channel Index (CCI) pour un DataFrame OHLC.
    Args:
        df (pd.DataFrame): Doit contenir les colonnes 'High', 'Low', 'Close'.
        period (int): Période de calcul du CCI.
    Returns:
        pd.Series: Valeurs du CCI.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad_tp = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
    cci = (tp - sma_tp) / (0.015 * mad_tp)
    cci = cci.replace([np.inf, -np.inf], np.nan)
    return cci


# =========== MODEL BUILDING FUNCTIONS ===========

def build_rnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_deep_lstm_model(input_shape):
    # Hyperparamètres optimisés par Optuna
    lstm_units1 = 185
    lstm_units2 = 78
    dense_units = 18
    dropout_rate_lstm1 = 0.4
    dropout_rate_lstm2 = 0.3
    learning_rate = 0.00042

    initializer = HeNormal(seed=selectSeed)

    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units1, return_sequences=True, kernel_initializer=initializer),
        Dropout(dropout_rate_lstm1),
        LSTM(lstm_units2, return_sequences=False, kernel_initializer=initializer),
        Dropout(dropout_rate_lstm2),
        Dense(dense_units, activation='relu', kernel_initializer=initializer),
        Dense(1, activation='sigmoid', kernel_initializer=initializer)
    ])

    # Changer le taux d'apprentissage de 0.001 à 0.0005
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def configure_early_stopping(patience=3):
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )


def create_deep_lstm_model(input_shape):
    model = build_deep_lstm_model(input_shape)
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = configure_early_stopping(patience=5)
    return model, early_stopping


# =========== DATA LOADING FUNCTIONS ===========

#def load_json_file(file_path):
#    with open(file_path, 'r') as file:
#        data = json.load(file)
#        if isinstance(data, list):
#            return pd.DataFrame(data)
#    return pd.DataFrame()


#def load_json_files_from_directory(directory_path):
#    data_frames = []
#    for filename in os.listdir(directory_path):
#        if filename.endswith('.json'):
#            file_path = os.path.join(directory_path, filename)
#            data_frames.append(load_json_file(file_path))
#    combined_df = pd.concat(data_frames, ignore_index=True)
#    return combined_df


def validate_json_structure(data, file_path):
    """
    Validate that the JSON data has the required structure with intervalsDataPoints and dataPoints.

    Required structure:
    - intervalsDataPoints array
    - Each intervalsDataPoint must have a dataPoints array
    - dataPoints must have timestamp, openPrice, highPrice, lowPrice, closePrice

    Returns:
    - True if structure is valid
    - False if structure is invalid
    """
    # Check for top-level intervalsDataPoints array
    if not isinstance(data, dict) or 'intervalsDataPoints' not in data:
        print(f"Invalid JSON structure in {file_path}: Missing 'intervalsDataPoints' array")
        return False

    # Check that intervalsDataPoints is an array
    if not isinstance(data['intervalsDataPoints'], list):
        print(f"Invalid JSON structure in {file_path}: 'intervalsDataPoints' is not an array")
        return False

    # Check each interval has dataPoints array
    for i, interval in enumerate(data['intervalsDataPoints']):
        if not isinstance(interval, dict) or 'dataPoints' not in interval:
            print(f"Invalid JSON structure in {file_path}: Missing 'dataPoints' array in interval {i}")
            return False

        if not isinstance(interval['dataPoints'], list):
            print(f"Invalid JSON structure in {file_path}: 'dataPoints' is not an array in interval {i}")
            return False

        # Check at least one datapoint for required fields
        if not interval['dataPoints']:
            continue

        # Check first datapoint for required fields
        sample_datapoint = interval['dataPoints'][0]
        required_fields = ['timestamp', 'openPrice', 'highPrice', 'lowPrice', 'closePrice']
        for field in required_fields:
            if field not in sample_datapoint:
                print(f"Invalid JSON structure in {file_path}: Missing required field '{field}' in dataPoints")
                return False

    return True


def load_json_file(file_path):
    """Load data from a single JSON file with structure validation."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Format 1: List of objects
            if isinstance(data, list):
                # Format for oil files
                return pd.DataFrame(data)

            # Format 2: Object with intervalsDataPoints array
            elif isinstance(data, dict):
                if 'intervalsDataPoints' in data:
                    # Validate the JSON structure
                    if not validate_json_structure(data, file_path):
                        # Log the invalid structure but continue processing
                        print(f"Warning: File has invalid structure: {file_path}")
                        return pd.DataFrame()

                    all_data = []
                    for interval in data['intervalsDataPoints']:
                        # Check if there are any data points
                        if not interval.get('dataPoints'):  # If 'dataPoints' is empty
                            continue  # Skip this interval if there are no data points

                        for datapoint in interval['dataPoints']:
                            # Process datapoint as usual
                            try:
                                if isinstance(datapoint.get('closePrice'), dict) and 'ask' in datapoint['closePrice'] and 'bid' in datapoint['closePrice']:
                                    record = {
                                        'FromDate': datetime.fromtimestamp(datapoint['timestamp'] / 1000, tz=pytz.UTC),
                                        'Open': (datapoint['openPrice']['ask'] + datapoint['openPrice']['bid']) / 2,
                                        'High': (datapoint['highPrice']['ask'] + datapoint['highPrice']['bid']) / 2,
                                        'Low': (datapoint['lowPrice']['ask'] + datapoint['lowPrice']['bid']) / 2,
                                        'Close': (datapoint['closePrice']['ask'] + datapoint['closePrice']['bid']) / 2,
                                        'Volume': datapoint.get('lastTradedVolume', 0)  # Use volume if available
                                    }
                                else:
                                    # Ensure closePrice and other fields are numeric values, not dicts
                                    close_price = datapoint['closePrice'] if not isinstance(datapoint['closePrice'], dict) else (datapoint['closePrice'].get('ask', 0) + datapoint['closePrice'].get('bid', 0)) / 2
                                    open_price = datapoint['openPrice'] if not isinstance(datapoint['openPrice'], dict) else (datapoint['openPrice'].get('ask', 0) + datapoint['openPrice'].get('bid', 0)) / 2
                                    high_price = datapoint['highPrice'] if not isinstance(datapoint['highPrice'], dict) else (datapoint['highPrice'].get('ask', 0) + datapoint['highPrice'].get('bid', 0)) / 2
                                    low_price = datapoint['lowPrice'] if not isinstance(datapoint['lowPrice'], dict) else (datapoint['lowPrice'].get('ask', 0) + datapoint['lowPrice'].get('bid', 0)) / 2

                                    record = {
                                        'FromDate': datetime.fromtimestamp(datapoint['timestamp'] / 1000, tz=pytz.UTC),
                                        'Open': open_price,
                                        'High': high_price,
                                        'Low': low_price,
                                        'Close': close_price,
                                        'Volume': datapoint.get('lastTradedVolume', 0)  # Use volume if available
                                    }

                                # Skip the record if Close is 0
                                if record['Close'] == 0:
                                    continue

                                all_data.append(record)
                            except Exception as e:
                                print(f"Error processing datapoint in {file_path}: {e}")
                                print(f"Datapoint structure: {datapoint}")
                                continue

                    return pd.DataFrame(all_data)
                else:
                    print(f"Unsupported JSON structure in {file_path}: Missing 'intervalsDataPoints'")
                    return pd.DataFrame()

            # Unknown format
            print(f"Unsupported JSON structure in {file_path}")
            return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON file {file_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_json_files_from_directory(directory_path):
    """Load and combine data from all JSON files in a directory."""
    data_frames = []
    invalid_files = []

    # Track the number of files processed
    total_files = 0
    valid_files = 0

    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            total_files += 1
            file_path = os.path.join(directory_path, filename)
            df = load_json_file(file_path)

            if not df.empty:
                data_frames.append(df)
                valid_files += 1
            else:
                invalid_files.append(filename)

    # Report summary
    print(f"Processed {total_files} JSON files:")
    print(f"- Successfully loaded {valid_files} files")
    print(f"- Failed to load {len(invalid_files)} files")

    if invalid_files:
        print("Invalid files with incorrect structure:")
        for file in invalid_files:
            print(f"- {file}")

    if not data_frames:
        print(f"Warning: No valid data loaded from {directory_path}")
        return pd.DataFrame()

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


# La fonction filter_by_time_interval filtre un DataFrame pour ne conserver que les lignes où l’intervalle de temps
# entre deux dates consécutives (dans la colonne FromDate) est exactement égal à la valeur spécifiée par interval (en minutes).
def filter_by_time_interval(df, interval):
    df['prev_date'] = df['FromDate'].shift(1)
    df['time_diff'] = (df['FromDate'] - df['prev_date']).dt.total_seconds() / 60
    return df[df['time_diff'] == interval]


def remove_zero_open_close(df):
    print(df[['Open', 'Close']].dtypes)
    print(df[(df['Open'] == 0) & (df['Close'] == 0)])
    print(df[(df['Open'] == '0') & (df['Close'] == '0')])
    """Supprime les lignes où 'Open' et 'Close' sont à 0.0."""
    return df[~((df['Open'] == 0.0) & (df['Close'] == 0.0))]


def preprocess_features(features_df):
    features_df['FromDate'] = pd.to_datetime(features_df['FromDate'])
    # features_df = remove_zero_open_close(features_df)
    features_df = features_df.sort_values(by='FromDate')

    features_df = filter_by_time_interval(features_df, interval=5)
    return features_df

# =========== FEATURE ENGINEERING FUNCTIONS ===========

def add_time_columns(df):
    df['year'] = df['FromDate'].dt.year
    df['month'] = df['FromDate'].dt.month
    df['day'] = df['FromDate'].dt.day
    df['hour'] = df['FromDate'].dt.hour
    df['minute'] = df['FromDate'].dt.minute
    df['day_of_week'] = df['FromDate'].dt.dayofweek
    return df


def volume_weighted_rsi_sma(df, rsi_period=14, sma_period=10):
    """Calcule le Vol_Weighted_RSI_SMA"""
    df['RSI'] = calculate_rsi(df, rsi_period)
    df['SMA_RSI'] = df['RSI'].rolling(window=sma_period, min_periods=1).mean()

    # Volume Weighted RSI SMA
    df['Vol_Weighted_RSI_SMA'] = (df['SMA_RSI'] * df['Volume']).rolling(window=sma_period, min_periods=1).sum() / df['Volume'].rolling(window=sma_period, min_periods=1).sum()

    return df


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_future_direction(df):
    # df['future_direction'] = ((df['Close'].shift(-5) - df['Close']) / df['Close'] > 0.005).astype(int)
    df['future_direction_2'] = (df['Close'].shift(-4) > df['Close']).astype(int) # prévision à 35 minutes
    #print(df['future_direction'].value_counts())
    print(df['future_direction_2'].value_counts())
    return df


def add_direction(df):
    df['direction'] = (df['Close'] > df['Open']).astype(int)
    print(df['direction'].value_counts())
    return df


def add_candle_features(df):
    df['candle_range'] = df['High'] - df['Low']
    print("==== Bougies avec candle_range == 0 ====")
    print(df[df['candle_range'] == 0][['FromDate', 'High', 'Low', 'Open', 'Close']])


    df['corps_candle'] = np.abs(df['Close'] - df['Open']) / df['candle_range']
    df['meche_haute'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['candle_range']
    df['meche_basse'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['candle_range']
    df['candle_trend'] = (df['Close'] > df['Open']).astype(int)
    return df



def add_doji(
        df,
        four_price_tolerance=0.01,
        body_to_range_ratio=0.1,
        dragonfly_threshold=0.5,
        gravestone_threshold=0.5,
        long_legged_threshold=0.25,
        perfect_doji_ratio=0.02,
        high_wave_threshold=0.4,
        spinning_top_ratio=0.3,
        cross_doji_ratio=0.005,
        inverted_dragonfly_threshold=0.5,
        inverted_gravestone_threshold=0.5,
        tri_star_window=3
):
    """
    Identifie et qualifie les bougies Doji et d'autres patterns japonais dans un DataFrame OHLC.
    Retourne un DataFrame enrichi de colonnes doji_type, doji, doji_strength, perfect_doji, upper_wick, lower_wick, ratio_corps, doji_invalid, pattern_type.
    """
    result = df.copy()

    # Calculs robustes
    if 'candle_range' not in result.columns:
        result['candle_range'] = result['High'] - result['Low']

    zero_range_mask = result['candle_range'] < 1e-9
    result.loc[zero_range_mask, 'candle_range'] = 1e-9

    body_size = (result['Close'] - result['Open']).abs()
    body_range_ratio = body_size / result['candle_range']

    doji_mask = body_range_ratio < body_to_range_ratio

    upper_wick = (result['High'] - result[['Open', 'Close']].max(axis=1)).clip(lower=0)
    lower_wick = (result[['Open', 'Close']].min(axis=1) - result['Low']).clip(lower=0)

    upper_wick_ratio = upper_wick / result['candle_range']
    lower_wick_ratio = lower_wick / result['candle_range']

    four_price_mask = (result['High'] - result['Low'] < four_price_tolerance) & (body_size < four_price_tolerance)
    dragonfly_mask = (lower_wick_ratio > dragonfly_threshold) & (upper_wick_ratio < 0.1)
    gravestone_mask = (upper_wick_ratio > gravestone_threshold) & (lower_wick_ratio < 0.1)
    long_legged_mask = (upper_wick_ratio > long_legged_threshold) & (lower_wick_ratio > long_legged_threshold)
    high_wave_mask = ((upper_wick_ratio > high_wave_threshold) | (lower_wick_ratio > high_wave_threshold)) & doji_mask
    spinning_top_mask = (
            (body_range_ratio < spinning_top_ratio)
            & (body_range_ratio >= body_to_range_ratio)
            & (upper_wick_ratio > 0.1)
            & (lower_wick_ratio > 0.1)
    )
    # Définition améliorée pour le hammer (ajustée pour une meilleure distinction)
    hammer_mask = (
            (lower_wick_ratio > 2 * body_range_ratio)
            & (upper_wick_ratio < 0.1)
            & (body_range_ratio < 0.3)
            & (result['Close'] > result['Open'])  # Ajout d'une condition pour différencier du hanging_man
    )
    shooting_star_mask = (
            (upper_wick_ratio > 2 * body_range_ratio)
            & (lower_wick_ratio < 0.1)
            & (body_range_ratio < 0.3)
    )

    # cross_doji_mask = (body_range_ratio < cross_doji_ratio) & (upper_wick_ratio > 0.2) & (lower_wick_ratio > 0.2)
    # inverted_dragonfly_mask = (upper_wick_ratio > inverted_dragonfly_threshold) & (lower_wick_ratio < 0.1) & doji_mask
    # inverted_gravestone_mask = (lower_wick_ratio > inverted_gravestone_threshold) & (upper_wick_ratio < 0.1) & doji_mask


    # Cross Doji avec une définition plus souple
    # Réduction du seuil pour les mèches (0.2 -> 0.1) et augmentation du ratio du corps (0.005 -> 0.01)
    cross_doji_mask = (body_range_ratio < cross_doji_ratio) & (upper_wick_ratio > 0.1) & (lower_wick_ratio > 0.1)

    # Clarification des types inversés (simplification et renommage pour éviter la confusion)
    # Utilisation d'une formulation plus directe et abandonnant l'utilisation du terme "inversé"
    upper_wick_doji_mask = (upper_wick_ratio > inverted_dragonfly_threshold) & (lower_wick_ratio < 0.1)
    lower_wick_doji_mask = (lower_wick_ratio > inverted_gravestone_threshold) & (upper_wick_ratio < 0.1)

    # Ajout de nouveaux types de bougies
    hanging_man_mask = (
            (lower_wick_ratio > 2 * body_range_ratio)
            & (upper_wick_ratio < 0.1)
            & (body_range_ratio < 0.3)
            & (result['Close'] < result['Open'])  # Corps baissier
    )

    # Umbrella (parapluie) pattern
    umbrella_mask = (
            (lower_wick_ratio > 0.6)
            & (upper_wick_ratio < 0.1)
            & (body_range_ratio < 0.4)
    )

    # Belt Hold
    belt_hold_bull_mask = (
            (result['Open'] == result['Low'])
            & (result['Close'] > result['Open'])
            & (body_range_ratio > 0.7)
            & (upper_wick_ratio < 0.1)
    )

    belt_hold_bear_mask = (
            (result['Open'] == result['High'])
            & (result['Close'] < result['Open'])
            & (body_range_ratio > 0.7)
            & (lower_wick_ratio < 0.1)
    )

    # Bougie à corps plein (Marubozu complet)
    full_marubozu_bull_mask = (
            (result['Open'] == result['Low'])
            & (result['Close'] == result['High'])
    )

    full_marubozu_bear_mask = (
            (result['Open'] == result['High'])
            & (result['Close'] == result['Low'])
    )

    # Bougie à corps long (Long Body)
    long_body_mask = (body_range_ratio > 0.7)
    long_body_bull_mask = long_body_mask & (result['Close'] > result['Open'])
    long_body_bear_mask = long_body_mask & (result['Close'] < result['Open'])

    # Tri-Star Doji (3 doji consécutifs)
    tri_star_mask = pd.Series(False, index=result.index)
    if len(result) >= tri_star_window:
        rolling_doji = doji_mask.rolling(window=tri_star_window, min_periods=tri_star_window).sum()
        tri_star_mask = (rolling_doji == tri_star_window)
        tri_star_mask = tri_star_mask.shift(-(tri_star_window - 1)).fillna(False)

    # --- Ajout d'autres patterns classiques ---
    # Engulfing (Avalement)
    engulfing_bull = ((result['Open'] < result['Close'].shift(1)) &
                      (result['Close'] > result['Open'].shift(1)) &
                      (result['Open'] < result['Close']) &
                      (result['Close'].shift(1) < result['Open'].shift(1)))
    engulfing_bear = ((result['Open'] > result['Close'].shift(1)) &
                      (result['Close'] < result['Open'].shift(1)) &
                      (result['Open'] > result['Close']) &
                      (result['Close'].shift(1) > result['Open'].shift(1)))
    harami_bull = ((result['Open'] > result['Open'].shift(1)) &
                   (result['Close'] < result['Close'].shift(1)) &
                   (result['Open'] < result['Close']) &
                   (result['Open'] > result['Close'].shift(1)) &
                   (result['Close'] < result['Open'].shift(1)))
    harami_bear = ((result['Open'] < result['Open'].shift(1)) &
                   (result['Close'] > result['Close'].shift(1)) &
                   (result['Open'] > result['Close']) &
                   (result['Open'] < result['Close'].shift(1)) &
                   (result['Close'] > result['Open'].shift(1)))
    marubozu_bull = ((result['Open'] == result['Low']) & (result['Close'] == result['High']))
    marubozu_bear = ((result['Open'] == result['High']) & (result['Close'] == result['Low']))

    # Morning Star / Evening Star (3 bougies)
    morning_star = pd.Series(False, index=result.index)
    evening_star = pd.Series(False, index=result.index)
    if len(result) >= 3:
        prev = result.shift(1)
        prev2 = result.shift(2)
        # Morning Star: baissière, petite bougie, haussière
        morning_star = (
                (prev2['Close'] < prev2['Open']) &
                (prev['Close'] < prev['Open']) &
                (result['Close'] > result['Open']) &
                (result['Close'] > prev2['Open'])
        )
        # Evening Star: haussière, petite bougie, baissière
        evening_star = (
                (prev2['Close'] > prev2['Open']) &
                (prev['Close'] > prev['Open']) &
                (result['Close'] < result['Open']) &
                (result['Close'] < prev2['Open'])
        )

        # Abandoned Baby (bébé abandonné) pattern
        abandoned_baby_bull = (
                (prev2['Close'] < prev2['Open']) &  # Première bougie baissière
                (prev['Low'] > prev2['Low']) &      # Gap baissier
                (prev['High'] < result['Low']) &    # Gap haussier
                (result['Close'] > result['Open'])  # Dernière bougie haussière
        )

        abandoned_baby_bear = (
                (prev2['Close'] > prev2['Open']) &  # Première bougie haussière
                (prev['High'] < prev2['High']) &    # Gap haussier
                (prev['Low'] > result['High']) &    # Gap baissier
                (result['Close'] < result['Open'])  # Dernière bougie baissière
        )

        # Three Inside Up/Down
        three_inside_up = (
                harami_bull.shift(1) &              # Harami haussier
                (result['Close'] > prev['Close'])   # Confirmation haussière
        )

        three_inside_down = (
                harami_bear.shift(1) &              # Harami baissier
                (result['Close'] < prev['Close'])   # Confirmation baissière
        )

    # Piercing Line / Dark Cloud Cover (2 bougies)
    piercing_line = ((result['Open'] < result['Close'].shift(1)) &
                     (result['Close'] > (result['Open'].shift(1) + result['Close'].shift(1)) / 2) &
                     (result['Close'] > result['Open']))
    dark_cloud = ((result['Open'] > result['Close'].shift(1)) &
                  (result['Close'] < (result['Open'].shift(1) + result['Close'].shift(1)) / 2) &
                  (result['Close'] < result['Open']))

    # Kicking patterns (2 bougies marubozu avec gap)
    if len(result) >= 2:
        kicking_bull = (
                marubozu_bear.shift(1) &           # Première bougie marubozu baissière
                marubozu_bull &                    # Deuxième bougie marubozu haussière
                (result['Low'] > result['Close'].shift(1))  # Gap haussier
        )

        kicking_bear = (
                marubozu_bull.shift(1) &           # Première bougie marubozu haussière
                marubozu_bear &                    # Deuxième bougie marubozu baissière
                (result['High'] < result['Close'].shift(1))  # Gap baissier
        )

        # Tasuki Gaps
        rising_window = (result['Low'] > result['High'].shift(1))
        falling_window = (result['High'] < result['Low'].shift(1))

        up_tasuki_gap = (
                (result['Close'].shift(2) < result['Open'].shift(2)) &  # Première bougie baissière
                (result['Close'].shift(1) > result['Open'].shift(1)) &  # Deuxième bougie haussière
                rising_window.shift(1) &                                # Gap haussier
                (result['Close'] < result['Open']) &                    # Troisième bougie baissière
                (result['Open'] > result['Close'].shift(1)) &           # Ouverture dans le corps de la deuxième bougie
                (result['Close'] > result['Open'].shift(1))             # Fermeture au-dessus de l'ouverture de la deuxième bougie
        )

        down_tasuki_gap = (
                (result['Close'].shift(2) > result['Open'].shift(2)) &  # Première bougie haussière
                (result['Close'].shift(1) < result['Open'].shift(1)) &  # Deuxième bougie baissière
                falling_window.shift(1) &                               # Gap baissier
                (result['Close'] > result['Open']) &                    # Troisième bougie haussière
                (result['Open'] < result['Close'].shift(1)) &           # Ouverture dans le corps de la deuxième bougie
                (result['Close'] < result['Open'].shift(1))             # Fermeture sous l'ouverture de la deuxième bougie
        )

    # Three White Soldiers / Three Black Crows (3 bougies)
    three_white_soldiers = pd.Series(False, index=result.index)
    three_black_crows = pd.Series(False, index=result.index)
    if len(result) >= 3:
        three_white_soldiers = (
                (result['Close'] > result['Open']) &
                (result['Close'].shift(1) > result['Open'].shift(1)) &
                (result['Close'].shift(2) > result['Open'].shift(2)) &
                (result['Open'] > result['Open'].shift(1)) &  # Critère d'ouverture progressive
                (result['Open'].shift(1) > result['Open'].shift(2))
        )
        three_black_crows = (
                (result['Close'] < result['Open']) &
                (result['Close'].shift(1) < result['Open'].shift(1)) &
                (result['Close'].shift(2) < result['Open'].shift(2)) &
                (result['Open'] < result['Open'].shift(1)) &  # Critère d'ouverture progressive
                (result['Open'].shift(1) < result['Open'].shift(2))
        )

        # Mat Hold pattern (tendance haussière qui continue)
        mat_hold = (
                (result['Close'].shift(4) > result['Open'].shift(4)) &  # Première bougie haussière
                (result['Close'].shift(3) < result['Open'].shift(3)) &  # Deuxième bougie baissière
                (result['Close'].shift(2) < result['Open'].shift(2)) &  # Troisième bougie baissière
                (result['Close'].shift(1) < result['Open'].shift(1)) &  # Quatrième bougie baissière
                (result['Close'] > result['Open']) &                    # Cinquième bougie haussière
                (result['Close'] > result['Close'].shift(4))            # Clôture au-dessus de la première bougie
        )

    # Ordre de priorité des types de doji
    conditions = [
        four_price_mask & doji_mask,
        dragonfly_mask & doji_mask & ~four_price_mask,
        gravestone_mask & doji_mask & ~four_price_mask,
        long_legged_mask & doji_mask & ~four_price_mask,
        cross_doji_mask & ~four_price_mask,  # Déplacé plus haut dans l'ordre de priorité
        high_wave_mask & ~four_price_mask & ~long_legged_mask,
        spinning_top_mask,
        hammer_mask & ~dragonfly_mask,
        hanging_man_mask,
        shooting_star_mask & ~gravestone_mask,
        # Renommage des types pour plus de clarté (éviter confusion avec inverted_*)
        upper_wick_doji_mask & ~gravestone_mask,
        lower_wick_doji_mask & ~dragonfly_mask,
        umbrella_mask,
        belt_hold_bull_mask,
        belt_hold_bear_mask,
        full_marubozu_bull_mask,
        full_marubozu_bear_mask,
        long_body_bull_mask,
        long_body_bear_mask,
        tri_star_mask,
        doji_mask  # Doji standard comme dernier recours
    ]
    choices = [
        'four_price',
        'dragonfly',
        'gravestone',
        'long_legged',
        'cross',  # Déplacé plus haut pour correspondre à la nouvelle priorité
        'high_wave',
        'spinning_top',
        'hammer',
        'hanging_man',
        'shooting_star',
        'upper_wick_doji',  # Renommé pour éviter la confusion (précédemment inverted_dragonfly)
        'lower_wick_doji',  # Renommé pour éviter la confusion (précédemment inverted_gravestone)
        'umbrella',
        'belt_hold_bull',
        'belt_hold_bear',
        'full_marubozu_bull',
        'full_marubozu_bear',
        'long_body_bull',
        'long_body_bear',
        'tri_star',
        'standard'
    ]

    result['doji_type'] = np.select(conditions, choices, default='none')
    result['doji'] = (result['doji_type'] != 'none').astype(int)
    result['doji_strength'] = np.where(
        doji_mask,
        1 - np.clip(body_range_ratio / body_to_range_ratio, 0, 1),
        0
    )
    result['perfect_doji'] = ((result['doji'] == 1) & (body_range_ratio < perfect_doji_ratio)).astype(int)
    result['upper_wick'] = upper_wick
    result['lower_wick'] = lower_wick
    result['ratio_corps'] = body_range_ratio
    result['doji_invalid'] = zero_range_mask.astype(int)

    # Ajout d'une colonne pattern_type pour les autres patterns
    result['pattern_type'] = 'none'
    result.loc[engulfing_bull, 'pattern_type'] = 'engulfing_bull'
    result.loc[engulfing_bear, 'pattern_type'] = 'engulfing_bear'
    result.loc[harami_bull, 'pattern_type'] = 'harami_bull'
    result.loc[harami_bear, 'pattern_type'] = 'harami_bear'
    result.loc[marubozu_bull, 'pattern_type'] = 'marubozu_bull'
    result.loc[marubozu_bear, 'pattern_type'] = 'marubozu_bear'
    result.loc[morning_star, 'pattern_type'] = 'morning_star'
    result.loc[evening_star, 'pattern_type'] = 'evening_star'
    result.loc[piercing_line, 'pattern_type'] = 'piercing_line'
    result.loc[dark_cloud, 'pattern_type'] = 'dark_cloud'
    result.loc[three_white_soldiers, 'pattern_type'] = 'three_white_soldiers'
    result.loc[three_black_crows, 'pattern_type'] = 'three_black_crows'

    # Ajout des nouveaux patterns
    if len(result) >= 2:
        result.loc[kicking_bull, 'pattern_type'] = 'kicking_bull'
        result.loc[kicking_bear, 'pattern_type'] = 'kicking_bear'

    if len(result) >= 3:
        result.loc[abandoned_baby_bull, 'pattern_type'] = 'abandoned_baby_bull'
        result.loc[abandoned_baby_bear, 'pattern_type'] = 'abandoned_baby_bear'
        result.loc[three_inside_up, 'pattern_type'] = 'three_inside_up'
        result.loc[three_inside_down, 'pattern_type'] = 'three_inside_down'
        result.loc[up_tasuki_gap, 'pattern_type'] = 'up_tasuki_gap'
        result.loc[down_tasuki_gap, 'pattern_type'] = 'down_tasuki_gap'

    if len(result) >= 5:
        result.loc[mat_hold, 'pattern_type'] = 'mat_hold'

    return result


def add_candle_trend_relation(df):
    # Si la bougie actuelle et la bougie précédente sont dans la même direction
    df['same_direction'] = (df['candle_trend'] == df['candle_trend'].shift(1)).astype(int)
    return df


def add_engulfing(df):
    """
    Ajoute deux colonnes au DataFrame : bullish_engulfing et bearish_engulfing.

    Cette fonction identifie les patterns d'englobement (engulfing patterns) qui sont
    des indicateurs importants en analyse technique des chandeliers japonais.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes OHLC (Open, High, Low, Close)

    Returns:
        pd.DataFrame: DataFrame avec les colonnes 'bullish_engulfing' et 'bearish_engulfing' ajoutées
    """
    # Vérification des colonnes nécessaires
    required_columns = ['Open', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est requise mais absente du DataFrame")

    # Création de copies pour éviter les SettingWithCopyWarning
    df_copy = df.copy()

    # Calcul du corps des bougies
    current_body = (df_copy['Close'] - df_copy['Open']).abs()
    prev_body = (df_copy['Close'].shift(1) - df_copy['Open'].shift(1)).abs()

    # Identification des tendances des bougies
    current_bullish = df_copy['Close'] > df_copy['Open']
    current_bearish = df_copy['Close'] < df_copy['Open']
    prev_bullish = df_copy['Close'].shift(1) > df_copy['Open'].shift(1)
    prev_bearish = df_copy['Close'].shift(1) < df_copy['Open'].shift(1)

    # Définition de l'englobement
    # Pour le pattern haussier engloutissant (bullish engulfing)
    df_copy['bullish_engulfing'] = (
            current_bullish &                           # Bougie actuelle haussière
            prev_bearish &                              # Bougie précédente baissière
            (df_copy['Open'] <= df_copy['Close'].shift(1)) &  # Ouverture actuelle sous/égale à la clôture précédente
            (df_copy['Close'] >= df_copy['Open'].shift(1)) &  # Clôture actuelle au-dessus/égale à l'ouverture précédente
            (current_body > prev_body * 0.95)           # Corps actuel plus grand que le précédent (avec tolérance de 5%)
    ).astype(int)

    # Pour le pattern baissier engloutissant (bearish engulfing)
    df_copy['bearish_engulfing'] = (
            current_bearish &                           # Bougie actuelle baissière
            prev_bullish &                              # Bougie précédente haussière
            (df_copy['Open'] >= df_copy['Open'].shift(1)) &  # Ouverture actuelle au-dessus/égale à l'ouverture précédente
            (df_copy['Close'] <= df_copy['Close'].shift(1)) &  # Clôture actuelle sous/égale à la clôture précédente
            (current_body > prev_body * 0.95)           # Corps actuel plus grand que le précédent (avec tolérance de 5%)
    ).astype(int)

    # Gestion des valeurs manquantes (première ligne)
    df_copy.loc[df_copy.index[0], ['bullish_engulfing', 'bearish_engulfing']] = 0

    # Calcul d'un score d'intensité pour ces patterns (facultatif)
    df_copy['engulfing_strength'] = np.where(
        (df_copy['bullish_engulfing'] == 1) | (df_copy['bearish_engulfing'] == 1),
        current_body / prev_body,
        0
    )

    return df_copy


def add_wick_features(df):
    df['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['candle_range']
    df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['candle_range']
    return df


def add_support_resistance_levels(df, timeframe, window=10, n_points=2):
    """
    Identify support and resistance levels based on local minima and maxima.

    Args:
        df: DataFrame with price data
        timeframe: Timeframe string ('1h', '4h', '1d')
        window: Number of periods to look for local extrema
        n_points: Number of points required to confirm a level

    Returns:
        DataFrame with added support and resistance features
    """
    # Column prefixes for the specific timeframe
    high_col = f'{timeframe}_High'
    low_col = f'{timeframe}_Low'

    # Create columns for support and resistance
    support_col = f'{timeframe}_support'
    resistance_col = f'{timeframe}_resistance'

    # Initialize columns
    df[support_col] = np.nan
    df[resistance_col] = np.nan

    # Function to find local minima (supports)
    def is_support(i, data, low_column, n=n_points):
        # Check if a point is a support level
        if i - n < 0 or i + n >= len(data):
            return False

        # Check if the current point is a local minimum
        support_condition = True
        for j in range(1, n + 1):
            support_condition = support_condition and (data.iloc[i][low_column] <= data.iloc[i - j][low_column]) and \
                                (data.iloc[i][low_column] <= data.iloc[i + j][low_column])
        return support_condition

    # Function to find local maxima (resistances)
    def is_resistance(i, data, high_column, n=n_points):
        # Check if a point is a resistance level
        if i - n < 0 or i + n >= len(data):
            return False

        # Check if the current point is a local maximum
        resistance_condition = True
        for j in range(1, n + 1):
            resistance_condition = resistance_condition and (data.iloc[i][high_column] >= data.iloc[i - j][high_column]) and \
                                   (data.iloc[i][high_column] >= data.iloc[i + j][high_column])
        return resistance_condition

    # Create a copy of the dataframe with only the timeframe data
    # Drop duplicates to get the actual timeframe data points
    tf_df = df[[f'{timeframe}_Open', f'{timeframe}_High', f'{timeframe}_Low', f'{timeframe}_Close', 'FromDate']].copy()
    tf_df.drop_duplicates(subset=['FromDate', f'{timeframe}_Open', f'{timeframe}_Close'], inplace=True)
    tf_df.sort_values('FromDate', inplace=True)
    tf_df.reset_index(drop=True, inplace=True)

    # Find all support and resistance levels in the timeframe data
    supports = []
    resistances = []

    for i in range(len(tf_df)):
        if is_support(i, tf_df, low_col):
            supports.append((tf_df.iloc[i]['FromDate'], tf_df.iloc[i][low_col]))
        if is_resistance(i, tf_df, high_col):
            resistances.append((tf_df.iloc[i]['FromDate'], tf_df.iloc[i][high_col]))

    # Group nearby levels (within 0.5% of each other)
    def group_levels(levels, threshold=0.005):
        if not levels:
            return []

        levels.sort(key=lambda x: x[1])  # Sort by price
        grouped = []
        current_group = [levels[0]]

        for i in range(1, len(levels)):
            # If the current level is within threshold of the previous level's average
            avg_price = sum(l[1] for l in current_group) / len(current_group)
            if abs(levels[i][1] - avg_price) / avg_price < threshold:
                current_group.append(levels[i])
            else:
                # Add the average of the current group
                avg_date = current_group[-1][0]  # Use the most recent date
                avg_price = sum(l[1] for l in current_group) / len(current_group)
                grouped.append((avg_date, avg_price))
                current_group = [levels[i]]

        # Add the last group
        if current_group:
            avg_date = current_group[-1][0]
            avg_price = sum(l[1] for l in current_group) / len(current_group)
            grouped.append((avg_date, avg_price))

        return grouped

    grouped_supports = group_levels(supports)
    grouped_resistances = group_levels(resistances)

    # Sort by date
    grouped_supports.sort(key=lambda x: x[0])
    grouped_resistances.sort(key=lambda x: x[0])

    # For each 5min data point, find the most recent support and resistance levels
    for i in range(len(df)):
        current_date = df.iloc[i]['FromDate']

        # Find the most recent n support levels before the current date
        recent_supports = [s[1] for s in grouped_supports if s[0] <= current_date][-window:]
        if recent_supports:
            df.at[i, support_col] = min(recent_supports)  # Use the lowest support level

        # Find the most recent n resistance levels before the current date
        recent_resistances = [r[1] for r in grouped_resistances if r[0] <= current_date][-window:]
        if recent_resistances:
            df.at[i, resistance_col] = max(recent_resistances)  # Use the highest resistance level

    # Create a feature that shows how many support/resistance levels are in proximity
    df[f'{timeframe}_sup_res_count'] = 0

    for i in range(len(df)):
        current_price = df.iloc[i]['Close']
        proximity_threshold = 0.01 * current_price  # 1% of current price

        support_count = sum(1 for s in grouped_supports
                            if s[0] <= df.iloc[i]['FromDate'] and
                            abs(s[1] - current_price) < proximity_threshold)

        resistance_count = sum(1 for r in grouped_resistances
                               if r[0] <= df.iloc[i]['FromDate'] and
                               abs(r[1] - current_price) < proximity_threshold)

        df.at[i, f'{timeframe}_sup_res_count'] = support_count + resistance_count

    # Add information about the strength of support/resistance (how many times tested)
    df[f'{timeframe}_support_strength'] = 0
    df[f'{timeframe}_resistance_strength'] = 0

    for i in range(len(df)):
        current_price = df.iloc[i]['Close']
        current_support = df.iloc[i][support_col]
        current_resistance = df.iloc[i][resistance_col]

        if not np.isnan(current_support):
            # Count how many times price approached this support level
            support_touches = sum(1 for j in range(max(0, i-window*5), i)
                                  if abs(df.iloc[j]['Low'] - current_support) / current_support < 0.005)
            df.at[i, f'{timeframe}_support_strength'] = support_touches

        if not np.isnan(current_resistance):
            # Count how many times price approached this resistance level
            resistance_touches = sum(1 for j in range(max(0, i-window*5), i)
                                     if abs(df.iloc[j]['High'] - current_resistance) / current_resistance < 0.005)
            df.at[i, f'{timeframe}_resistance_strength'] = resistance_touches

    return df


def calculate_price_to_level_proximity(df):
    """
    Calculate the proximity of current price to support and resistance levels
    and create features about price in relation to these levels.

    Args:
        df: DataFrame with support and resistance levels already calculated

    Returns:
        DataFrame with added proximity features
    """
    # For each timeframe
    for timeframe in ['1h', '4h', '1d']:
        support_col = f'{timeframe}_support'
        resistance_col = f'{timeframe}_resistance'

        # Distance to nearest support (normalized by price)
        df[f'{timeframe}_dist_to_support'] = (df['Close'] - df[support_col]) / df['Close']

        # Distance to nearest resistance (normalized by price)
        df[f'{timeframe}_dist_to_resistance'] = (df[resistance_col] - df['Close']) / df['Close']

        # Relative position between support and resistance
        df[f'{timeframe}_sup_res_position'] = (df['Close'] - df[support_col]) / (df[resistance_col] - df[support_col])

        # Is price close to support? (within 0.5%)
        df[f'{timeframe}_at_support'] = (df[f'{timeframe}_dist_to_support'].abs() < 0.005).astype(int)

        # Is price close to resistance? (within 0.5%)
        df[f'{timeframe}_at_resistance'] = (df[f'{timeframe}_dist_to_resistance'].abs() < 0.005).astype(int)

        # Is price breaking support? (price below support)
        df[f'{timeframe}_breaking_support'] = (df[f'{timeframe}_dist_to_support'] < 0).astype(int)

        # Is price breaking resistance? (price above resistance)
        df[f'{timeframe}_breaking_resistance'] = (df[f'{timeframe}_dist_to_resistance'] < 0).astype(int)

        # Support/Resistance Range Width (normalized by price)
        df[f'{timeframe}_sr_range_width'] = (df[resistance_col] - df[support_col]) / df['Close']

    # Create combined features across timeframes

    # Are we at support on multiple timeframes?
    df['multi_timeframe_support'] = df['1h_at_support'] + df['4h_at_support'] + df['1d_at_support']

    # Are we at resistance on multiple timeframes?
    df['multi_timeframe_resistance'] = df['1h_at_resistance'] + df['4h_at_resistance'] + df['1d_at_resistance']

    # Strong support zone (support in at least 2 timeframes)
    df['strong_support_zone'] = (df['multi_timeframe_support'] >= 2).astype(int)

    # Strong resistance zone (resistance in at least 2 timeframes)
    df['strong_resistance_zone'] = (df['multi_timeframe_resistance'] >= 2).astype(int)

    # In range or breaking out?
    df['in_sr_range'] = ((df['1h_breaking_support'] == 0) &
                         (df['1h_breaking_resistance'] == 0)).astype(int)

    # Breaking out of range?
    df['breaking_sr_range'] = ((df['1h_breaking_support'] == 1) |
                               (df['1h_breaking_resistance'] == 1)).astype(int)

    return df


def add_body_ratio(df):
    df['corps_candle_prev'] = df['corps_candle'].shift(1)
    df['corps_sum'] = df['corps_candle'] + df['corps_candle_prev']
    df['ratio_corps'] = df['corps_candle'] / df['corps_sum']
    return df


def get_market_opening(date):
    """
    Détermine l'heure d'ouverture de la bourse américaine des commodities
    et si c'est l'horaire d'hiver ou d'été.
    """
    ny_tz = pytz.timezone("America/New_York")
    market_open_winter = 15  # 9h00 AM heure de New York (hiver, heure de Paris)
    market_open_summer = 14  # 8h00 AM heure de New York (été, heure de Paris)
    stock_open_winter = 15.5  # 9h30 AM NY (hiver, 15h30 heure de Paris)
    stock_open_summer = 14.5  # 8h30 AM NY (été, 14h30 heure de Paris)

    # Déterminer si la date est en heure d'été ou d'hiver
    localized_date = ny_tz.localize(datetime(date.year, date.month, date.day))
    is_summer = localized_date.dst() != pd.Timedelta(0)

    return market_open_summer if is_summer else market_open_winter, \
        stock_open_summer if is_summer else stock_open_winter, is_summer


def get_period_of_day_with_timezone(date):
    """
    Détermine la période de la journée en tenant compte de l'heure d'été et d'hiver.
    """
    # Définit le fuseau horaire (ici, New York par exemple, à adapter selon ton besoin)
    tz = pytz.timezone("America/New_York")

    # Si la date est déjà localisée, on la garde telle quelle
    if date.tzinfo is None:
        # Localiser la date si elle n'est pas déjà dans un fuseau horaire
        local_time = tz.localize(date)
    else:
        # Si la date est déjà localisée, on convertit juste en fuseau horaire
        local_time = date.astimezone(tz)

    # Vérifie si la date est en heure d'été (Daylight Saving Time)
    if local_time.dst() != timedelta(0):
        # Heure d'été : décaler les périodes d'une heure (par exemple, commencer plus tard)
        if 0 <= local_time.hour < 7:  # Nuit (00h à 07h)
            return 0  # Retourne 0 pour Night
        elif 7 <= local_time.hour < 13:  # Matin (07h à 13h)
            return 1  # Retourne 1 pour Morning
        elif 13 <= local_time.hour < 19:  # Après-midi (13h à 19h)
            return 2  # Retourne 2 pour Afternoon
        else:  # Soir (19h à 00h)
            return 3  # Retourne 3 pour Evening
    else:
        # Heure d'hiver : plage horaire classique
        if 0 <= local_time.hour < 6:  # Nuit (00h à 06h)
            return 0  # Retourne 0 pour Night
        elif 6 <= local_time.hour < 12:  # Matin (06h à 12h)
            return 1  # Retourne 1 pour Morning
        elif 12 <= local_time.hour < 18:  # Après-midi (12h à 18h)
            return 2  # Retourne 2 pour Afternoon
        else:  # Soir (18h à 00h)
            return 3  # Retourne 3 pour Evening

# =========== VISUALIZATION FUNCTIONS ===========

def plot_training_history(history, filename="./doc/assets/training_history.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    try:
        plt.show()
    except:
        print("Affichage désactivé en mode non interactif, mais l'image est sauvegardée.")

    print(f"Image enregistrée sous {filename}")


def visualize_lstm_weights(model, filename="./doc/assets/lstm_weights.png"):
    plt.switch_backend('agg')
    weights, biases = model.layers[1].get_weights()
    print("Poids de la première couche LSTM :")
    print(weights.shape)
    print("Biais de la première couche LSTM :")
    print(biases.shape)
    plt.figure(figsize=(10, 6))
    plt.imshow(weights, aspect='auto', cmap='coolwarm')
    plt.colorbar()
    plt.title('Poids de la première couche LSTM')
    plt.xlabel('Neurones de la couche suivante')
    plt.ylabel('Neurones de la couche précédente')

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"Image enregistrée sous {filename}")

    try:
        plt.show()
    except:
        print("Affichage désactivé en mode non interactif, mais l'image est sauvegardée.")


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calcule le MACD et la ligne de signal."""
    # Calcul des moyennes mobiles exponentielles (EMA)
    df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    # Calcul du MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Calcul du signal (EMA 9 du MACD)
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    return df


def analyze_errors(features_df, y_test, y_pred, filename="./doc/assets/wrong_predictions.png"):
    """Analyze prediction errors and save visualizations with more detailed insights."""
    # Identify wrong predictions
    wrong_indices = np.where(y_pred != y_test)[0]
    wrong_predictions = features_df.iloc[wrong_indices].copy()

    if len(wrong_predictions) == 0:
        print("No prediction errors found!")
        return pd.DataFrame()

    wrong_predictions['true_direction'] = y_test[wrong_indices]
    wrong_predictions['predicted_direction'] = y_pred[wrong_indices]

    # Calculate error rate
    error_rate = len(wrong_indices) / len(y_test) * 100
    print(f"Error rate: {error_rate:.2f}% ({len(wrong_indices)} out of {len(y_test)})")

    # Error types analysis
    false_positives = wrong_predictions[wrong_predictions['predicted_direction'] == 1]
    false_negatives = wrong_predictions[wrong_predictions['predicted_direction'] == 0]

    print(f"False positives (predicted Up when actually Down): {len(false_positives)}")
    print(f"False negatives (predicted Down when actually Up): {len(false_negatives)}")

    # Create a directory for error analysis
    error_dir = os.path.dirname(filename)
    if error_dir and not os.path.exists(error_dir):
        os.makedirs(error_dir)

    # Errors by hour
    plt.figure(figsize=(12, 6))
    hour_counts = wrong_predictions['hour'].value_counts().sort_index()
    total_by_hour = features_df['hour'].value_counts().sort_index()
    error_rates_by_hour = (hour_counts / total_by_hour * 100).fillna(0)

    ax = sns.barplot(x=error_rates_by_hour.index, y=error_rates_by_hour.values, palette='Reds')
    plt.title("Error Rate by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Error Rate (%)")

    # Add count labels
    for i, v in enumerate(error_rates_by_hour.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')

    plt.savefig(f"{os.path.splitext(filename)[0]}_by_hours.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Errors by day of week
    plt.figure(figsize=(12, 6))
    day_counts = wrong_predictions['day_of_week'].value_counts().sort_index()
    total_by_day = features_df['day_of_week'].value_counts().sort_index()
    error_rates_by_day = (day_counts / total_by_day * 100).fillna(0)

    ax = sns.barplot(x=error_rates_by_day.index, y=error_rates_by_day.values, palette='Blues')
    plt.title("Error Rate by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Error Rate (%)")
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Add count labels
    for i, v in enumerate(error_rates_by_day.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')

    plt.savefig(f"{os.path.splitext(filename)[0]}_by_days.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Volume analysis of errors
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(features_df['Volume'], color='blue', label='All data', kde=True, alpha=0.6)
    sns.histplot(wrong_predictions['Volume'], color='red', label='Wrong predictions', kde=True, alpha=0.6)
    plt.legend()
    plt.title("Volume Distribution - Errors vs Overall")
    plt.xlabel("Volume")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=[features_df['Volume'], wrong_predictions['Volume']],
                palette=['blue', 'red'])
    plt.xticks([0, 1], ['All Data', 'Errors'])
    plt.title("Volume Comparison")

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(filename)[0]}_volume.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Volatility analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(features_df['candle_range'], color='blue', label='All data', kde=True, alpha=0.6)
    sns.histplot(wrong_predictions['candle_range'], color='red', label='Wrong predictions', kde=True, alpha=0.6)
    plt.legend()
    plt.title("Candle Range Distribution")
    plt.xlabel("Candle Range")

    plt.subplot(1, 2, 2)
    if 'hourly_volatility' in features_df.columns and 'hourly_volatility' in wrong_predictions.columns:
        sns.histplot(features_df['hourly_volatility'], color='blue', label='All data', kde=True, alpha=0.6)
        sns.histplot(wrong_predictions['hourly_volatility'], color='red', label='Wrong predictions', kde=True, alpha=0.6)
        plt.legend()
        plt.title("Volatility Distribution")
        plt.xlabel("Hourly Volatility")

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(filename)[0]}_volatility.png", dpi=300, bbox_inches='tight')
    plt.close()

    # RSI analysis
    plt.figure(figsize=(12, 6))
    if 'RSI_14' in features_df.columns and 'RSI_14' in wrong_predictions.columns:
        sns.histplot(features_df['RSI_14'], color='blue', label='All data', kde=True, alpha=0.6, bins=20)
        sns.histplot(wrong_predictions['RSI_14'], color='red', label='Wrong predictions', kde=True, alpha=0.6, bins=20)
        plt.axvline(30, color='black', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.axvline(70, color='black', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.legend()
        plt.title("RSI Distribution")
        plt.xlabel("RSI Value")
        plt.xlim(0, 100)

    plt.savefig(f"{os.path.splitext(filename)[0]}_rsi.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Price action analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(features_df['Close'], color='blue', label='All data', kde=True, alpha=0.6)
    sns.histplot(wrong_predictions['Close'], color='red', label='Wrong predictions', kde=True, alpha=0.6)
    plt.legend()
    plt.title("Close Price Distribution")
    plt.xlabel("Close Price")

    plt.subplot(1, 2, 2)
    if 'log_return_5m' in features_df.columns and 'log_return_5m' in wrong_predictions.columns:
        sns.histplot(features_df['log_return_5m'], color='blue', label='All data', kde=True, alpha=0.6)
        sns.histplot(wrong_predictions['log_return_5m'], color='red', label='Wrong predictions', kde=True, alpha=0.6)
        plt.legend()
        plt.title("5-Min Return Distribution")
        plt.xlabel("Log Return")

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(filename)[0]}_price.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Feature correlation with errors
    error_indicator = np.zeros(len(features_df))
    error_indicator[wrong_indices] = 1

    features_with_errors = features_df.copy()
    features_with_errors['is_error'] = error_indicator


    # Select numerical columns for correlation
    # numeric_cols = features_with_errors.select_dtypes(include=['float64', 'int64']).columns
    # correlation_cols = [col for col in numeric_cols if col not in ['is_error', 'true_direction', 'predicted_direction', 'timestamp']]


    # N'utiliser que les colonnes qui sont dans selected_features pour calculer les corrélations
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
    correlation_cols = [col for col in numeric_cols if col in selected_features]
    correlation_cols = [col for col in correlation_cols if col not in ['is_error', 'true_direction', 'predicted_direction', 'timestamp']]


    # Calculate correlations
    correlations = {}
    for col in correlation_cols:
        if col in features_with_errors.columns:
            corr = features_with_errors[col].corr(features_with_errors['is_error'])
            correlations[col] = corr

    # Sort and plot top correlations
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(corr_df['Feature'], corr_df['Correlation'], color=corr_df['Correlation'].apply(
        lambda x: 'red' if x > 0 else 'green'))
    plt.title("Top 20 Feature Correlations with Prediction Errors")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Feature")
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)

    # Add correlation values
    for bar in bars:
        width = bar.get_width()
        label_x = width + 0.01 if width > 0 else width - 0.01
        plt.text(label_x, bar.get_y() + bar.get_height()/2,
                 f"{width:.3f}",
                 ha='left' if width > 0 else 'right',
                 va='center')

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(filename)[0]}_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed errors to CSV
    wrong_predictions.to_csv(f"{os.path.splitext(filename)[0]}.csv", index=True)
    # print(f"Detailed error analysis saved to img/{os.path.splitext(filename)[0]}*.png")

    return wrong_predictions


def obv(df):
    # Calculer l'OBV correctement
    df['OBV'] = 0
    # Différentes fenêtres pour l'OBV
    short_window = 5
    medium_window = 15
    long_window = 30

    # Calcul de l'OBV de base
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'Volume']
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'Volume']
        else:
            df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']

    # Normalisation de l'OBV de base
    # Méthode 1: Normalisation par le volume moyen - transformation en "équivalent jours de volume"
    mean_volume = df['Volume'].mean()
    if mean_volume > 0:  # Éviter la division par zéro
        df['OBV_normalized'] = df['OBV'] / mean_volume
    else:
        df['OBV_normalized'] = df['OBV']

    # Méthode 2: Normalisation Min-Max entre -1 et 1
    obv_min, obv_max = df['OBV'].min(), df['OBV'].max()
    if obv_max > obv_min:  # Éviter la division par zéro
        df['OBV_norm_minmax'] = 2 * (df['OBV'] - obv_min) / (obv_max - obv_min) - 1
    else:
        df['OBV_norm_minmax'] = 0

    # Calcul des moyennes mobiles avec différentes fenêtres (utilisant l'OBV normalisé)
    df['OBV_SMA_short'] = df['OBV_normalized'].rolling(window=short_window).mean()
    df['OBV_SMA_medium'] = df['OBV_normalized'].rolling(window=medium_window).mean()
    df['OBV_SMA_long'] = df['OBV_normalized'].rolling(window=long_window).mean()

    # Tendances pour différentes périodes
    df['OBV_Trend_short'] = 0
    df['OBV_Trend_medium'] = 0
    df['OBV_Trend_long'] = 0

    # Calculer les tendances pour chaque fenêtre
    for i in range(short_window, len(df)):
        if df.loc[df.index[i], 'OBV_SMA_short'] > df.loc[df.index[i-1], 'OBV_SMA_short'] * 1.001:
            df.loc[df.index[i], 'OBV_Trend_short'] = 1
        elif df.loc[df.index[i], 'OBV_SMA_short'] < df.loc[df.index[i-1], 'OBV_SMA_short'] * 0.999:
            df.loc[df.index[i], 'OBV_Trend_short'] = -1
        else:
            df.loc[df.index[i], 'OBV_Trend_short'] = 0

    for i in range(medium_window, len(df)):
        # Tendance moyen terme
        if df.loc[df.index[i], 'OBV_SMA_medium'] > df.loc[df.index[i-1], 'OBV_SMA_medium'] * 1.0008:
            df.loc[df.index[i], 'OBV_Trend_medium'] = 1
        elif df.loc[df.index[i], 'OBV_SMA_medium'] < df.loc[df.index[i-1], 'OBV_SMA_medium'] * 0.9992:
            df.loc[df.index[i], 'OBV_Trend_medium'] = -1
        else:
            df.loc[df.index[i], 'OBV_Trend_medium'] = 0

    for i in range(long_window, len(df)):
        # Tendance long terme
        if df.loc[df.index[i], 'OBV_SMA_long'] > df.loc[df.index[i-1], 'OBV_SMA_long'] * 1.0005:
            df.loc[df.index[i], 'OBV_Trend_long'] = 1
        elif df.loc[df.index[i], 'OBV_SMA_long'] < df.loc[df.index[i-1], 'OBV_SMA_long'] * 0.9995:
            df.loc[df.index[i], 'OBV_Trend_long'] = -1
        else:
            df.loc[df.index[i], 'OBV_Trend_long'] = 0

    # Feature composite pour la tendance globale
    df['OBV_Trend'] = df['OBV_Trend_short'] + df['OBV_Trend_medium'] + df['OBV_Trend_long']

    return df


def adl(df):
    """
    Calcule l'indicateur Accumulation/Distribution Line (ADL) complet avec dérivés.

    L'ADL est un indicateur de volume qui évalue la relation entre le prix et le volume
    pour confirmer les tendances de prix ou anticiper les renversements.
    """
    # Calcul du Money Flow Multiplier
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])

    # Remplacer les valeurs infinies et NaN qui se produisent quand High = Low
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Money Flow Volume = Money Flow Multiplier * Volume
    df['MFV'] = money_flow_multiplier * df['Volume']

    # ADL = Somme cumulative du Money Flow Volume
    df['ADL'] = df['MFV'].cumsum()

    # Normalisation de l'ADL pour faciliter l'interprétation
    adl_min, adl_max = df['ADL'].min(), df['ADL'].max()
    if adl_max > adl_min:  # Éviter la division par zéro
        df['ADL_norm'] = 2 * (df['ADL'] - adl_min) / (adl_max - adl_min) - 1
    else:
        df['ADL_norm'] = 0

    # Moyennes mobiles de l'ADL pour différentes périodes
    df['ADL_SMA5'] = df['ADL'].rolling(window=5).mean()
    df['ADL_SMA15'] = df['ADL'].rolling(window=15).mean()
    df['ADL_SMA30'] = df['ADL'].rolling(window=30).mean()

    # Tendance de l'ADL (court terme)
    df['ADL_Trend'] = 0
    for i in range(5, len(df)):
        if df['ADL_SMA5'].iloc[i] > df['ADL_SMA5'].iloc[i-1] * 1.0001:
            df.loc[df.index[i], 'ADL_Trend'] = 1  # Tendance haussière
        elif df['ADL_SMA5'].iloc[i] < df['ADL_SMA5'].iloc[i-1] * 0.9999:
            df.loc[df.index[i], 'ADL_Trend'] = -1  # Tendance baissière

    # Divergence entre ADL et prix (signaux importants)
    df['Price_Trend'] = 0
    df['ADL_Divergence'] = 0

    for i in range(5, len(df)):
        # Tendance du prix sur 5 périodes
        if df['Close'].iloc[i] > df['Close'].iloc[i-5] * 1.0005:  # Réduit de 0.5% à 0.05%
            df.loc[df.index[i], 'Price_Trend'] = 1
        elif df['Close'].iloc[i] < df['Close'].iloc[i-5] * 0.9995:  # Réduit de 0.5% à 0.05%
            df.loc[df.index[i], 'Price_Trend'] = -1

        # Divergence: tendance de prix différente de la tendance ADL
        if df['Price_Trend'].iloc[i] != 0 and df['ADL_Trend'].iloc[i] != 0:  # S'assurer qu'on a des tendances
            if df['Price_Trend'].iloc[i] == 1 and df['ADL_Trend'].iloc[i] == -1:
                df.loc[df.index[i], 'ADL_Divergence'] = -1  # Divergence baissière
            elif df['Price_Trend'].iloc[i] == -1 and df['ADL_Trend'].iloc[i] == 1:
                df.loc[df.index[i], 'ADL_Divergence'] = 1   # Divergence haussière

    print("Répartition des valeurs Price_Trend:", df['Price_Trend'].value_counts())
    print("Répartition des valeurs ADL_Trend:", df['ADL_Trend'].value_counts())
    print("Répartition des valeurs ADL_Divergence:", df['ADL_Divergence'].value_counts())

    return df


def pvt(df):
    """
    Calcule l'indicateur Price Volume Trend (PVT) et ses dérivés.
    Le PVT relie le changement de prix au volume, pondérant le volume par le pourcentage de variation du prix.
    """
    # Calcul du PVT de base
    df['PVT_change'] = df['Close'].pct_change().fillna(0) * df['Volume']
    df['PVT'] = df['PVT_change'].cumsum()

    # Normalisation du PVT pour une meilleure interprétation
    pvt_min, pvt_max = df['PVT'].min(), df['PVT'].max()
    if pvt_max > pvt_min:
        df['PVT_norm'] = 2 * (df['PVT'] - pvt_min) / (pvt_max - pvt_min) - 1
    else:
        df['PVT_norm'] = 0

    # Moyennes mobiles du PVT
    df['PVT_SMA10'] = df['PVT'].rolling(window=10, min_periods=1).mean()
    df['PVT_SMA30'] = df['PVT'].rolling(window=30, min_periods=1).mean()

    # Signal de tendance basé sur le croisement des moyennes mobiles
    df['PVT_signal'] = 0
    df.loc[df['PVT_SMA10'] > df['PVT_SMA30'], 'PVT_signal'] = 1
    df.loc[df['PVT_SMA10'] < df['PVT_SMA30'], 'PVT_signal'] = -1

    # Ajout d'un signal de croisement (1 si croisement haussier, -1 si baissier, 0 sinon)
    df['PVT_cross'] = df['PVT_SMA10'] - df['PVT_SMA30']
    df['PVT_cross_signal'] = df['PVT_cross'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Ajout d'une variation du PVT sur 5 périodes pour détecter les accélérations
    df['PVT_delta5'] = df['PVT'].diff(5).fillna(0)

    return df


def add_volume_indicators(df):

    df['Volume'] = df['Volume'].replace(0, 1e-6)
    # df['Volume'] = df['Volume'].fillna(df['Volume'].median())

    """Add volume-based technical indicators to the dataframe."""
    # Basic volume metrics
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

    # Volume relative to moving averages
    df['Volume_Ratio_SMA5'] = df['Volume'] / df['Volume_SMA_5']
    df['Volume_Ratio_SMA10'] = df['Volume'] / df['Volume_SMA_10']
    df['Volume_Ratio_SMA20'] = df['Volume'] / df['Volume_SMA_20']

    # Volume change rate
    df['Volume_Change_1'] = df['Volume'].pct_change(1)
    df['Volume_Change_5'] = df['Volume'].pct_change(5)

    df['Volume_Change_1'] = df['Volume_Change_1'].replace([np.inf, -np.inf], np.nan)
    df['Volume_Change_5'] = df['Volume_Change_5'].replace([np.inf, -np.inf], np.nan)

    # Optionnel : Remplacer les NaN par une valeur spécifique, comme 0, si nécessaire
    df['Volume_Change_1'] = df['Volume_Change_1'].fillna(0)
    df['Volume_Change_5'] = df['Volume_Change_5'].fillna(0)

    # Price-volume relationship
    df['PV_Ratio'] = df['Close'] * df['Volume']
    df['PV_Change'] = df['PV_Ratio'].pct_change()

    df = obv(df)

    # Accumulation/Distribution Line
    df = adl(df)

    # Chaikin Money Flow (period 20)
    df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['MFV'] = df['MFM'] * df['Volume']
    df['CMF_20'] = df['MFV'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Price-Volume Trend
    df = pvt(df)

    df['CCI_5'] = compute_cci(df, 5)
    df['CCI_10'] = compute_cci(df, 10)
    df['CCI_15'] = compute_cci(df, 15)
    # df['CCI_20'] = compute_cci(df, 20)
    # df['CCI_40'] = compute_cci(df, 40)
    # df['CCI_80'] = compute_cci(df, 80)

    #print(df[['Close', 'Volume', 'PVT']].head())
    #print("NaN dans 'Close':", df['Close'].isna().sum())
    #print("Inf dans 'Close':", np.isinf(df['Close']).sum())
    #print("NaN dans 'Volume':", df['Volume'].isna().sum())
    #print("Inf dans 'Volume':", np.isinf(df['Volume']).sum())
    #print("Inf dans 'PVT':", np.isinf(df['PVT']).sum())

    #threshold = 0.1  # Ajuste ce seuil en fonction de tes besoins
    # Compter les valeurs proches de zéro dans 'Close'
    #close_near_zero_rows = df[df['Close'].abs() < threshold]  # Ajuste le seuil selon tes besoins

    # Afficher le résultat
    #print(f"Nombre de valeurs proches de zéro dans 'Close' : {close_near_zero_rows}")
    # print(close_near_zero_rows[['Close', 'Volume', 'PVT']])


    # Volume Oscillator (difference between two volume moving averages)
    df['Volume_Oscillator'] = df['Volume_SMA_5'] - df['Volume_SMA_20']

    # Volume-Weighted Average Price (VWAP)
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df = pd.concat([df, pd.DataFrame({'VWAP': vwap})], axis=1)
    df = df.copy()

    # Volume-Weighted Moving Average (VWMA)
    df['VWMA_10'] = (df['Close'] * df['Volume']).rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
    df['VWMA_20'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Force Index (Elder)
    df['Force_Index_1'] = df['Close'].diff(1) * df['Volume']
    df['Force_Index_13'] = df['Force_Index_1'].ewm(span=13, adjust=False).mean()

    # Money Flow Index (MFI)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
    # Fix Money Flow calculation - avoid using shift in apply
    df['Typical_Price_Prev'] = df['Typical_Price'].shift(1)
    df['Money_Flow_Positive'] = np.where(df['Typical_Price'] > df['Typical_Price_Prev'], df['Raw_Money_Flow'], 0)
    df['Money_Flow_Negative'] = np.where(df['Typical_Price'] < df['Typical_Price_Prev'], df['Raw_Money_Flow'], 0)

    # Calculate MFI for 14 periods
    for i in range(14, len(df)):
        pos_flow = df['Money_Flow_Positive'].iloc[i-14:i].sum()
        neg_flow = df['Money_Flow_Negative'].iloc[i-14:i].sum()
        if neg_flow != 0:
            money_ratio = pos_flow / neg_flow
            df.loc[df.index[i], 'MFI_14'] = 100 - (100 / (1 + money_ratio))
        else:
            df.loc[df.index[i], 'MFI_14'] = 100

    # Volume Weighted RSI
    df['Vol_Weighted_Up'] = np.where(df['Close'] > df['Close'].shift(1), df['Close'] - df['Close'].shift(1), 0) * df['Volume']
    df['Vol_Weighted_Down'] = np.where(df['Close'] < df['Close'].shift(1), df['Close'].shift(1) - df['Close'], 0) * df['Volume']

    df['Vol_Weighted_Up_Avg'] = df['Vol_Weighted_Up'].rolling(window=14).mean()
    df['Vol_Weighted_Down_Avg'] = df['Vol_Weighted_Down'].rolling(window=14).mean()

    # Calculate Volume Weighted RSI
    df['Vol_Weighted_RSI'] = 100 - (100 / (1 + (df['Vol_Weighted_Up_Avg'] / df['Vol_Weighted_Down_Avg'])))


    # print(df[df['Volume'] == 0].head(10))
    # print(f"Nombre de volumes à zéro : {df['Volume'].eq(0).sum()}")
    # print(df['Volume'].value_counts())
    # print(df['Volume'].describe())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # print(df.isna().sum())  # Voir combien de NaN restent
    # print((df == np.inf).sum())  # Voir s'il reste des inf
    # print((df == -np.inf).sum())  # Voir s'il reste des -inf

    return df


def add_vwap_10(df):
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
    df['VWAP_10'] = df['Raw_Money_Flow'].rolling(window=10, min_periods=1).sum() / df['Volume'].rolling(window=10, min_periods=1).sum()
    return df


def load_timeframe_data(base_directory, timeframe):
    """
    Load data from a specific timeframe directory and format it consistently.

    Args:
        base_directory: Base directory path (e.g., 'seeds/commodities/brent')
        timeframe: Timeframe to load ('5min', '1h', '4h', '1d')

    Returns:
        DataFrame with data for the specified timeframe
    """
    directory_path = os.path.join(base_directory, timeframe)

    if not os.path.exists(directory_path):
        print(f"Warning: {directory_path} does not exist. Skipping this timeframe.")
        return pd.DataFrame()

    print(f"Loading data from {directory_path}...")

    # Special case for 1d data with new format
    if timeframe == '1d' and os.path.exists(os.path.join(directory_path, 'all_v2.json')):
        file_path = os.path.join(directory_path, 'all_v2.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

                if 'Interval' in data and data['Interval'] == 'OneDay' and 'Candles' in data:
                    all_candles = []
                    for instrument in data['Candles']:
                        all_candles.extend(instrument['Candles'])

                    df = pd.DataFrame(all_candles)
                    if not df.empty:
                        print(f"Successfully loaded {len(df)} daily candles with new format.")

                        # Convert time to datetime and sort
                        df['FromDate'] = pd.to_datetime(df['FromDate'])
                        df = df.sort_values('FromDate')

                        # Add a column to identify the timeframe
                        df['TimeFrame'] = timeframe

                        print(f"Loaded {len(df)} {timeframe} data points from {df['FromDate'].min()} to {df['FromDate'].max()}")
                        return df
        except Exception as e:
            print(f"Error loading 1d data with new format: {e}")

    # Default case: Use the same loading function we use for 5min data
    df = load_json_files_from_directory(directory_path)

    if df.empty:
        print(f"Warning: No data found in {directory_path}")
        return df

    # Convert time to datetime and sort
    df['FromDate'] = pd.to_datetime(df['FromDate'])
    df = df.sort_values('FromDate')

    # Add a column to identify the timeframe
    df['TimeFrame'] = timeframe

    print(f"Loaded {len(df)} {timeframe} data points from {df['FromDate'].min()} to {df['FromDate'].max()}")
    return df


def get_last_closed_row(df, delta, date):
    # On cherche la dernière bougie dont la fin de période < date
    # On suppose que 'FromDate' est le début de la période
    df = df[df['FromDate'] + delta < date]
    if df.empty:
        return None
    return df.iloc[-1]


# Pour chaque séquence 5min, associer dynamiquement les features 1h, 4h, 1d correspondant à la même période (par exemple, la dernière valeur 1h, 4h, 1d disponible à la fin de la séquence).
def get_multi_timeframe_features_for_sequence(seq_5min, df_1h, df_4h, df_1d):
    """
    Pour chaque séquence 5min, associe dynamiquement les features 1h, 4h, 1d correspondant à la même période,
    c'est-à-dire la dernière valeur 1h, 4h, 1d dont la période est complètement terminée
    (évite toute fuite d'information du futur).
    """
    last_5min_date = seq_5min['FromDate'].iloc[-1]

    # Définir la durée de chaque timeframe
    tf_deltas = {
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        '1d': pd.Timedelta(days=1)
    }

    row_1h = get_last_closed_row(df_1h, tf_deltas['1h'], last_5min_date)
    row_4h = get_last_closed_row(df_4h, tf_deltas['4h'], last_5min_date)
    row_1d = get_last_closed_row(df_1d, tf_deltas['1d'], last_5min_date)

    features = {}
    if row_1h is not None:
        features.update({f'1h_{col}': row_1h[col] for col in row_1h.index if col != 'FromDate'})
    if row_4h is not None:
        features.update({f'4h_{col}': row_4h[col] for col in row_4h.index if col != 'FromDate'})
    if row_1d is not None:
        features.update({f'1d_{col}': row_1d[col] for col in row_1d.index if col != 'FromDate'})

    return features


def add_multi_timeframe_features(df_5min, df_1h, df_4h, df_1d, base_directory='brent'):
    """
    Add features from multiple timeframes (1h, 4h, 1d) to provide broader market context.
    This helps the model understand both short and long-term trends.

    This version loads actual data from separate timeframe directories rather than
    approximating from 5min data.

    Args:
        df_5min: DataFrame with 5min data
        base_directory: Base directory containing timeframe subdirectories

    Returns:
        DataFrame with added multi-timeframe features
    """
    # Ensure the dataframe is sorted by time
    # df_5min = df_5min.sort_values('FromDate')

    # print(df_5min[['FromDate', 'Open', 'High', 'Low', 'Close', 'Volume']].head())


    # If any of the dataframes are empty, fall back to using the shift method
    if df_1h.empty or df_4h.empty or df_1d.empty:
        print("One or more timeframe datasets are empty. Falling back to using shift method.")
        return df_5min

    # For each 5min data point, find the most recent 1h, 4h, and 1d data points
    # We'll use merge_asof to efficiently join based on timestamps

    # 1h data matching
    df_5min = pd.merge_asof(
        df_5min.sort_values('FromDate'),
        df_1h[['FromDate', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('FromDate'),
        on='FromDate',
        direction='backward',
        suffixes=('', '_1h')
    )

    # Rename 1h columns
    df_5min['1h_Open'] = df_5min['Open_1h']
    df_5min['1h_High'] = df_5min['High_1h']
    df_5min['1h_Low'] = df_5min['Low_1h']
    df_5min['1h_Close'] = df_5min['Close_1h']
    df_5min['1h_Volume'] = df_5min['Volume_1h']

    # print(df_5min[['1h_Open', '1h_High', '1h_Low', '1h_Close', '1h_Volume']].head())

    # Drop the original suffixed columns
    df_5min = df_5min.drop(['Open_1h', 'High_1h', 'Low_1h', 'Close_1h', 'Volume_1h'], axis=1)

    # 4h data matching
    df_5min = pd.merge_asof(
        df_5min.sort_values('FromDate'),
        df_4h[['FromDate', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('FromDate'),
        on='FromDate',
        direction='backward',
        suffixes=('', '_4h')
    )

    # Rename 4h columns
    df_5min['4h_Open'] = df_5min['Open_4h']
    df_5min['4h_High'] = df_5min['High_4h']
    df_5min['4h_Low'] = df_5min['Low_4h']
    df_5min['4h_Close'] = df_5min['Close_4h']
    df_5min['4h_Volume'] = df_5min['Volume_4h']

    # print(df_5min[['4h_Open', '4h_High', '4h_Low', '4h_Close', '4h_Volume']].head())

    # Drop the original suffixed columns
    df_5min = df_5min.drop(['Open_4h', 'High_4h', 'Low_4h', 'Close_4h', 'Volume_4h'], axis=1)

    # 1d data matching
    df_5min = pd.merge_asof(
        df_5min.sort_values('FromDate'),
        df_1d[['FromDate', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('FromDate'),
        on='FromDate',
        direction='backward',
        suffixes=('', '_1d')
    )

    # Rename 1d columns
    df_5min['1d_Open'] = df_5min['Open_1d']
    df_5min['1d_High'] = df_5min['High_1d']
    df_5min['1d_Low'] = df_5min['Low_1d']
    df_5min['1d_Close'] = df_5min['Close_1d']
    df_5min['1d_Volume'] = df_5min['Volume_1d']

    print(df_5min[['1d_Open', '1d_High', '1d_Low', '1d_Close', '1d_Volume']].head())

    # Drop the original suffixed columns
    df_5min = df_5min.drop(['Open_1d', 'High_1d', 'Low_1d', 'Close_1d', 'Volume_1d'], axis=1)

    print("Multi-timeframe data successfully merged.")

    # Continue with additional calculations using the actual timeframe data

    # --- Price changes from previous timeframes ---
    # Change from 1-hour ago (percentage)
    df_5min['1h_price_change_pct'] = (df_5min['Close'] - df_5min['1h_Close']) / df_5min['1h_Close'].replace(0, np.nan) * 100

    print(df_5min[['1h_price_change_pct', 'Close', '1h_Close']].head())

    # Change from 4-hour ago (percentage)
    df_5min['4h_price_change_pct'] = (df_5min['Close'] - df_5min['4h_Close']) / df_5min['4h_Close'].replace(0, np.nan) * 100

    print(df_5min[['4h_price_change_pct', 'Close', '4h_Close']].head())

    # Change from 1-day ago (percentage)
    df_5min['1d_price_change_pct'] = (df_5min['Close'] - df_5min['1d_Close']) / df_5min['1d_Close'].replace(0, np.nan) * 100

    print(df_5min[['1d_price_change_pct', 'Close', '1d_Close']].head())

    # --- Price relation to previous timeframe ranges ---

    # Current price position relative to 1-hour range
    df_5min['1h_range'] = df_5min['1h_High'] - df_5min['1h_Low']
    df_5min['1h_position'] = (df_5min['Close'] - df_5min['1h_Low']) / df_5min['1h_range'].replace(0, np.nan)

    # Current price position relative to 4-hour range
    df_5min['4h_range'] = df_5min['4h_High'] - df_5min['4h_Low']
    df_5min['4h_position'] = (df_5min['Close'] - df_5min['4h_Low']) / df_5min['4h_range'].replace(0, np.nan)

    # Current price position relative to day range
    df_5min['1d_range'] = df_5min['1d_High'] - df_5min['1d_Low']
    df_5min['1d_position'] = (df_5min['Close'] - df_5min['1d_Low']) / df_5min['1d_range'].replace(0, np.nan)

    # --- Volume comparison across timeframes ---

    # Volume ratios with past timeframes
    df_5min['1h_volume_ratio'] = df_5min['Volume'] / df_5min['1h_Volume'].replace(0, np.nan)
    df_5min['4h_volume_ratio'] = df_5min['Volume'] / df_5min['4h_Volume'].replace(0, np.nan)
    df_5min['1d_volume_ratio'] = df_5min['Volume'] / df_5min['1d_Volume'].replace(0, np.nan)

    # --- Moving averages from multiple timeframes ---

    # Calculate Simple Moving Averages
    df_5min['1h_SMA'] = df_5min['1h_Close']  # Already a 1h average
    df_5min['4h_SMA'] = df_5min['4h_Close']  # Already a 4h average
    df_5min['1d_SMA'] = df_5min['1d_Close']  # Already a 1d average

    # Price relative to moving averages
    df_5min['close_over_1h_SMA'] = (df_5min['Close'] / df_5min['1h_SMA']).replace([np.inf, -np.inf], np.nan)
    df_5min['close_over_4h_SMA'] = (df_5min['Close'] / df_5min['4h_SMA']).replace([np.inf, -np.inf], np.nan)
    df_5min['close_over_1d_SMA'] = (df_5min['Close'] / df_5min['1d_SMA']).replace([np.inf, -np.inf], np.nan)

    # --- Trend direction over multiple timeframes ---

    # Direction of the trend (1 for uptrend, 0 for downtrend)
    df_5min['1h_trend'] = (df_5min['Close'] > df_5min['1h_Close']).astype(int)
    df_5min['4h_trend'] = (df_5min['Close'] > df_5min['4h_Close']).astype(int)
    df_5min['1d_trend'] = (df_5min['Close'] > df_5min['1d_Close']).astype(int)

    # --- Multi-timeframe trend agreement ---
    # Check if trends agree across timeframes (bullish alignment)
    df_5min['bullish_alignment'] = ((df_5min['1h_trend'] + df_5min['4h_trend'] + df_5min['1d_trend']) == 3).astype(int)

    # Check if trends agree across timeframes (bearish alignment)
    df_5min['bearish_alignment'] = ((df_5min['1h_trend'] + df_5min['4h_trend'] + df_5min['1d_trend']) == 0).astype(int)

    # Check for trend disagreement (mixed signals)
    df_5min['mixed_trend_signals'] = ((df_5min['1h_trend'] + df_5min['4h_trend'] + df_5min['1d_trend']) > 0).astype(int) & \
                                     ((df_5min['1h_trend'] + df_5min['4h_trend'] + df_5min['1d_trend']) < 3).astype(int)

    # --- Support and Resistance levels ---
    # Calculate support and resistance levels for each timeframe

    # For 1h timeframe
    df_5min = add_support_resistance_levels(df_5min, '1h', window=12)  # 12 hours window

    # For 4h timeframe
    df_5min = add_support_resistance_levels(df_5min, '4h', window=6)   # 24 hours window (6 x 4h)

    # For 1d timeframe
    df_5min = add_support_resistance_levels(df_5min, '1d', window=5)   # 5 days window

    # Calculate price proximity to support and resistance levels
    df_5min = calculate_price_to_level_proximity(df_5min)

    # ATR calculation not needed - we already have actual 1h, 4h, and 1d data
    # Replace any NaN values with 0 (using the new recommended approach)
    df_5min = df_5min.fillna(0).infer_objects(copy=False)

    # print(df_5min[['1h_trend','4h_trend', '1d_trend', 'bullish_alignment', 'bearish_alignment', 'mixed_trend_signals']].head())
    return df_5min



def detect_patterns_multi_tf(df, df_1h, df_4h, df_1d, timeframes=['5min', '1h', '4h', '1d']):
    """
    Applique la détection de double top et double bottom sur plusieurs timeframes et fusionne les résultats.

    Args:
        df (pd.DataFrame): Données contenant 'FromDate', 'High', 'Low', 'Close'.
        timeframes (list): Liste des timeframes à analyser (par défaut ['5min', '1h', '4h', '1d']).

    Returns:
        pd.DataFrame: DataFrame enrichi avec les signaux sur plusieurs timeframes.
    """
    # df = df.copy()  # On évite de modifier l'original

    last_row = get_last_closed_row(df, pd.Timedelta(minutes=5), df['FromDate'].iloc[-1])
    df_1h_limited = df_1h[df_1h['FromDate'] <= last_row['FromDate']] if last_row is not None else df_1h.iloc[0:0]
    df_4h_limited = df_4h[df_4h['FromDate'] <= last_row['FromDate']] if last_row is not None else df_4h.iloc[0:0]
    df_1d_limited = df_1d[df_1d['FromDate'] <= last_row['FromDate']] if last_row is not None else df_1d.iloc[0:0]

    dfs = {'5min': df, '1h': df_1h_limited, '4h': df_4h_limited, '1d': df_1d_limited}

    for tf in timeframes:
        df_tf = dfs[tf]
        results = detect_patterns(df_tf)
        if not results.empty:
            df[f'double_top_{tf}'] = results['double_top'].iloc[-1]
            df[f'double_top_value_{tf}'] = results['double_top_value'].iloc[-1] if 'double_top_value' in results else np.nan
            df[f'double_bottom_{tf}'] = results['double_bottom'].iloc[-1]
            df[f'double_bottom_value_{tf}'] = results['double_bottom_value'].iloc[-1] if 'double_bottom_value' in results else np.nan
            df[f'breakout_price_{tf}'] = results['breakout_price'].iloc[-1]
            # Ajout explicite de la ligne de résistance du double top si présente
            if 'double_top_line' in results:
                df[f'double_top_line_{tf}'] = results['double_top_line'].iloc[-1]
            else:
                df[f'double_top_line_{tf}'] = np.nan

            if 'double_bottom_line' in results:
                df[f'double_bottom_line_{tf}'] = results['double_bottom_line'].iloc[-1]
            else:
                df[f'double_bottom_line_{tf}'] = np.nan
        else:
            df[f'double_top_{tf}'] = False
            df[f'double_bottom_{tf}'] = False
            df[f'breakout_price_{tf}'] = False
            df[f'double_top_value_{tf}'] = np.nan
            df[f'double_bottom_value_{tf}'] = np.nan
            df[f'double_top_line_{tf}'] = np.nan
            df[f'double_bottom_line_{tf}'] = np.nan


    return df



def detect_patterns(df):
    """
    Détecte les patterns Double Top, Double Bottom et prédit les niveaux de cassure potentiels.
    Ajoute la ligne de résistance du double top (valeur du sommet) pour la dernière valeur trouvée.
    Args:
        df: DataFrame contenant les colonnes 'FromDate', 'High', 'Low', 'Close'
    Returns:
        DataFrame avec colonnes supplémentaires indiquant les patterns détectés, le niveau de cassure et la résistance.
    """

    # Détection des sommets et creux locaux
    df['prev_high'] = df['High'].shift(1)
    df['next_high'] = df['High'].shift(-1)
    df['is_peak'] = (df['High'] > df['prev_high']) & (df['High'] > df['next_high'])

    df['prev_low'] = df['Low'].shift(1)
    df['next_low'] = df['Low'].shift(-1)
    df['is_trough'] = (df['Low'] < df['prev_low']) & (df['Low'] < df['next_low'])

    # Listes pour stocker les résultats
    double_tops = []
    double_bottoms = []
    breakouts = []
    double_top_lines = []

    peaks = df[df['is_peak']].index
    troughs = df[df['is_trough']].index

    for i in range(2, len(peaks)):  # Parcourir les sommets
        first_peak = peaks[i-2]
        second_peak = peaks[i]

        # Vérifier si les sommets sont proches en prix (tolérance de 1%)
        if abs(df.loc[first_peak, 'High'] - df.loc[second_peak, 'High']) / df.loc[first_peak, 'High'] < 0.01:
            neckline = df.loc[first_peak:second_peak, 'Low'].min()
            double_tops.append((df.loc[second_peak, 'FromDate'], neckline))
            double_top_lines.append((df.loc[second_peak, 'FromDate'], df.loc[first_peak, 'High']))

            # Prédiction du breakdown après cassure du support
            target_price = neckline - (df.loc[first_peak, 'High'] - neckline)
            breakouts.append((df.loc[second_peak, 'FromDate'], 'Double Top Breakdown', target_price))

    for i in range(2, len(troughs)):  # Parcourir les creux
        first_trough = troughs[i-2]
        second_trough = troughs[i]

        # Vérifier si les creux sont proches en prix (tolérance de 1%)
        if abs(df.loc[first_trough, 'Low'] - df.loc[second_trough, 'Low']) / df.loc[first_trough, 'Low'] < 0.01:
            neckline = df.loc[first_trough:second_trough, 'High'].max()
            double_bottoms.append((df.loc[second_trough, 'FromDate'], neckline))

            # Prédiction du breakout après cassure de la résistance
            target_price = neckline + (neckline - df.loc[first_trough, 'Low'])
            breakouts.append((df.loc[second_trough, 'FromDate'], 'Double Bottom Breakout', target_price))

    # Ajouter les résultats au DataFrame
    df['double_top'] = df['FromDate'].isin([x[0] for x in double_tops])
    df['double_bottom'] = df['FromDate'].isin([x[0] for x in double_bottoms])
    df['double_top'] = df['double_top'].fillna(False).astype(bool)
    df['double_bottom'] = df['double_bottom'].fillna(False).astype(bool)

    # Initialisation propre de breakout_price avec NaN (float par défaut)
    df['breakout_price'] = np.nan

    # Initialisation de la ligne de résistance du double top
    df['double_top_line'] = np.nan
    df['double_bottom_line'] = np.nan

    # Ajout des valeurs de breakout_price et de la ligne de résistance
    for date, pattern, price in breakouts:
        df.loc[df['FromDate'] == date, 'breakout_price'] = float(price)
    for date, resistance in double_top_lines:
        df.loc[df['FromDate'] == date, 'double_top_line'] = float(resistance)
    for date, support in double_bottoms:
        df.loc[df['FromDate'] == date, 'double_bottom_line'] = float(support)

    # Création d'une colonne indicatrice (0 si NaN, 1 sinon)
    df['has_breakout'] = df['breakout_price'].notna().astype(int)

    # Vérification des types après modification
    print(df.dtypes)
    print(df[['breakout_price', 'double_top_line', 'double_bottom_line']].dropna().head())

    return df


# =========== MAIN FUNCTION ===========
def load_data(base_directory, df_1h, df_4h, df_1d):
    # Chemin vers le dossier contenant les fichiers JSON
    features_df = load_json_files_from_directory(base_directory + "/5min")
    # Charger les fichiers JSON et créer le DataFrame combiné
    features_df = preprocess_features(features_df)

    # Load data from other timeframes
    # print("Load features (1h, 4h, 1d)...")
    # df_1h = load_timeframe_data(base_directory, '1h')
    # df_4h = load_timeframe_data(base_directory, '4h')
    # df_1d = load_timeframe_data(base_directory, '1d')

    return create_parquet(features_df, df_1h, df_4h, df_1d)


def create_parquet(features_df, df_1h, df_4h, df_1d):
    print(f"Nombre de lignes dans create_parquet v0.0.0 : {len(features_df)}")
    print("Adding multi-timeframe features (1h, 4h, 1d)...")
    # features_df = add_multi_timeframe_features(features_df, df_1h, df_4h, df_1d)

    features_df = add_volume_indicators(features_df)

    # Filtrer les lignes avec un intervalle de 5 minutes
    features_df['prev_date'] = features_df['FromDate'].shift(1)
    features_df["timestamp"] = features_df['FromDate'].apply(lambda x: int(x.timestamp()))
    features_df['time_diff'] = (features_df['FromDate'] - features_df['prev_date']).dt.total_seconds() / 60

    features_df = add_time_columns(features_df)

    features_df[['market_open_hour', 'stock_open_hour', 'is_summer']] = features_df['FromDate'].apply(
        lambda x: pd.Series(get_market_opening(x))
    )

    # Calculer l'indicateur RSI
    features_df['RSI_14'] = calculate_rsi(features_df)

    # Moyenne mobile simple (SMA) du RSI sur 7 et 14 périodes
    features_df['RSI_SMA_7'] = features_df['RSI_14'].rolling(window=7).mean()
    features_df['RSI_SMA_14'] = features_df['RSI_14'].rolling(window=14).mean()

    features_df = volume_weighted_rsi_sma(features_df)

    # Moyenne mobile exponentielle (EMA) du RSI sur 7 et 14 périodes
    features_df['RSI_EMA_7'] = features_df['RSI_14'].ewm(span=7, adjust=False).mean()
    features_df['RSI_EMA_14'] = features_df['RSI_14'].ewm(span=14, adjust=False).mean()
    features_df['RSI_Trend'] = features_df['RSI_14'].diff()
    features_df['RSI_Trend_Direction'] = features_df['RSI_Trend'].apply(lambda x: 1 if x > 0 else 0)
    features_df['RSI_Crossover_SMA'] = (features_df['RSI_14'] > features_df['RSI_SMA_7']).astype(int)
    features_df['RSI_Crossover_EMA'] = (features_df['RSI_14'] > features_df['RSI_EMA_7']).astype(int)

    # Ajouter diverses caractéristiques
    features_df = add_future_direction(features_df)
    features_df = add_direction(features_df)
    features_df = add_candle_features(features_df)
    features_df = add_doji(features_df)
    features_df = add_candle_trend_relation(features_df)
    features_df = add_engulfing(features_df)
    features_df = add_wick_features(features_df)
    features_df = add_body_ratio(features_df)
    features_df = calculate_macd(features_df)

    features_df['body_ratio_prev'] = features_df['candle_range'] / features_df['candle_range'].shift(1).replace(0, np.nan)
    features_df['body_ratio_prev'] = features_df['body_ratio_prev'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ajouter des indicateurs techniques
    features_df['SMA_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['EMA_10'] = features_df['Close'].ewm(span=10, adjust=False).mean()

    print(f"Nombre de lignes dans features_df avant ATR : {len(features_df)}")
    features_df['ATR_14'] = ta.volatility.AverageTrueRange(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=14
    ).average_true_range()

    # features_df['ATR_50'] = ta.volatility.AverageTrueRange(
    #    high=features_df['High'],
    #    low=features_df['Low'],
    #    close=features_df['Close'],
    #    window=50
    # ).average_true_range()

    features_df['MACD'] = ta.trend.MACD(
        close=features_df['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    ).macd()

    bb = ta.volatility.BollingerBands(
        close=features_df['Close'],
        window=20,
        window_dev=2
    )
    features_df['Bollinger_High'] = bb.bollinger_hband()  # Bande supérieure
    features_df['Bollinger_Low'] = bb.bollinger_lband()   # Bande inférieure
    features_df['Bollinger_Width'] = bb.bollinger_wband() # Largeur de la bande

    supertrend_result = pta.overlap.supertrend(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=10,  # Période
        multiplier=3  # Facteur de volatilité
    )

    features_df['SuperTrend_Trend'] = supertrend_result['SUPERT_7_3.0']  # Trend
    features_df['SuperTrend_Direction'] = supertrend_result['SUPERTd_7_3.0']  # Direction
    features_df['SuperTrend_Long'] = supertrend_result['SUPERTl_7_3.0']  # Long
    features_df['SuperTrend_Short'] = supertrend_result['SUPERTs_7_3.0']  # Short

    features_df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(
        close=features_df['Close'],
        window=14,
        smooth1=3,
        smooth2=3
    ).stochrsi()

    print(f"Nombre de lignes dans create_parquet v0.1.0 : {len(features_df)}")
    kc = ta.volatility.KeltnerChannel(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=20
    )
    # print(f"Nombre de lignes dans create_parquet v0.1.1 : {len(features_df)}")
    # Après la ligne 1870
    middle_band = kc.keltner_channel_mband()
    # print("Middle Band sample:", middle_band.head(30)) # Afficher les premières valeurs
    # print("Is Middle Band zero?:", (middle_band == 0).sum()) # Compter les zéros
    # print("Is Middle Band NaN?:", middle_band.isna().sum()) # Compter les NaN

    features_df['Keltner_High'] = kc.keltner_channel_hband()
    features_df['Keltner_Low'] = kc.keltner_channel_lband()
    features_df['Keltner_Mid'] = middle_band # Stocker pour inspection
    features_df['Keltner_Width'] = kc.keltner_channel_wband()

    # Vérifier les lignes où Width est NaN mais High/Low ne le sont pas
    problematic_rows = features_df[features_df['Keltner_Width'].isna() & features_df['Keltner_High'].notna()]
    # print("Problematic Rows Sample:\n", problematic_rows[['Close', 'Keltner_High', 'Keltner_Low', 'Keltner_Mid', 'Keltner_Width']].head())

    # print(f"Nombre de lignes dans create_parquet v0.1.2 : {len(features_df)}")
    # print("features_df info:")
    # print(features_df.info())
    # print("features_df head:")
    # print(features_df.head(20))
    # print("features_df tail:")
    print(features_df.drop(['FromDate', 'time_diff', 'Open', 'High', 'Low', 'Close',  'Volume',
                            'date', 'prev_date', 'Volume_SMA_5', 'Volume_SMA_10', 'Stoch_RSI', 'Keltner_High',
                            'Volume_SMA_20', 'Volume_Ratio_SMA5', 'Volume_Ratio_SMA10', 'Volume_Ratio_SMA20',
                            'SuperTrend_Trend', 'SuperTrend_Direction',
                            'Volume_Change_1', 'Volume_Change_5', 'PV_Ratio', 'PV_Change', 'ATR_14',
                            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',  'MFM', 'MFV', 'CMF_20',
                            'CCI_5',
                            'MACD', 'MACD_Signal', 'body_ratio_prev', 'SMA_10', 'EMA_10',
                            'CCI_10', 'CCI_15',
                            'corps_candle_prev', 'corps_sum', 'ratio_corps', 'EMA_12',
                            'EMA_26', 'upper_wick', 'lower_wick', 'same_direction', 'candle_trend',
                            'meche_basse', 'meche_haute', 'corps_candle', 'candle_range', 'direction',
                            'future_direction_2', 'RSI_Crossover_EMA', 'RSI_Crossover_SMA', 'RSI_Trend_Direction',
                            'RSI_Trend', 'RSI_EMA_14', 'RSI_EMA_7', 'Vol_Weighted_RSI_SMA', 'Volume_Oscillator',
                            'SMA_RSI', 'RSI', 'VWAP', 'RSI_SMA_7', 'RSI_SMA_14', 'VWMA_10', 'is_summer', 'RSI_14',
                            'stock_open_hour', 'market_open_hour', 'VWMA_20', 'Force_Index_1', 'minute', 'day_of_week',
                            'Force_Index_13', 'day', 'hour', 'year', 'month', 'timestamp', 'Vol_Weighted_RSI',
                            'Vol_Weighted_Down_Avg', 'Vol_Weighted_Up_Avg', 'Vol_Weighted_Down', 'Vol_Weighted_Up',
                            'MFI_14', 'Money_Flow_Negative', 'Money_Flow_Positive', 'Typical_Price_Prev',
                            'Raw_Money_Flow', 'Typical_Price', 'OBV', 'OBV_normalized', 'OBV_norm_minmax',
                            'OBV_SMA_short', 'OBV_SMA_medium', 'OBV_SMA_long', 'OBV_Trend_short', 'OBV_Trend_medium',
                            'OBV_Trend_long', 'OBV_Trend',
                            'ADL', 'ADL_norm', 'ADL_SMA5', 'ADL_SMA15', 'ADL_SMA30', 'ADL_Trend',
                            'Price_Trend', 'ADL_Divergence',
                            'PVT_change', 'PVT', 'PVT_norm', 'PVT_SMA10', 'PVT_SMA30', 'PVT_signal', 'PVT_cross',
                            'PVT_cross_signal', 'PVT_delta5',
                            'doji_type', 'doji', 'doji_strength', 'perfect_doji', 'doji_invalid', 'pattern_type',
                            'bullish_engulfing', 'bearish_engulfing', 'engulfing_strength',
                            'Keltner_Low', 'Keltner_Mid', 'Keltner_Width'], axis=1).tail(20))
    features_df['ADX'] = ta.trend.adx(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=14
    )

    print(f"Nombre de lignes dans create_parquet v0.1.3 : {len(features_df)}")
    features_df['Williams_R'] = ta.momentum.williams_r(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        lbp=14
    )

    # Ajouter des caractéristiques cycliques
    features_df['day_of_year'] = features_df['FromDate'].dt.dayofyear
    features_df['sin_day'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['cos_day'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['period_of_day'] = features_df['FromDate'].apply(lambda x: get_period_of_day_with_timezone(x))
    features_df['hourly_return'] = features_df['Close'].pct_change()

    # Définir la taille de la fenêtre (par exemple, 24 pour une fenêtre de 24 heures)
    window = 24
    features_df['hourly_volatility'] = features_df['hourly_return'].rolling(window=window).std()

    # Calcul de la volatilité pour chaque période de la journée
    volatility_by_period = features_df.groupby('period_of_day')['hourly_return'].std().reset_index()
    volatility_by_period.rename(columns={'hourly_return': 'volatility_by_period'}, inplace=True)
    features_df = features_df.merge(volatility_by_period, on='period_of_day', how='left')

    # Ajouter des mesures de volatilité
    features_df['volatility_6h'] = features_df['hourly_return'].rolling(window=6).std()
    features_df['volatility_12h'] = features_df['hourly_return'].rolling(window=12).std()
    features_df['volatility_24h'] = features_df['hourly_return'].rolling(window=24).std()

    # Volatilité par période
    for period in range(4):  # Les périodes vont de 0 à 3
        features_df[f'volatility_period_{period}'] = features_df[features_df['period_of_day'] == period]['hourly_return'].rolling(window=6).std()

    # Volatilité par période nommée (non utilisée actuellement)
    for period in ['morning', 'afternoon', 'evening', 'night']:
        features_df[f'volatility_{period}'] = features_df[features_df['period_of_day'] == period]['hourly_return'].rolling(window=6).std()

    # Caractéristiques de retour logarithmique
    features_df['log_return_5m'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
    features_df['log_return_1h'] = np.log(features_df['Close'] / features_df['Close'].shift(12))
    features_df['log_return_4h'] = np.log(features_df['Close'] / features_df['Close'].shift(48))

    # Momentum
    momentum_window_5m = 1  # Momentum sur 5 minutes
    momentum_window_1h = 12  # Momentum sur 1 heure (12 bougies)
    momentum_window_4h = 48  # Momentum sur 4 heures (48 bougies)

    # Momentum à différentes périodes
    features_df[f'momentum_5m'] = features_df['Close'] - features_df['Close'].shift(momentum_window_5m)
    features_df[f'momentum_1h'] = features_df['Close'] - features_df['Close'].shift(momentum_window_1h)
    features_df[f'momentum_4h'] = features_df['Close'] - features_df['Close'].shift(momentum_window_4h)


    # Supprimer les lignes avec des valeurs manquantes pour certaines caractéristiques
    features_df = features_df.infer_objects()  # Infère les types des colonnes d'objets
    features_df = features_df.dropna(subset=['log_return_1h', 'log_return_4h', 'momentum_1h', 'momentum_4h'])
    features_df = features_df.fillna(0)  # Remplir les valeurs manquantes restantes


    # Money Flow Index (MFI)
    features_df = add_vwap_10(features_df)

    # Préparation finale des données
    #print(features_df[['log_return_5m', 'log_return_1h', 'log_return_4h', 'volatility_by_period']].head())
    #print(features_df[['momentum_5m', 'momentum_1h', 'momentum_4h', 'volatility_by_period']].head())

    features_df = features_df.drop(columns=["timestamp"], errors="ignore")

    # Convertir explicitement double_top_1H en bool s'il existe
    for col in [
        'double_top_1H', 'double_bottom_1H',
        'double_top_4H', 'double_bottom_4H',
        'double_top_1D', 'double_bottom_1D'
    ]:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(bool)

    return features_df


def assign_multi_timeframe_features(features):
    """Assigne dynamiquement les features multi-timeframes à la séquence."""
    return features


def main():

    print("Load features (1h, 4h, 1d)...")
    base_directory = "./brent"
    df_1h = load_timeframe_data(base_directory, '1h')
    df_4h = load_timeframe_data(base_directory, '4h')
    df_1d = load_timeframe_data(base_directory, '1d')

    cache_file = "./cache/features_cache.parquet"
    if os.path.exists(cache_file):
        print("🔄 Chargement des features depuis le cache...")
        features_df = pd.read_parquet(cache_file)
        # Sort by date to ensure deterministic order
        features_df = features_df.sort_values('FromDate')
        # features_df.to_parquet(cache_file, index=False)
        # features_df = pd.read_csv("features_cache.csv")  # Alternative si besoin CSV
    else:
        print("⚙️ Calcul des features...")
        features_df = load_data(base_directory, df_1h, df_4h, df_1d)
        print("⚙️ Sauvegarde des features...")
        features_df.to_parquet(cache_file, index=False)


    # Créer des séquences temporelles
    sequence_length = 16
    sequences = []

    # avant de pousser une séquence dans la liste des séquences, je voudrais vérifier que chaque feature poussée dans la séquence a bien un écart de 5 minutes avec la feature précédente puis que chaque séquence possède bien le sequence_length attendu
    # features_df['day_of_year'] = features_df['FromDate'].dt.dayofyear
    # features_df['sin_day'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    # features_df['cos_day'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    # features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    # features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    # features_df['period_of_day'] = features_df['FromDate'].apply(lambda x: get_period_of_day_with_timezone(x))
    # features_df['hourly_return'] = features_df['Close'].pct_change()
    # ici, j'ai des séquences de 16 données de la timeline de 5 minutes. Est ce qu'au lieu d'agréger les données 1h, 4h et 1 day dans add_multi_timeframe_features alors que j'ai que les 16 dernières features de la timeline 5minute, est ce qu'il ne faudrait pas associer à la séquence elle même les données des X dernières heures, X derniers jours, et calculer aussi day_of_year, sin_day, cos_day...

    # Oui, ta remarque est pertinente. Si tu ne conserves que les 16 dernières données 5min (soit 1h20), alors les features 1h, 4h, 1d ajoutées via add_multi_timeframe_features ne sont pertinentes que si elles correspondent exactement à la même période que ta séquence.
    # Sinon, tu risques d’introduire un décalage temporel ou d’utiliser des valeurs qui ne sont pas synchronisées avec ta séquence d’entrée.
    # Ce qu’il faudrait idéalement faire :
    # Pour chaque séquence 5min, associer dynamiquement les features 1h, 4h, 1d correspondant à la même période (par exemple, la dernière valeur 1h, 4h, 1d disponible à la fin de la séquence).
    # Calculer les features cycliques (day_of_year, sin_day, cos_day, etc.) pour chaque timestamp de la séquence.
    # Si tu veux enrichir la séquence avec des agrégats sur 1h, 4h, 1d, tu peux calculer ces agrégats (moyenne, min, max, etc.) sur la séquence elle-même ou sur les données brutes couvrant la période voulue.
    # En résumé :
    # Il vaut mieux associer les features multi-timeframes et cycliques à la séquence au moment de la création de la séquence, et non pas globalement sur tout le DataFrame.
    # Cela garantit que chaque séquence d’entrée au modèle contient des features cohérentes et synchronisées temporellement.
    # À faire :
    # Adapter la fonction de création de séquences pour inclure dynamiquement les features 1h, 4h, 1d et cycliques, calculées à partir des données couvrant la période de la séquence.
    # Éviter d’agréger globalement dans add_multi_timeframe_features si tu ne travailles que sur des séquences courtes.
    # Cela améliorera la pertinence temporelle de tes features pour la prédiction.
    for i in range(len(features_df) - sequence_length):
        seq = features_df.iloc[i:i+sequence_length]
        # Vérifie la longueur et l'écart de temps de 5 minutes entre chaque élément
        if len(seq) == sequence_length and all(
                (seq['FromDate'].iloc[j] - seq['FromDate'].iloc[j-1]).total_seconds() == 300
                for j in range(1, sequence_length)
        ):
            features = get_multi_timeframe_features_for_sequence(seq, df_1h, df_4h, df_1d)
            seq.assign(**features)
            # seq = assign_multi_timeframe_features(seq, features)
            seq = detect_patterns_multi_tf(seq, df_1h, df_4h, df_1d)
            sequences.append(seq)

    # Séparer les données en X (features) et y (labels)
    X = np.array([seq[0] for seq in sequences])
    y = np.array([seq[1] for seq in sequences])

    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=selectSeed)

    # Évaluer avec un classificateur de base
    #dummy_clf = DummyClassifier(strategy="most_frequent", random_state=selectSeed)
    #dummy_clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    #dummy_acc = dummy_clf.score(X_test.reshape(X_test.shape[0], -1), y_test)
    #print(f"Dummy accuracy: {dummy_acc:.2f}")

    # Construire et entraîner le modèle
    model, early_stopping = create_deep_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Calculer les poids des classes pour l'équilibrage
    # class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    # class_weight_dict = dict(zip(np.unique(y), class_weights))

    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        # class_weight=class_weight_dict,
        shuffle=True,  # Assure que le paramètre shuffle est explicite
        verbose=1      # Affiche la progression
    )

    # Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Prédire les étiquettes
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    # Analyser les erreurs
    analyze_errors(features_df, y_test, y_pred)

    # Calculer et afficher les métriques
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}, AUC: {auc_score:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.2f}")

    # Visualiser les résultats
    plot_training_history(history)
    # visualize_lstm_weights(model)
    # Sauvegarder le modèle
    # model.save('rnn_model.h5')

if __name__ == "__main__":
    main()