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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
from sklearn.dummy import DummyClassifier
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import ta
import pandas_ta as pta
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import pytz
from datetime import timedelta
import fastparquet
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
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad_tp = (tp - sma_tp).abs().rolling(window=period).mean()
    cci = (tp - sma_tp) / (0.015 * mad_tp)
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


def filter_by_time_interval(df, interval):
    df['prev_date'] = df['FromDate'].shift(1)
    df['time_diff'] = (df['FromDate'] - df['prev_date']).dt.total_seconds() / 60
    return df[df['time_diff'] == interval]


def preprocess_features(directory_path):
    features_df = load_json_files_from_directory(directory_path)
    features_df['FromDate'] = pd.to_datetime(features_df['FromDate'])
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
    df['corps_candle'] = np.abs(df['Close'] - df['Open']) / df['candle_range']
    df['meche_haute'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['candle_range']
    df['meche_basse'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['candle_range']
    df['candle_trend'] = (df['Close'] > df['Open']).astype(int)
    return df


def add_doji(df, tolerance=0.01):
    # Définir un Doji comme une bougie où la différence entre Open et Close est très faible par rapport à la plage
    df['doji'] = (abs(df['Close'] - df['Open']) / df['candle_range'] < tolerance).astype(int)
    return df


def add_candle_trend_relation(df):
    # Si la bougie actuelle et la bougie précédente sont dans la même direction
    df['same_direction'] = (df['candle_trend'] == df['candle_trend'].shift(1)).astype(int)
    return df


def add_engulfing(df):
    # Bougie haussière engloutissante : La bougie actuelle engloutit la bougie précédente (et est haussière)
    df['bullish_engulfing'] = (
            (df['Close'] > df['Open']) &  # La bougie actuelle est haussière
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # La bougie précédente est baissière
            (df['Open'] < df['Close'].shift(1)) &  # Le bas de la bougie actuelle est inférieur au haut de la bougie précédente
            (df['Close'] > df['Open'].shift(1)) &  # Le haut de la bougie actuelle est supérieur au bas de la bougie précédente
            (df['candle_range'] > df['candle_range'].shift(1)) &  # La bougie actuelle est plus grande que la précédente
            (df['High'] > df['High'].shift(1)) &  # Le haut de la bougie actuelle dépasse celui de la bougie précédente
            (df['Low'] < df['Low'].shift(1))  # Le bas de la bougie actuelle est inférieur à celui de la bougie précédente
    ).astype(int)

    # Bougie baissière engloutissante : La bougie actuelle engloutit la bougie précédente (et est baissière)
    df['bearish_engulfing'] = (
            (df['Close'] < df['Open']) &  # La bougie actuelle est baissière
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # La bougie précédente est haussière
            (df['Open'] > df['Close'].shift(1)) &  # Le bas de la bougie actuelle est supérieur au haut de la bougie précédente
            (df['Close'] < df['Open'].shift(1)) &  # Le haut de la bougie actuelle est inférieur au bas de la bougie précédente
            (df['candle_range'] > df['candle_range'].shift(1)) &  # La bougie actuelle est plus grande que la précédente
            (df['High'] < df['High'].shift(1)) &  # Le haut de la bougie actuelle est inférieur à celui de la bougie précédente
            (df['Low'] > df['Low'].shift(1))  # Le bas de la bougie actuelle est supérieur à celui de la bougie précédente
    ).astype(int)

    return df


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

    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[1:, 'OBV'] = ((df['Close'].diff() > 0) * 2 - 1) * df['Volume']
    df['OBV'] = df['OBV'].cumsum()

    # Accumulation/Distribution Line
    df['ADL'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['ADL'] = df['ADL'].cumsum()

    # Chaikin Money Flow (period 20)
    df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['MFV'] = df['MFM'] * df['Volume']
    df['CMF_20'] = df['MFV'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Price-Volume Trend
    df['PVT'] = df['Close'].pct_change() * df['Volume']
    df['PVT'] = df['PVT'].cumsum()
    df['PVT'] = df['PVT'].clip(-0.1, 0.1)
    df['PVT'] = df['PVT'].replace([np.inf, -np.inf], np.nan)


    df['CCI_5'] = compute_cci(df, 5)
    df['CCI_10'] = compute_cci(df, 10)
    df['CCI_20'] = compute_cci(df, 20)
    df['CCI_40'] = compute_cci(df, 40)
    df['CCI_80'] = compute_cci(df, 80)

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
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

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


    print(df[df['Volume'] == 0].head(10))
    print(f"Nombre de volumes à zéro : {df['Volume'].eq(0).sum()}")
    print(df['Volume'].value_counts())
    print(df['Volume'].describe())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print(df.isna().sum())  # Voir combien de NaN restent
    print((df == np.inf).sum())  # Voir s'il reste des inf
    print((df == -np.inf).sum())  # Voir s'il reste des -inf

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


def add_multi_timeframe_features(df_5min, base_directory='brent'):
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
    df_5min = df_5min.sort_values('FromDate')


    print(df_5min[['FromDate', 'Open', 'High', 'Low', 'Close', 'Volume']].head())

    # Load data from other timeframes
    df_1h = load_timeframe_data(base_directory, '1h')
    df_4h = load_timeframe_data(base_directory, '4h')
    df_1d = load_timeframe_data(base_directory, '1d')

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



def detect_patterns_multi_tf(df, timeframes=['5T', '1H', '4H', '1D']):
    """
    Applique la détection de double top et double bottom sur plusieurs timeframes et fusionne les résultats.

    Args:
        df (pd.DataFrame): Données contenant 'FromDate', 'High', 'Low', 'Close'.
        timeframes (list): Liste des timeframes à analyser (par défaut ['5T', '1H', '4H', '1D']).

    Returns:
        pd.DataFrame: DataFrame enrichi avec les signaux sur plusieurs timeframes.
    """

    df = df.copy()  # On évite de modifier l'original

    for tf in timeframes:
        # Resample les données selon le timeframe
        df_tf = df.resample(tf, on='FromDate').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna().reset_index()

        # Applique la détection des patterns
        df_tf = detect_patterns(df_tf)

        # Renomme les colonnes pour éviter les conflits
        df_tf = df_tf.rename(columns={
            'double_top': f'double_top_{tf}',
            'double_bottom': f'double_bottom_{tf}',
            'breakout_price': f'breakout_price_{tf}'
        })

        df_tf[f'double_top_{tf}'] = df_tf[f'double_top_{tf}'].astype(bool)
        df_tf[f'double_bottom_{tf}'] = df_tf[f'double_bottom_{tf}'].astype(bool)

        # Fusion avec le dataframe original (en associant aux timestamps les plus proches)
        df = df.merge(df_tf[['FromDate', f'double_top_{tf}', f'double_bottom_{tf}', f'breakout_price_{tf}']],
                      on='FromDate', how='left')

    return df



def detect_patterns(df):
    """
    Détecte les patterns Double Top, Double Bottom et prédit les niveaux de cassure potentiels.

    Args:
        df: DataFrame contenant les colonnes 'FromDate', 'High', 'Low', 'Close'

    Returns:
        DataFrame avec colonnes supplémentaires indiquant les patterns détectés et le niveau de cassure.
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

    peaks = df[df['is_peak']].index
    troughs = df[df['is_trough']].index

    for i in range(2, len(peaks)):  # Parcourir les sommets
        first_peak = peaks[i-2]
        second_peak = peaks[i]

        # Vérifier si les sommets sont proches en prix (tolérance de 1%)
        if abs(df.loc[first_peak, 'High'] - df.loc[second_peak, 'High']) / df.loc[first_peak, 'High'] < 0.01:
            neckline = df.loc[first_peak:second_peak, 'Low'].min()
            double_tops.append((df.loc[second_peak, 'FromDate'], neckline))

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

    # Création d'une colonne indicatrice (0 si NaN, 1 sinon)
    df['has_breakout'] = df['breakout_price'].notna().astype(int)

    # Ajout des valeurs de breakout_price basées sur breakouts
    for date, pattern, price in breakouts:
        df.loc[df['FromDate'] == date, 'breakout_price'] = float(price)

    # Vérification des types après modification
    print(df.dtypes)
    print(df['breakout_price'].dropna().head())

    return df


# =========== MAIN FUNCTION ===========

def createdata(cache_file):
    # Chemin vers le dossier contenant les fichiers JSON
    directory_path = "./brent/5min"

    # Charger les fichiers JSON et créer le DataFrame combiné
    features_df = preprocess_features(directory_path)

    # Convertir 'FromDate' en datetime
    features_df['FromDate'] = pd.to_datetime(features_df['FromDate'])

    # Triez les données par 'FromDate'
    features_df = features_df.sort_values(by='FromDate')
    features_df = add_volume_indicators(features_df)

    print("Adding multi-timeframe features (1h, 4h, 1d)...")
    features_df = add_multi_timeframe_features(features_df)
    features_df = detect_patterns_multi_tf(features_df)

    # Filtrer les lignes avec un intervalle de 5 minutes
    features_df['prev_date'] = features_df['FromDate'].shift(1)
    features_df["timestamp"] = features_df['FromDate'].apply(lambda x: int(x.timestamp()))

    features_df['time_diff'] = (features_df['FromDate'] - features_df['prev_date']).dt.total_seconds() / 60
    features_df = features_df[features_df['time_diff'] == 5]

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

    features_df['body_ratio_prev'] = features_df['candle_range'] / features_df['candle_range'].shift(1)

    # Ajouter des indicateurs techniques
    features_df['SMA_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['EMA_10'] = features_df['Close'].ewm(span=10, adjust=False).mean()

    features_df['ATR_14'] = ta.volatility.AverageTrueRange(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=14
    ).average_true_range()

    features_df['ATR_50'] = ta.volatility.AverageTrueRange(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=50
    ).average_true_range()

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

    kc = ta.volatility.KeltnerChannel(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=20
    )
    features_df['Keltner_High'] = kc.keltner_channel_hband()
    features_df['Keltner_Low'] = kc.keltner_channel_lband()
    features_df['Keltner_Width'] = kc.keltner_channel_wband()

    features_df['ADX'] = ta.trend.adx(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        window=14
    )

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

    # Sauvegarde les features après leur calcul
    features_df.to_parquet(cache_file, index=False)  # Format plus efficace que CSV
    return features_df


def main():
    cache_file = "./cache/features_cache.parquet"
    if os.path.exists(cache_file):
        print("🔄 Chargement des features depuis le cache...")
        features_df = pd.read_parquet(cache_file)
        # features_df.to_parquet(cache_file, index=False)
        # features_df = pd.read_csv("features_cache.csv")  # Alternative si besoin CSV
    else:
        print("⚙️ Calcul des features...")
        features_df = createdata(cache_file)

    # Sort by date to ensure deterministic order
    features_df = features_df.sort_values('FromDate')

    # Créer des séquences temporelles
    sequence_length = 16
    sequences = []

    for i in range(len(features_df) - sequence_length):
        seq_x = features_df[selected_features].iloc[i:i + sequence_length].values
        seq_y = features_df['future_direction_2'].iloc[i + sequence_length]  # Direction du prochain point
        sequences.append((seq_x, seq_y))

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