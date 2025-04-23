#!/usr/bin/env python3
"""
Service WebSocket pour les prédictions de marché en temps réel.
Reçoit les données de cours sur différentes timeframes (5min, 1h, 4h, 1d),
applique les transformations du modèle RNN et génère des prédictions.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import pytz
import websockets
import asyncio
import logging
from sklearn.preprocessing import StandardScaler
import ta
from tensorflow.keras.models import load_model
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('prediction_service')

# Paramètres du modèle
MODEL_PATH = "best_model_seed_64.h5"  # Chemin vers le modèle entraîné
SEQUENCE_LENGTH = 24  # Longueur des séquences pour le modèle LSTM
HOST = "0.0.0.0"  # Hôte pour le WebSocket
PORT = 8765  # Port pour le WebSocket

# Liste des features utilisées par le modèle
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
    'OBV', 'ADL', 'MFM', 'CMF_20', 'PVT', 'Volume_Oscillator', 'Force_Index_1', 'Force_Index_13', 'Typical_Price', 'Raw_Money_Flow',
    'VWAP_10', '1h_price_change_pct', '4h_price_change_pct', '1d_price_change_pct', '1h_range', '1h_position',
    '4h_range', '4h_position', '1d_range', '1d_position', '1h_volume_ratio', '4h_volume_ratio', '1d_volume_ratio',
    'close_over_1h_SMA', 'close_over_4h_SMA', 'close_over_1d_SMA', '1h_trend', '4h_trend', '1d_trend',
    'bullish_alignment', 'bearish_alignment', 'mixed_trend_signals'
]

# Cache pour stocker les données historiques
historical_data = {
    '5min': [],
    '1h': [],
    '4h': [],
    '1d': []
}

# Scaler pour la normalisation des données
scaler = StandardScaler()

def compute_cci(df, period=20):
    """Calcule l'indicateur CCI (Commodity Channel Index)."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad_tp = (tp - sma_tp).abs().rolling(window=period).mean()
    cci = (tp - sma_tp) / (0.015 * mad_tp)
    return cci

def add_time_columns(df):
    """Ajoute les colonnes temporelles au DataFrame."""
    df['year'] = df['FromDate'].dt.year
    df['month'] = df['FromDate'].dt.month
    df['day'] = df['FromDate'].dt.day
    df['hour'] = df['FromDate'].dt.hour
    df['minute'] = df['FromDate'].dt.minute
    df['day_of_week'] = df['FromDate'].dt.dayofweek
    
    # Ajouter des transformations cycliques pour le jour et l'heure
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def get_market_opening(date):
    """
    Détermine l'heure d'ouverture de la bourse et si c'est l'horaire d'hiver ou d'été.
    """
    ny_tz = pytz.timezone("America/New_York")
    market_open_winter = 15  # 9h00 AM heure de New York (hiver, heure de Paris)
    market_open_summer = 14  # 8h00 AM heure de New York (été, heure de Paris)
    stock_open_winter = 15.5  # 9h30 AM NY (hiver, 15h30 heure de Paris)
    stock_open_summer = 14.5  # 8h30 AM NY (été, 14h30 heure de Paris)

    # Déterminer si la date est en heure d'été ou d'hiver
    localized_date = ny_tz.localize(datetime(date.year, date.month, date.day))
    is_summer = localized_date.dst() != timedelta(0)

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

def calculate_rsi(data, window=14):
    """Calcule l'indicateur RSI (Relative Strength Index)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_candle_features(df):
    """Ajoute les caractéristiques des bougies au DataFrame."""
    df['candle_range'] = df['High'] - df['Low']
    df['corps_candle'] = np.abs(df['Close'] - df['Open']) / df['candle_range']
    df['meche_haute'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['candle_range']
    df['meche_basse'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['candle_range']
    df['candle_trend'] = (df['Close'] > df['Open']).astype(int)
    return df

def add_wick_features(df):
    """Ajoute les caractéristiques des mèches des bougies."""
    df['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['candle_range']
    df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['candle_range']
    return df

def add_body_ratio(df):
    """Ajoute le rapport entre le corps de la bougie et le corps de la bougie précédente."""
    df['corps_candle_prev'] = df['corps_candle'].shift(1)
    df['corps_sum'] = df['corps_candle'] + df['corps_candle_prev']
    df['ratio_corps'] = df['corps_candle'] / df['corps_sum']
    return df

def add_volume_indicators(df):
    """Ajoute les indicateurs de volume au DataFrame."""
    # Volume moving averages
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Volume ratios
    df['Volume_Ratio_SMA5'] = df['Volume'] / df['Volume_SMA_5']
    df['Volume_Ratio_SMA10'] = df['Volume'] / df['Volume_SMA_10']
    df['Volume_Ratio_SMA20'] = df['Volume'] / df['Volume_SMA_20']
    
    # Volume changes
    df['Volume_Change_1'] = df['Volume'] / df['Volume'].shift(1) - 1
    df['Volume_Change_5'] = df['Volume'] / df['Volume'].shift(5) - 1
    
    # On Balance Volume (OBV)
    df['OBV'] = np.nan
    df.loc[0, 'OBV'] = df.loc[0, 'Volume']
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i-1, 'Close']:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] + df.loc[i, 'Volume']
        elif df.loc[i, 'Close'] < df.loc[i-1, 'Close']:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] - df.loc[i, 'Volume']
        else:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV']
    
    # Accumulation/Distribution Line (ADL)
    df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['MFM'] = df['MFM'].replace([np.inf, -np.inf], 0)
    df['MFV'] = df['MFM'] * df['Volume']
    df['ADL'] = df['MFV'].cumsum()
    
    # Chaikin Money Flow (CMF)
    df['CMF_20'] = df['MFV'].rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Price Volume Trend (PVT)
    df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
    
    # Volume Oscillator
    df['Volume_Oscillator'] = (df['Volume_SMA_5'] - df['Volume_SMA_20']) / df['Volume_SMA_20']
    
    # Force Index
    df['Force_Index_1'] = df['Close'].diff(1) * df['Volume']
    df['Force_Index_13'] = df['Force_Index_1'].ewm(span=13, adjust=False).mean()
    
    # Money Flow calculations
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
    df['Typical_Price_Prev'] = df['Typical_Price'].shift(1)
    df['Money_Flow_Positive'] = ((df['Typical_Price'] > df['Typical_Price_Prev']) * df['Raw_Money_Flow']).fillna(0)
    
    # Volume-Weighted indicators
    price_up = df['Close'] > df['Close'].shift(1)
    price_down = df['Close'] < df['Close'].shift(1)
    
    df['Vol_Weighted_Up'] = df.loc[price_up, 'Volume'].rolling(window=14, min_periods=1).sum()
    df['Vol_Weighted_Down'] = df.loc[price_down, 'Volume'].rolling(window=14, min_periods=1).sum()
    
    # Calculate average daily volume-weighted down volume
    df['Vol_Weighted_Down_Avg'] = df['Vol_Weighted_Down'].rolling(window=14, min_periods=1).mean()
    
    # Calculate Volume-Weighted RSI
    df['Vol_Weighted_RSI'] = 100 - (100 / (1 + df['Vol_Weighted_Up'] / df['Vol_Weighted_Down']))
    
    # Volume-Weighted RSI SMA
    df['Vol_Weighted_RSI_SMA'] = df['Vol_Weighted_RSI'].rolling(window=10, min_periods=1).mean()
    
    # VWAP calculations
    df['VWAP_10'] = (df['Typical_Price'] * df['Volume']).rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
    
    # VWMA calculations
    df['VWMA_10'] = (df['Close'] * df['Volume']).rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
    df['VWMA_20'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    # Price-Volume Ratio and Change
    df['PV_Ratio'] = df['Close'] / df['Volume']
    df['PV_Change'] = df['PV_Ratio'].pct_change()
    
    return df

def add_technical_indicators(df):
    """Ajoute les indicateurs techniques au DataFrame."""
    # Moyennes mobiles
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI et ses dérivés
    df['RSI_14'] = calculate_rsi(df, window=14)
    df['RSI_SMA_7'] = df['RSI_14'].rolling(window=7).mean()
    df['RSI_Trend'] = (df['RSI_14'] > df['RSI_SMA_7']).astype(int)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bandes de Bollinger
    df['Bollinger_Middle'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_STD'] = df['Close'].rolling(window=20).std()
    df['Bollinger_High'] = df['Bollinger_Middle'] + (df['Bollinger_STD'] * 2)
    df['Bollinger_Low'] = df['Bollinger_Middle'] - (df['Bollinger_STD'] * 2)
    df['Bollinger_Width'] = (df['Bollinger_High'] - df['Bollinger_Low']) / df['Bollinger_Middle']
    
    # ADX (Average Directional Index)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    
    # Stochastic RSI
    df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close'], window=14, smooth1=3, smooth2=3)
    
    # Williams %R
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    
    # Keltner Channel
    df['Keltner_Middle'] = df['Close'].rolling(window=20).mean()
    df['Keltner_High'] = df['Keltner_Middle'] + (df['ATR_14'] * 2)
    df['Keltner_Low'] = df['Keltner_Middle'] - (df['ATR_14'] * 2)
    df['Keltner_Width'] = (df['Keltner_High'] - df['Keltner_Low']) / df['Keltner_Middle']
    
    # SuperTrend
    factor = 3
    atr = df['ATR_14']
    
    df['upperband'] = ((df['High'] + df['Low']) / 2) + (factor * atr)
    df['lowerband'] = ((df['High'] + df['Low']) / 2) - (factor * atr)
    df['in_uptrend'] = True
    
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        
        # SuperTrend calculation
        if df['in_uptrend'].iloc[i-1]:
            # In uptrend
            if current_close < df['lowerband'].iloc[i-1]:
                df.loc[df.index[i], 'in_uptrend'] = False
            else:
                df.loc[df.index[i], 'in_uptrend'] = True
                
            # Adjust bands
            if df['in_uptrend'].iloc[i]:
                df.loc[df.index[i], 'lowerband'] = max(df['lowerband'].iloc[i], df['lowerband'].iloc[i-1])
        else:
            # In downtrend
            if current_close > df['upperband'].iloc[i-1]:
                df.loc[df.index[i], 'in_uptrend'] = True
            else:
                df.loc[df.index[i], 'in_uptrend'] = False
                
            # Adjust bands
            if not df['in_uptrend'].iloc[i]:
                df.loc[df.index[i], 'upperband'] = min(df['upperband'].iloc[i], df['upperband'].iloc[i-1])
    
    df['SuperTrend_Trend'] = df['in_uptrend'].astype(int)
    df['SuperTrend_Long'] = ((df['SuperTrend_Trend'] == 1) & (df['SuperTrend_Trend'].shift(1) == 0)).astype(int)
    df['SuperTrend_Short'] = ((df['SuperTrend_Trend'] == 0) & (df['SuperTrend_Trend'].shift(1) == 1)).astype(int)
    
    # CCI (Commodity Channel Index)
    df['CCI_5'] = compute_cci(df, period=5)
    df['CCI_10'] = compute_cci(df, period=10)
    df['CCI_20'] = compute_cci(df, period=20)
    df['CCI_40'] = compute_cci(df, period=40)
    df['CCI_80'] = compute_cci(df, period=80)
    
    # Momentum et Log Returns
    df['momentum_5m'] = df['Close'] / df['Close'].shift(1) - 1
    df['momentum_1h'] = df['Close'] / df['Close'].shift(12) - 1  # 12 * 5min = 1h
    df['momentum_4h'] = df['Close'] / df['Close'].shift(48) - 1  # 48 * 5min = 4h
    
    df['log_return_5m'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_1h'] = np.log(df['Close'] / df['Close'].shift(12))
    df['log_return_4h'] = np.log(df['Close'] / df['Close'].shift(48))
    
    # Volatilité
    df['volatility_by_period'] = df['log_return_5m'].rolling(window=20).std() * np.sqrt(252 * 78)  # Annualized volatility
    df['hourly_volatility'] = df['log_return_1h'].rolling(window=20).std() * np.sqrt(252 * 6.5)
    df['volatility_6h'] = df['log_return_5m'].rolling(window=72).std() * np.sqrt(252 * 78)
    df['volatility_12h'] = df['log_return_5m'].rolling(window=144).std() * np.sqrt(252 * 78)
    
    # Volatilité par période de la journée
    df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])
    for i in range(4):
        period_returns = df[df['hour_bin'] == i]['log_return_5m']
        if not period_returns.empty:
            volatility = period_returns.rolling(window=20).std().mean() * np.sqrt(252 * 78)
            df[f'volatility_period_{i}'] = volatility
        else:
            df[f'volatility_period_{i}'] = np.nan
    
    # Hourly return
    df['hourly_return'] = df['Close'].pct_change(12)
    
    return df

def add_multi_timeframe_features(df_5min, df_1h, df_4h, df_1d):
    """
    Ajoute les caractéristiques multi-timeframes au DataFrame 5 minutes.
    
    Args:
        df_5min: DataFrame avec les données 5 minutes
        df_1h: DataFrame avec les données 1 heure
        df_4h: DataFrame avec les données 4 heures
        df_1d: DataFrame avec les données 1 jour
    
    Returns:
        DataFrame avec les caractéristiques multi-timeframes
    """
    # Fonction pour trouver la correspondance la plus proche dans le temps
    def find_closest_timeframe_data(df_main, df_other, timestamp):
        if df_other.empty:
            return pd.Series([np.nan] * len(df_other.columns), index=df_other.columns)
        
        closest_idx = (df_other['FromDate'] - timestamp).abs().idxmin()
        return df_other.loc[closest_idx]
    
    # Pour chaque ligne du DataFrame 5min, ajoutez les données des autres timeframes
    for i, row in df_5min.iterrows():
        timestamp = row['FromDate']
        
        # Obtenir les données 1h les plus proches
        if not df_1h.empty:
            closest_1h = find_closest_timeframe_data(df_5min, df_1h, timestamp)
            
            # Calculer les features 1h
            df_5min.at[i, '1h_price_change_pct'] = (row['Close'] - closest_1h['Close']) / closest_1h['Close']
            df_5min.at[i, '1h_range'] = (closest_1h['High'] - closest_1h['Low']) / closest_1h['Close']
            df_5min.at[i, '1h_position'] = (row['Close'] - closest_1h['Low']) / (closest_1h['High'] - closest_1h['Low']) if closest_1h['High'] != closest_1h['Low'] else 0.5
            df_5min.at[i, '1h_volume_ratio'] = row['Volume'] / closest_1h['Volume'] if closest_1h['Volume'] > 0 else 1
            df_5min.at[i, 'close_over_1h_SMA'] = row['Close'] > closest_1h['SMA_10'] if 'SMA_10' in closest_1h else np.nan
            df_5min.at[i, '1h_trend'] = 1 if closest_1h['Close'] > closest_1h['Open'] else 0
        
        # Obtenir les données 4h les plus proches
        if not df_4h.empty:
            closest_4h = find_closest_timeframe_data(df_5min, df_4h, timestamp)
            
            # Calculer les features 4h
            df_5min.at[i, '4h_price_change_pct'] = (row['Close'] - closest_4h['Close']) / closest_4h['Close']
            df_5min.at[i, '4h_range'] = (closest_4h['High'] - closest_4h['Low']) / closest_4h['Close']
            df_5min.at[i, '4h_position'] = (row['Close'] - closest_4h['Low']) / (closest_4h['High'] - closest_4h['Low']) if closest_4h['High'] != closest_4h['Low'] else 0.5
            df_5min.at[i, '4h_volume_ratio'] = row['Volume'] / closest_4h['Volume'] if closest_4h['Volume'] > 0 else 1
            df_5min.at[i, 'close_over_4h_SMA'] = row['Close'] > closest_4h['SMA_10'] if 'SMA_10' in closest_4h else np.nan
            df_5min.at[i, '4h_trend'] = 1 if closest_4h['Close'] > closest_4h['Open'] else 0
        
        # Obtenir les données 1d les plus proches
        if not df_1d.empty:
            closest_1d = find_closest_timeframe_data(df_5min, df_1d, timestamp)
            
            # Calculer les features 1d
            df_5min.at[i, '1d_price_change_pct'] = (row['Close'] - closest_1d['Close']) / closest_1d['Close']
            df_5min.at[i, '1d_range'] = (closest_1d['High'] - closest_1d['Low']) / closest_1d['Close']
            df_5min.at[i, '1d_position'] = (row['Close'] - closest_1d['Low']) / (closest_1d['High'] - closest_1d['Low']) if closest_1d['High'] != closest_1d['Low'] else 0.5
            df_5min.at[i, '1d_volume_ratio'] = row['Volume'] / closest_1d['Volume'] if closest_1d['Volume'] > 0 else 1
            df_5min.at[i, 'close_over_1d_SMA'] = row['Close'] > closest_1d['SMA_10'] if 'SMA_10' in closest_1d else np.nan
            df_5min.at[i, '1d_trend'] = 1 if closest_1d['Close'] > closest_1d['Open'] else 0
    
    # Créer des caractéristiques d'alignement de tendance
    df_5min['bullish_alignment'] = ((df_5min['1h_trend'] == 1) & (df_5min['4h_trend'] == 1) & (df_5min['1d_trend'] == 1)).astype(int)
    df_5min['bearish_alignment'] = ((df_5min['1h_trend'] == 0) & (df_5min['4h_trend'] == 0) & (df_5min['1d_trend'] == 0)).astype(int)
    df_5min['mixed_trend_signals'] = (~df_5min['bullish_alignment'].astype(bool) & ~df_5min['bearish_alignment'].astype(bool)).astype(int)
    
    return df_5min

def prepare_data_for_prediction(df_5min, df_1h, df_4h, df_1d, sequence_length=24):
    """
    Prépare les données pour la prédiction.
    
    Args:
        df_5min: DataFrame avec les données 5 minutes
        df_1h: DataFrame avec les données 1 heure
        df_4h: DataFrame avec les données 4 heures
        df_1d: DataFrame avec les données 1 jour
        sequence_length: Longueur de la séquence pour le modèle LSTM
    
    Returns:
        X: Données normalisées pour la prédiction
    """
    # Assurez-vous d'avoir suffisamment de données
    if len(df_5min) < sequence_length:
        logger.warning(f"Pas assez de données pour la prédiction. Nécessite {sequence_length} points de données, seulement {len(df_5min)} disponibles.")
        return None
    
    # Ajouter les colonnes temporelles
    df_5min = add_time_columns(df_5min)
    
    # Ajouter les caractéristiques des bougies
    df_5min = add_candle_features(df_5min)
    df_5min = add_wick_features(df_5min)
    df_5min = add_body_ratio(df_5min)
    
    # Ajouter les indicateurs de volume
    df_5min = add_volume_indicators(df_5min)
    
    # Ajouter les indicateurs techniques
    df_5min = add_technical_indicators(df_5min)
    
    # Ajouter les heures d'ouverture du marché
    for i, row in df_5min.iterrows():
        market_open, stock_open, is_summer = get_market_opening(row['FromDate'])
        df_5min.at[i, 'market_open_hour'] = market_open
        df_5min.at[i, 'stock_open_hour'] = stock_open
        df_5min.at[i, 'is_summer'] = is_summer
        df_5min.at[i, 'period_of_day'] = get_period_of_day_with_timezone(row['FromDate'])
    
    # Ajouter les features multi-timeframes
    df_5min = add_multi_timeframe_features(df_5min, df_1h, df_4h, df_1d)
    
    # Supprimer les lignes avec des valeurs manquantes
    df_5min = df_5min.dropna()
    
    # Sélectionner uniquement les features nécessaires
    features_df = df_5min[selected_features].copy()
    
    # Vérifier s'il y a suffisamment de données après le nettoyage
    if len(features_df) < sequence_length:
        logger.warning(f"Pas assez de données pour la prédiction après le nettoyage. Nécessite {sequence_length} points de données, seulement {len(features_df)} disponibles.")
        return None
    
    # Préparer les séquences
    seq_x = features_df.iloc[-sequence_length:].values
    X = np.array([seq_x])
    
    # Normaliser les données
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    return X_scaled

async def handle_websocket(websocket, path):
    """Gère la connexion WebSocket."""
    logger.info(f"Nouvelle connexion WebSocket: {websocket.remote_address}")
    
    # Charger le modèle
    model = load_model(MODEL_PATH)
    logger.info(f"Modèle chargé depuis {MODEL_PATH}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Message reçu: {data['timeframe']} - {len(data['data'])} barres")
                
                # Extraire le timeframe et les données
                timeframe = data['timeframe']
                ohlcv_data = data['data']
                
                # Mettre à jour les données historiques pour le timeframe correspondant
                update_historical_data(timeframe, ohlcv_data)
                
                # Préparer les données pour la prédiction
                X = prepare_prediction_data()
                
                if X is not None:
                    # Faire la prédiction
                    prediction = make_prediction(model, X)
                    
                    # Envoyer la prédiction au client
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'prediction': prediction.tolist(),
                        'timestamp': datetime.now().isoformat()
                    }))
                else:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': 'Pas assez de données pour faire une prédiction',
                        'timestamp': datetime.now().isoformat()
                    }))
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON")
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': 'Format JSON invalide',
                    'timestamp': datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"Erreur: {str(e)}")
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }))
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connexion fermée: {websocket.remote_address}")

def update_historical_data(timeframe, ohlcv_data):
    """Met à jour les données historiques pour un timeframe donné."""
    # Convertir les données en DataFrame
    df = pd.DataFrame(ohlcv_data)
    
    # Renommer les colonnes pour correspondre au format attendu
    df = df.rename(columns={
        'timestamp': 'FromDate',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Convertir les timestamps en datetime
    df['FromDate'] = pd.to_datetime(df['FromDate'], unit='ms')
    
    # Mettre à jour les données historiques
    historical_data[timeframe] = df
    
    logger.info(f"Données historiques mises à jour pour {timeframe}: {len(df)} barres du {df['FromDate'].min()} au {df['FromDate'].max()}")

def prepare_prediction_data():
    """Prépare les données pour la prédiction à partir des données historiques."""
    # Vérifier si nous avons des données pour tous les timeframes requis
    if all(len(historical_data[tf]) > 0 for tf in ['5min', '1h', '4h', '1d']):
        # Convertir les DataFrames en types attendus
        df_5min = historical_data['5min'].copy()
        df_1h = historical_data['1h'].copy()
        df_4h = historical_data['4h'].copy()
        df_1d = historical_data['1d'].copy()
        
        # Préparer les données pour la prédiction
        X = prepare_data_for_prediction(df_5min, df_1h, df_4h, df_1d, sequence_length=SEQUENCE_LENGTH)
        return X
    else:
        logger.warning("Données manquantes pour un ou plusieurs timeframes")
        return None

def make_prediction(model, X):
    """Fait une prédiction à partir des données préparées."""
    y_proba = model.predict(X)
    return y_proba.flatten()

async def main():
    """Fonction principale."""
    # Charger le modèle pour s'assurer qu'il est disponible
    try:
        model = load_model(MODEL_PATH)
        logger.info(f"Modèle chargé avec succès depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return
    
    # Démarrer le serveur WebSocket
    logger.info(f"Démarrage du serveur WebSocket sur {HOST}:{PORT}")
    async with websockets.serve(handle_websocket, HOST, PORT):
        await asyncio.Future()  # Exécuter indéfiniment

if __name__ == "__main__":
    asyncio.run(main())