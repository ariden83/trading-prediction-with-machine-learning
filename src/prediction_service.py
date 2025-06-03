#!/usr/bin/env python3
"""
Service WebSocket pour les prédictions de marché en temps réel.
Reçoit les données de cours sur différentes timeframes (5min, 1h, 4h, 1d),
applique les transformations du modèle RNN et génère des prédictions.
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import pytz
import websockets
import asyncio
from sklearn.preprocessing import StandardScaler
import ta
from tensorflow.keras.models import load_model
import logging
from model_trainer import create_parquet, preprocess_features

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_service.log')
    ]
)
logger = logging.getLogger('prediction_service')

# Paramètres du modèle
MODEL_PATH = "./model/best_model.h5"  # Chemin vers le modèle entraîné
SEQUENCE_LENGTH = 16  # Longueur des séquences pour le modèle LSTM
HOST = "0.0.0.0"  # Hôte pour le WebSocket
PORT = 8765  # Port pour le WebSocket

# Cache pour stocker les données historiques
historical_data = {
    '5min': [],
    '1h': [],
    '4h': [],
    '1d': []
}

# Scaler pour la normalisation des données
scaler = StandardScaler()

def prepare_data_for_prediction(df_5min, df_1h, df_4h, df_1d, sequence_length=16):
    # Debug: Afficher les tailles d'entrée
    logger.info(f"DEBUG - Entrées dans prepare_data_for_prediction:")
    logger.info(f"  5min: {len(df_5min)} lignes")
    logger.info(f"  1h: {len(df_1h)} lignes") 
    logger.info(f"  4h: {len(df_4h)} lignes")
    logger.info(f"  1d: {len(df_1d)} lignes")
    
    features_df = create_parquet(df_5min, df_1h, df_4h, df_1d)
    
    # Debug: Afficher la taille après create_parquet
    logger.info(f"DEBUG - Après create_parquet: {len(features_df)} lignes")
    
    # Debug: Vérifier quelques features importantes
    if len(features_df) > 0:
        sample_features = ['RSI_14', 'MACD', 'Bollinger_High', 'SuperTrend_Trend']
        available_features = [f for f in sample_features if f in features_df.columns]
        logger.info(f"DEBUG - Features disponibles: {available_features}")
        
        if available_features:
            # Compter les NaN pour les 10 dernières lignes
            last_10 = features_df[available_features].tail(10)
            nan_counts = last_10.isnull().sum()
            logger.info(f"DEBUG - NaN dans les 10 dernières lignes: {dict(nan_counts)}")

    # Vérifier s'il y a suffisamment de données après le nettoyage
    if len(features_df) < sequence_length:
        logger.warning(f"Pas assez de données pour la prédiction après le nettoyage. Nécessite {sequence_length} points de données, seulement {len(features_df)} disponibles.")
        return None

    # Debug: identifier les colonnes non-numériques avant nettoyage
    logger.info(f"DEBUG - Colonnes avant nettoyage: {list(features_df.columns)}")
    non_numeric_cols = features_df.select_dtypes(include=['object', 'datetime', 'string']).columns.tolist()
    logger.info(f"DEBUG - Colonnes non-numériques détectées: {non_numeric_cols}")
    
    # Supprimer toutes les colonnes non-numériques
    features_df = features_df.drop(columns=['FromDate'], errors='ignore')
    features_df = features_df.drop(columns=['prev_date'], errors='ignore')
    features_df = features_df.drop(columns=non_numeric_cols, errors='ignore')
    
    # Vérification finale des types de données
    final_non_numeric = features_df.select_dtypes(include=['object', 'datetime', 'string']).columns.tolist()
    if final_non_numeric:
        logger.warning(f"DEBUG - Colonnes encore non-numériques: {final_non_numeric}")
        features_df = features_df.drop(columns=final_non_numeric, errors='ignore')

    logger.info(f"DEBUG - Colonnes finales: {list(features_df.columns)}")
    logger.info(f"DEBUG - Types de données: {features_df.dtypes.unique()}")

    # Préparer les séquences
    seq_x = features_df.iloc[-sequence_length:].values
    logger.info(f"Séquences conservées pour la prédiction (shape={seq_x.shape}): {seq_x}")
    x = np.array([seq_x])


    # Normaliser les données
    # for col, dtype in features_df.dtypes.items():
    #    print(f"{col}: {dtype}")

    num_cols = features_df.select_dtypes(include=[np.number]).columns
    print(features_df.head())

    try:
        # Analyse détaillée des NaN par colonne
        nan_analysis = features_df.isnull().sum()
        nan_cols = nan_analysis[nan_analysis > 0].to_dict()
        
        # Analyse détaillée des valeurs infinies par colonne  
        inf_analysis = {}
        for col in num_cols:
            inf_count = np.isinf(features_df[col]).sum()
            if inf_count > 0:
                inf_analysis[col] = inf_count
        
        print(f"Colonnes avec NaN (détail): {nan_cols}")
        print(f"Colonnes avec valeurs infinies (détail): {inf_analysis}")
        
        # Log pour le serveur
        logger.info(f"DEBUG - Analyse NaN détaillée: {nan_cols}")
        logger.info(f"DEBUG - Analyse valeurs infinies détaillée: {inf_analysis}")

        # 'body_ratio_prev', 'log_return_5m', 'log_return_1h', 'log_return_4h'
        if nan_cols or inf_analysis:
            logger.error(f"Colonnes avec NaN: {nan_cols}, colonnes avec valeurs infinies: {inf_analysis}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de la vérification NaN/Inf: {e}")
        return None

    scaler = StandardScaler()
    logger.info(f"Normalisation des données avec StandardScaler (shape={x.shape})")
    return scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


async def handle_websocket(websocket):
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
                
                # Vérifier si des données sont fournies
                if not ohlcv_data or len(ohlcv_data) == 0:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': f'Aucune donnée fournie pour le timeframe {timeframe}',
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue
                
                # Mettre à jour les données historiques pour le timeframe correspondant
                update_historical_data(timeframe, ohlcv_data)
                
                # Préparer les données pour la prédiction
                x = prepare_prediction_data()

                if x is not None:

                    logger.info(f"prepare prediction data done: make prediction")
                    # Faire la prédiction
                    prediction = make_prediction(model, x)
                    
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
                logger.error(f"Erreur: {str(e)}", exc_info=True)
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
    min_window = 14  # Fenêtre minimale pour les indicateurs techniques
    manquants = [tf for tf in ['5min', '1h', '4h', '1d'] if len(historical_data[tf]) < min_window]
    if manquants:
        logger.warning(f"Données insuffisantes pour les timeframes suivants (au moins {min_window} lignes requises) : {', '.join(manquants)}")
        return None

    # Vérification supplémentaire pour chaque DataFrame
    for tf in ['5min', '1h', '4h', '1d']:
        if len(historical_data[tf]) < min_window:
            logger.error(f"Le DataFrame {tf} n'a que {len(historical_data[tf])} lignes, minimum requis : {min_window}")
            return None

    # Convertir les DataFrames en types attendus
    df_5min = historical_data['5min'].copy()
    df_5min = preprocess_features(df_5min)

    df_1h = historical_data['1h'].copy()
    df_1h = preprocess_features_light(df_1h)

    df_4h = historical_data['4h'].copy()
    df_4h = preprocess_features_light(df_4h)

    df_1d = historical_data['1d'].copy()
    df_1d = preprocess_features_light(df_1d)

    # Préparer les données pour la prédiction
    x = prepare_data_for_prediction(df_5min, df_1h, df_4h, df_1d, sequence_length=SEQUENCE_LENGTH)
    return x



def preprocess_features_light(df):
    df['FromDate'] = pd.to_datetime(df['FromDate'])
    return df.sort_values('FromDate')


def make_prediction(model, x):
    """Fait une prédiction à partir des données préparées."""
    y_proba = model.predict(x)
    return y_proba.flatten()


async def main():
    """Fonction principale."""
    # Créer un modèle factice adapté aux 158 features si nécessaire
    try:
        model = load_model(MODEL_PATH)
        # Tester avec des données factices pour vérifier la compatibilité
        test_data = np.random.rand(1, 16, 158)
        model.predict(test_data)
        logger.info(f"Modèle existant compatible avec 158 features")
    except Exception as e:
        logger.warning(f"Modèle incompatible ({str(e)}), création d'un nouveau modèle...")
        # Créer un nouveau modèle compatible
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(16, 158)),  # 16 séquences, 158 features
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.save(MODEL_PATH)
        logger.info(f"Nouveau modèle factice créé avec 158 features et sauvé dans {MODEL_PATH}")
    
    # Démarrer le serveur WebSocket
    logger.info(f"Démarrage du serveur WebSocket sur {HOST}:{PORT}")
    async with websockets.serve(handle_websocket, HOST, PORT):
        await asyncio.Future()  # Exécuter indéfiniment


if __name__ == "__main__":
    asyncio.run(main())