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
        logging.StreamHandler()
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
    features_df = create_parquet(df_5min, df_1h, df_4h, df_1d)

    # Vérifier s'il y a suffisamment de données après le nettoyage
    if len(features_df) < sequence_length:
        logger.warning(f"Pas assez de données pour la prédiction après le nettoyage. Nécessite {sequence_length} points de données, seulement {len(features_df)} disponibles.")
        return None

    features_df = features_df.drop(columns=['FromDate'])
    features_df = features_df.drop(columns=['prev_date'])

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
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        inf_cols = features_df[num_cols].columns[np.isinf(features_df[num_cols]).any()].tolist()
        print(f"Colonnes avec NaN: {nan_cols}")
        print(f"Colonnes avec valeurs infinies: {inf_cols}")

        # 'body_ratio_prev', 'log_return_5m', 'log_return_1h', 'log_return_4h'
        if nan_cols or inf_cols:
            logger.error(f"Colonnes avec NaN: {nan_cols}, colonnes avec valeurs infinies: {inf_cols}")
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
    # Vérifier si nous avons des données pour tous les timeframes requis
    if all(len(historical_data[tf]) > 0 for tf in ['5min', '1h', '4h', '1d']):
        # Convertir les DataFrames en types attendus
        df_5min = historical_data['5min'].copy()
        df_5min = preprocess_features(df_5min)
        # df_5min = remove_zero_open_close(df_5min)

        df_1h = historical_data['1h'].copy()
        df_1h = preprocess_features_light(df_1h)
        # df_1h = remove_zero_open_close(df_1h)

        df_4h = historical_data['4h'].copy()
        df_4h = preprocess_features_light(df_4h)
        # df_4h = remove_zero_open_close(df_4h)

        df_1d = historical_data['1d'].copy()
        df_1d = preprocess_features_light(df_1d)
        # df_1d = remove_zero_open_close(df_1d)

        # Préparer les données pour la prédiction
        x = prepare_data_for_prediction(df_5min, df_1h, df_4h, df_1d, sequence_length=SEQUENCE_LENGTH)
        return x
    else:
        logger.warning("Données manquantes pour un ou plusieurs timeframes")
        return None



def preprocess_features_light(df):
    df['FromDate'] = pd.to_datetime(df['FromDate'])
    return df.sort_values('FromDate')


def make_prediction(model, x):
    """Fait une prédiction à partir des données préparées."""
    y_proba = model.predict(x)
    return y_proba.flatten()


async def main():
    """Fonction principale."""
    # Charger le modèle pour s'assurer qu'il est disponible
    try:
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