#!/usr/bin/env python3
"""
Service WebSocket simplifié pour tester la connexion et l'envoi de données.
"""

import json
import numpy as np
import pandas as pd
import asyncio
import websockets
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_prediction_service.log')
    ]
)
logger = logging.getLogger('test_prediction_service')

HOST = "0.0.0.0"
PORT = 8765

# Cache pour stocker les données historiques
historical_data = {
    '5min': [],
    '1h': [],
    '4h': [],
    '1d': []
}

async def handle_websocket(websocket):
    """Gère la connexion WebSocket."""
    logger.info(f"Nouvelle connexion WebSocket: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                timeframe = data['timeframe']
                ohlcv_data = data['data']
                
                logger.info(f"Message reçu: {timeframe} - {len(ohlcv_data)} barres")
                
                # Vérifier si des données sont fournies
                if not ohlcv_data or len(ohlcv_data) == 0:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': f'Aucune donnée fournie pour le timeframe {timeframe}',
                        'timestamp': datetime.now().isoformat()
                    }))
                    continue
                
                # Stocker les données
                historical_data[timeframe] = ohlcv_data
                logger.info(f"Données stockées pour {timeframe}: {len(ohlcv_data)} barres")
                
                # Vérifier si on a des données pour tous les timeframes
                all_timeframes_ready = all(
                    len(historical_data[tf]) > 0 
                    for tf in ['5min', '1h', '4h', '1d']
                )
                
                if all_timeframes_ready:
                    # Simuler une prédiction
                    prediction_value = np.random.rand()  # Valeur aléatoire entre 0 et 1
                    
                    logger.info(f"Prédiction générée: {prediction_value}")
                    
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'prediction': [prediction_value],
                        'timestamp': datetime.now().isoformat(),
                        'message': f'Prédiction basée sur {sum(len(historical_data[tf]) for tf in historical_data)} barres totales'
                    }))
                else:
                    missing = [tf for tf in ['5min', '1h', '4h', '1d'] if len(historical_data[tf]) == 0]
                    await websocket.send(json.dumps({
                        'status': 'waiting',
                        'message': f'En attente des données pour: {", ".join(missing)}',
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

async def main():
    """Fonction principale."""
    logger.info(f"Démarrage du serveur WebSocket de test sur {HOST}:{PORT}")
    async with websockets.serve(handle_websocket, HOST, PORT):
        await asyncio.Future()  # Exécuter indéfiniment

if __name__ == "__main__":
    asyncio.run(main())