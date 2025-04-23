#!/usr/bin/env python3
"""
Script pour charger un modèle entraîné et effectuer des prédictions
sur de nouvelles données de marché avec visualisation des résultats.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
import model_rnn_improved as model_lib
from sklearn.preprocessing import StandardScaler

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prédire les mouvements de prix avec un modèle entraîné")
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['deep_lstm', 'bidirectional', 'attention', 'ensemble'],
                       help='Type de modèle à utiliser pour les prédictions')
    parser.add_argument('--data', type=str, default='seeds/commodities/brent/5min',
                       help='Chemin vers les données à analyser')
    parser.add_argument('--model-path', type=str, 
                       help='Chemin vers le fichier du modèle (par défaut: model_{modèle}.h5)')
    parser.add_argument('--sequence-length', type=int, default=24,
                       help='Longueur des séquences pour le modèle LSTM')
    parser.add_argument('--output', type=str, default='predictions.png',
                       help='Fichier de sortie pour la visualisation')
    parser.add_argument('--limit', type=int, default=200,
                       help='Nombre de barres à prédire')
    
    return parser.parse_args()

def load_and_prepare_data(data_path, limit=None):
    """
    Charge et prépare les données depuis le chemin spécifié.
    Applique toutes les transformations nécessaires.
    """
    print(f"Chargement des données depuis {data_path}...")
    
    # Utiliser les fonctions de model_lib pour charger et traiter les données
    features_df = model_lib.preprocess_features(data_path)
    
    if limit:
        print(f"Limitation à {limit} barres les plus récentes")
        features_df = features_df.sort_values('FromDate').tail(limit + 50)  # +50 pour avoir assez de données pour les calculs
    
    # Ajouter les features temporelles
    features_df = model_lib.add_time_columns(features_df)
    
    # Calculer l'objectif (pour l'évaluation)
    features_df = model_lib.add_future_direction(features_df)
    
    # Ajouter les features de bougies
    features_df = model_lib.add_direction(features_df)
    features_df = model_lib.add_candle_features(features_df)
    features_df = model_lib.add_doji(features_df)
    features_df = model_lib.add_candle_trend_relation(features_df)
    features_df = model_lib.add_engulfing(features_df)
    features_df = model_lib.add_wick_features(features_df)
    features_df = model_lib.add_body_ratio(features_df)
    
    # Ajouter les features multi-timeframe
    print("Ajout des features multi-timeframe...")
    base_dir = os.path.dirname(data_path)
    features_df = model_lib.add_multi_timeframe_features(features_df, base_directory=base_dir)
    
    # Ajouter les indicateurs de volume
    features_df = model_lib.add_volume_indicators(features_df)
    
    # Ajouter les autres indicateurs techniques
    features_df['SMA_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['EMA_10'] = features_df['Close'].ewm(span=10, adjust=False).mean()
    features_df['RSI_14'] = model_lib.calculate_rsi(features_df)
    
    # Nettoyer les données 
    features_df = features_df.dropna()
    
    if limit:
        features_df = features_df.sort_values('FromDate').tail(limit)
    
    print(f"Données préparées: {len(features_df)} barres du {features_df['FromDate'].min()} au {features_df['FromDate'].max()}")
    
    return features_df

def prepare_sequences(features_df, feature_list, sequence_length):
    """
    Prépare des séquences d'entrée pour le modèle LSTM.
    """
    sequences = []
    targets = []
    
    # Pour chaque point de donnée valide (avec une cible valide)
    for i in range(len(features_df) - sequence_length):
        if features_df['has_valid_target'].iloc[i + sequence_length] == 1:
            # Extraire la séquence X
            seq_x = features_df[feature_list].iloc[i:i + sequence_length].values
            # Extraire la cible y
            seq_y = features_df['future_direction_2'].iloc[i + sequence_length]
            
            sequences.append(seq_x)
            targets.append(seq_y)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    return X_scaled, y, features_df.iloc[sequence_length:sequence_length+len(y)]

def load_model(model_type, model_path=None):
    """
    Charge un modèle entraîné depuis un fichier.
    """
    if model_path is None:
        model_path = f'model_{model_type}.h5'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle {model_path} n'existe pas.")
    
    print(f"Chargement du modèle depuis {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès.")
    
    return model

def predict_and_evaluate(model, X, y, dates_df):
    """
    Effectue des prédictions et évalue les performances.
    """
    # Faire des prédictions
    y_proba = model.predict(X)
    y_pred = (y_proba > 0.5).astype(int).flatten()
    
    # Évaluer la précision
    accuracy = (y_pred == y).mean()
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame({
        'Date': dates_df['FromDate'].values,
        'Close': dates_df['Close'].values,
        'Volume': dates_df['Volume'].values,
        'Actual': y,
        'Predicted': y_pred,
        'Confidence': y_proba.flatten()
    })
    
    print(f"Précision: {accuracy:.4f}")
    print(f"Distribution des prédictions: {np.bincount(y_pred)}")
    
    return results_df, accuracy

def plot_predictions(results_df, output_file):
    """
    Visualise les prédictions sous forme de graphique.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Tracer le prix de clôture
    ax1.plot(results_df['Date'], results_df['Close'], color='blue', label='Prix de clôture')
    ax1.set_title('Analyse prédictive du cours et du volume')
    ax1.set_ylabel('Prix')
    ax1.grid(True, alpha=0.3)
    
    # Marquer les prédictions haussières correctes (vrai positif)
    true_positive = results_df[(results_df['Predicted'] == 1) & (results_df['Actual'] == 1)]
    ax1.scatter(true_positive['Date'], true_positive['Close'], color='green', marker='^', label='Hausse prédite correcte')
    
    # Marquer les prédictions baissières correctes (vrai négatif)
    true_negative = results_df[(results_df['Predicted'] == 0) & (results_df['Actual'] == 0)]
    ax1.scatter(true_negative['Date'], true_negative['Close'], color='red', marker='v', label='Baisse prédite correcte')
    
    # Marquer les prédictions incorrectes
    false_positive = results_df[(results_df['Predicted'] == 1) & (results_df['Actual'] == 0)]
    ax1.scatter(false_positive['Date'], false_positive['Close'], color='orange', marker='x', label='Prédiction incorrecte (hausse)')
    
    false_negative = results_df[(results_df['Predicted'] == 0) & (results_df['Actual'] == 1)]
    ax1.scatter(false_negative['Date'], false_negative['Close'], color='purple', marker='x', label='Prédiction incorrecte (baisse)')
    
    ax1.legend()
    
    # Tracer le volume
    ax2.bar(results_df['Date'], results_df['Volume'], color='blue', alpha=0.6)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Tracer la confiance du modèle
    ax3.plot(results_df['Date'], results_df['Confidence'], color='purple', label='Confiance du modèle')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.fill_between(results_df['Date'], 0.5, results_df['Confidence'], 
                     where=(results_df['Confidence'] > 0.5), color='green', alpha=0.3)
    ax3.fill_between(results_df['Date'], results_df['Confidence'], 0.5,
                     where=(results_df['Confidence'] < 0.5), color='red', alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Confiance')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualisation sauvegardée dans {output_file}")
    
    try:
        plt.show()
    except:
        print("Affichage désactivé en mode non-interactif.")

def generate_trading_signals(results_df):
    """
    Génère des signaux de trading basés sur les prédictions et la confiance.
    """
    # Seuil de confiance
    high_confidence_threshold = 0.75
    low_confidence_threshold = 0.25
    
    # Créer des signaux basés sur les prédictions et le niveau de confiance
    signals = pd.DataFrame({
        'Date': results_df['Date'],
        'Close': results_df['Close'],
        'Signal': 'Hold'  # Par défaut, on tient la position
    })
    
    # Signal d'achat: confiance élevée pour une hausse
    signals.loc[results_df['Confidence'] > high_confidence_threshold, 'Signal'] = 'Buy'
    
    # Signal de vente: confiance élevée pour une baisse
    signals.loc[results_df['Confidence'] < low_confidence_threshold, 'Signal'] = 'Sell'
    
    # Compter les signaux
    signal_counts = signals['Signal'].value_counts()
    print("\nSignaux de trading générés:")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count}")
    
    # Identifier les points d'entrée et de sortie (changements de signal)
    signals['PrevSignal'] = signals['Signal'].shift(1)
    trade_points = signals[signals['Signal'] != signals['PrevSignal']]
    
    print("\nPoints d'entrée/sortie potentiels:")
    for i, row in trade_points.iterrows():
        print(f"  {row['Date']}: {row['PrevSignal'] if not pd.isna(row['PrevSignal']) else 'Début'} -> {row['Signal']} à {row['Close']}")
    
    return signals

def main():
    """Fonction principale."""
    args = parse_arguments()
    
    # Définir le chemin du modèle s'il n'est pas spécifié
    if args.model_path is None:
        args.model_path = f'model_{args.model}.h5'
    
    # Vérifier si le fichier modèle existe
    if not os.path.exists(args.model_path):
        print(f"Erreur: Le fichier modèle {args.model_path} n'existe pas.")
        print("Vous devez d'abord entraîner le modèle. Utilisez la commande:")
        print(f"  make train-{args.model}")
        sys.exit(1)
    
    # Obtenir la liste des features depuis model_lib
    feature_list = model_lib.feature_list
    
    # Charger et préparer les données
    features_df = load_and_prepare_data(args.data, limit=args.limit)
    
    # Préparer les séquences
    X, y, dates_df = prepare_sequences(features_df, feature_list, args.sequence_length)
    
    # Charger le modèle
    model = load_model(args.model, args.model_path)
    
    # Faire des prédictions et évaluer
    results_df, accuracy = predict_and_evaluate(model, X, y, dates_df)
    
    # Générer des signaux de trading
    signals = generate_trading_signals(results_df)
    
    # Visualiser les résultats
    plot_predictions(results_df, args.output)
    
    # Sauvegarder les résultats dans un CSV
    results_csv = args.output.replace('.png', '.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Résultats détaillés sauvegardés dans {results_csv}")

if __name__ == "__main__":
    main()