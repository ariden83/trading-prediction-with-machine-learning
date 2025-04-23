# Optimisations à effecuer

- ~~Ajouter une couche LSTM supplémentaire~~
- Ajouter des couches de Dropout et Batch Normalization après chaque LSTM.
- Expérimenter un modèle Bidirectional LSTM.
- Expérimenter un autre modèle Transformer-based modèle (Attention) : un Transformer peut mieux capter les relations temporelles.
- ~~Optimisation du taux d’apprentissage (learning_rate) : essayer 0.0005 au lieu de 0.001 dans Adam.~~
- ~~Batch size : Essayer 64 ou 128 (au lieu de 32).~~
- Nombre d’époques : Vérifier le val_loss pour voir si ça overfit (éventuellement ajouter un ReduceLROnPlateau).
- Analyser les erreurs : quelles conditions spécifiques rendent la prédiction fausse ?
- Ajouter une incertitude : prédire un seuil dynamique pour classifier certaines prédictions comme "incertaines".
- ~~Ajoute des tendances de RSI ou des moyennes mobiles sur différentes périodes.~~
- ~~Si l’ajout de données améliore un peu le score mais que tu restes autour de 0.52-0.55 d’AUC~~
- essaye une architecture plus avancée (CNN-LSTM, Transformer).
- ~~le modèle essaie bien de prédire la direction de la bougie suivante (ou plus exactement, la direction du prix après 5 bougies). Tu peux ajuster cette période si nécessaire (par exemple, tester shift(-1) au lieu de shift(-5)).~~
- ~~Ajoute du Dropout supplémentaire (ou augmente le taux de Dropout existant).~~
- Utilise la L2 Regularization sur les couches denses.
- Si possible, essaye d’augmenter la taille de ton dataset ou d’ajouter du data augmentation si cela est pertinent.
- Essaye une annealing learning rate schedule, en diminuant progressivement le taux d’apprentissage au fil des epochs :
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
```
- Indicateurs de Momentum :
   RSI sur différentes périodes : Ajouter des périodes différentes pour le RSI, comme RSI_30, RSI_50, ou même RSI_200 (pour des périodes plus longues).
   Stochastic Oscillator (Stoch) : Ce modèle peut être utile pour détecter les conditions de surachat ou de survente, comme le stoch mais avec différents paramètres.
   MFI (Money Flow Index) : Le MFI combine le prix et le volume, ce qui peut être utile pour détecter la force du mouvement des prix.
- Indicateurs de Volatilité :
   ATR (Average True Range) sur différentes périodes : Par exemple, ATR_7, ATR_21 pour détecter la volatilité à court terme et à long terme.
   Chaikin Volatility (CHV) : Un autre indicateur pour mesurer la volatilité basée sur la différence entre les bandes de prix.
- Indicateurs basés sur les Bandes de Prix :
   Envelope : Indicateur similaire aux bandes de Bollinger mais basé sur des pourcentages fixes du prix moyen.
   Donchian Channels : Utilisé pour détecter les points de rupture potentiels.
- Indicateurs basés sur la tendance :
   Parabolic SAR : Un autre indicateur de suivi de tendance qui peut indiquer des points de retournement potentiels.
   ADX + DI (Directional Indicators) : Ajouter l'ADX avec ses composants +DI et -DI peut donner une meilleure compréhension de la force de la tendance.
- Pattern de bougies (Candlestick patterns) :
   Candlestick Patterns classiques : Vous pouvez essayer d'ajouter des patterns de bougies comme "Hammer", "Doji", "Engulfing", "Shooting Star", etc., pour aider le modèle à identifier des retournements potentiels.
   Heikin-Ashi : C'est une version modifiée des bougies traditionnelles, souvent utilisée pour identifier les tendances plus facilement.
- Autres indicateurs de tendance :
   Trix (Triple Exponential Average) : Un indicateur de lissage qui aide à identifier les tendances.
   KAMA (Kaufman's Adaptive Moving Average) : Une autre moyenne mobile qui s'adapte en fonction de la volatilité du marché.
- Analyse des volumes :
   On-Balance Volume (OBV) : Utilisé pour mesurer la pression d'achat et de vente à travers les volumes.
   Accumulation/Distribution Index (ADI) : Peut ajouter des informations sur le flux des volumes dans le marché.
- Analyse des Renversements :
   Ichimoku Cloud : Un indicateur complet qui inclut plusieurs lignes pour déterminer la direction, l'élan et les niveaux de support/résistance.
   Zig Zag Indicator : Un indicateur pour détecter les renversements dans les mouvements de prix, utile pour identifier les tendances.
- ~~Caractéristiques de la bougie :~~
  ~~Délai de la bougie précédente (Previous candle's range or close) : Par exemple, la variation de la clôture par rapport à la bougie précédente pourrait aider.~~
  ~~Proportions de la bougie (Proportions de la mèche et du corps) : La relation entre la longueur du corps et des mèches pourrait donner un bon aperçu du sentiment du marché.~~
- Autres idées :
    Distance au plus haut/plus bas de la période précédente : Par exemple, combien de points le prix actuel est éloigné du plus haut ou bas de la période précédente.
    Support/Résistance dynamique : Calculer des niveaux de support et résistance basés sur les prix historiques.
