import pandas as pd
import numpy as np
import sys
import os

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import load_data, load_timeframe_data

# Charger les données
print("Loading timeframe data...")
df_1h = load_timeframe_data('brent', '1h')
df_4h = load_timeframe_data('brent', '4h')
df_1d = load_timeframe_data('brent', '1d')

print("Loading and processing main data...")
features_df = load_data('brent', df_1h, df_4h, df_1d)

# Exporter les 10 premières lignes
print("Exporting first 10 rows to CSV...")
features_df.head(10).to_csv('features_sample.csv', index=False)
print("CSV file created successfully!") 