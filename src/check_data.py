import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def analyze_timeframe_data(file_path):
    print(f"\nAnalyzing {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_candles = []
    for instrument in data['Candles']:
        all_candles.extend(instrument['Candles'])
    
    df = pd.DataFrame(all_candles)
    df['FromDate'] = pd.to_datetime(df['FromDate'])
    df = df.sort_values('FromDate')
    
    # Vérifier le tri
    is_sorted = df['FromDate'].is_monotonic_increasing
    print(f"Data is properly sorted: {is_sorted}")
    
    # Vérifier les trous dans les données
    expected_freq = {
        '1d': 'D',
        '1h': 'H',
        '4h': '4H'
    }
    
    timeframe = os.path.basename(os.path.dirname(file_path))
    freq = expected_freq.get(timeframe)
    
    if freq:
        date_range = pd.date_range(start=df['FromDate'].min(), 
                                 end=df['FromDate'].max(), 
                                 freq=freq)
        missing_dates = set(date_range) - set(df['FromDate'])
        print(f"Missing dates: {len(missing_dates)}")
        if missing_dates:
            print("First 5 missing dates:", sorted(list(missing_dates))[:5])
    
    # Vérifier les valeurs aberrantes
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[abs(df[col] - mean) > 3 * std]
            print(f"\nOutliers in {col}:")
            print(f"Mean: {mean:.2f}")
            print(f"Std: {std:.2f}")
            print(f"Number of outliers: {len(outliers)}")
            if not outliers.empty:
                print("Sample of outliers:")
                print(outliers[['FromDate', col]].head())
    
    # Vérifier la cohérence des données
    print("\nData consistency checks:")
    print(f"High >= Open: {(df['High'] >= df['Open']).all()}")
    print(f"High >= Close: {(df['High'] >= df['Close']).all()}")
    print(f"Low <= Open: {(df['Low'] <= df['Open']).all()}")
    print(f"Low <= Close: {(df['Low'] <= df['Close']).all()}")
    
    # Statistiques générales
    print("\nGeneral statistics:")
    print(f"Date range: {df['FromDate'].min()} to {df['FromDate'].max()}")
    print(f"Total number of candles: {len(df)}")
    print(f"Number of unique instruments: {df['InstrumentID'].nunique()}")

def main():
    base_dir = "brent"
    timeframes = ['1d', '1h', '4h']
    
    for timeframe in timeframes:
        file_path = os.path.join(base_dir, timeframe, 'all_v2.json')
        if os.path.exists(file_path):
            analyze_timeframe_data(file_path)
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main() 