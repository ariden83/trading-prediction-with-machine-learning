#!/bin/bash

# Script pour corriger les problèmes d'environnement conda

# Chemin vers Miniconda
CONDA_PATH="$HOME/miniconda"
ENV_NAME="ml-trading"

echo "Activating Conda environment: $ENV_NAME"
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

echo "Current Python path: $(which python)"
echo "Current pip path: $(which pip)"

echo "Force reinstalling required packages..."
pip install -v --force-reinstall numpy pandas scikit-learn scipy tensorflow flask flask-cors websockets ta

# Vérifier l'installation
python -c "
try:
    import pandas as pd
    print(f'✅ pandas {pd.__version__} installed correctly')
except ImportError:
    print('❌ pandas not found')

try:
    import tensorflow as tf
    print(f'✅ tensorflow {tf.__version__} installed correctly')
except ImportError:
    print('❌ tensorflow not found')

try:
    import websockets
    print(f'✅ websockets installed correctly')
except ImportError:
    print('❌ websockets not found')
"

echo "Setup completed. Try running: ./start_service.sh"