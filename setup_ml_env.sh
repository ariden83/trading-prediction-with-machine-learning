#!/bin/bash

# Chemin vers Miniconda et nom de l'environnement
CONDA_PATH="$HOME/miniconda"
ENV_NAME="ml-env"

# Activer l'environnement
echo "Activating Conda environment: $ENV_NAME"
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

# Installer les packages nécessaires avec pip
echo "Installing required packages with pip..."
pip install numpy==1.22.4 pandas==1.4.3 scikit-learn==1.1.2 scipy==1.8.1 flask flask-cors websockets ta

# Vous pouvez également installer TensorFlow si nécessaire
if [ "$1" == "--with-tensorflow" ]; then
    echo "Installing TensorFlow..."
    pip install tensorflow==2.10.0
fi

echo "Setup complete. You can now run the prediction service with:"
echo "./start_service.sh"