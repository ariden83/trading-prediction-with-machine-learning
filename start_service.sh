#!/bin/bash

# Chemin vers Miniconda et nom de l'environnement
CONDA_PATH="$HOME/miniconda"
ENV_NAME="ml-trading"
CONDA_PYTHON="$CONDA_PATH/envs/$ENV_NAME/bin/python"

# Activer l'environnement Conda
echo "Activating Conda environment: $ENV_NAME"
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

# Ajouter le répertoire courant au PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Vérifier que le répertoire model existe
mkdir -p model

# Créer un mock pour TensorFlow si nécessaire (pour tester sans TensorFlow)
if [ "$1" == "--mock-tensorflow" ]; then
    echo "Creating TensorFlow mock for testing..."
    cat > ./src/tensorflow_mock.py << 'EOF'
#!/usr/bin/env python3
"""
Mock module pour TensorFlow (pour les tests sans TensorFlow installé)
"""
print("Using TensorFlow MOCK")

class MockObject:
    def __getattr__(self, name):
        return self
    
    def __call__(self, *args, **kwargs):
        return self

tf = MockObject()
keras = MockObject()

# Pour que les imports de tensorflow.keras fonctionnent
tf.keras = keras
EOF
    # Utiliser le script de test sans TensorFlow
    echo "Running test service..."
    python ./src/service_test.py
else
    # Lancer le vrai service
    echo "Starting prediction service with Python 3.10..."
    # Si TensorFlow n'est pas disponible, affichez un message d'erreur clair
    set +e  # Ne pas s'arrêter en cas d'erreur
    $CONDA_PYTHON -c "import tensorflow" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: TensorFlow n'est pas installé. Pour installer TensorFlow:"
        echo "1. Activez l'environnement: source $CONDA_PATH/bin/activate $ENV_NAME"
        echo "2. Installez TensorFlow: pip install tensorflow==2.10.0"
        echo ""
        echo "Pour tester sans TensorFlow, exécutez: $0 --mock-tensorflow"
        exit 1
    fi
    
    # Exécuter le service
    $CONDA_PYTHON ./src/prediction_service.py
fi

# Désactiver l'environnement Conda à la fin
conda deactivate