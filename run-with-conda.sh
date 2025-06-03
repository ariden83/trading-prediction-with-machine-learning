#!/bin/bash

# Chemin vers Miniconda
CONDA_PATH="$HOME/miniconda"

# Vérifier si une cible du Makefile a été spécifiée
if [ $# -eq 0 ]; then
    echo "Usage: $0 <make_target>"
    echo "Example: $0 start-service"
    exit 1
fi

echo "Using Conda base environment"
export PATH="$CONDA_PATH/bin:$PATH"

# Définir le Python de conda comme interpréteur par défaut
PYTHON_BIN="$CONDA_PATH/bin/python"

# Exécuter make avec la cible spécifiée, en remplaçant 'python3' par le chemin vers l'interpréteur conda
echo "Running: make $@ PYTHON=$PYTHON_BIN"
make "$@" PYTHON="$PYTHON_BIN"

# Stocker le code de retour
status=$?

exit $status