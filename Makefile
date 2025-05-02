# Makefile for training and evaluating machine learning models

# Python interpreter
PYTHON = python3

# Main script
SCRIPT = src/model_trainer.py
PREDICT_SCRIPT = src/predict.py
MODEL = model/best_model.h5

# Environment setup
.PHONY: setup
setup:
	pip install -r requirements.txt

# Display help information
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make setup                    - Install required dependencies"
	@echo "  make train                    - Train the model with default settings"
	@echo "  make train-cv                 - Train with cross-validation"
	@echo "  make train-ensemble           - Train only the ensemble model"
	@echo "  make train-attention          - Train only the attention model"
	@echo "  make train-bidirectional      - Train only the bidirectional LSTM model"
	@echo "  make train-deep-lstm          - Train only the deep LSTM model"
	@echo "  make evaluate                 - Evaluate the best model"
	@echo "  make feature-importance       - Run feature importance analysis"
	@echo "  make start-service            - Start the WebSocket prediction service"
	@echo "  make clean                    - Remove all generated files"
	@echo "  make help                     - Display this help message"

# Train the model with default settings
.PHONY: train
train:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=deep_lstm --epochs=2 --batch-size=512

# Train with cross-validation
.PHONY: train-cv
train-cv:
	$(PYTHON) $(SCRIPT) --mode=model_comparison

# Train only the ensemble model
.PHONY: train-ensemble
train-ensemble:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=ensemble

# Train only the attention model
.PHONY: train-attention
train-attention:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=attention

# Train only the bidirectional LSTM model
.PHONY: train-bidirectional
train-bidirectional:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=bidirectional

# Train only the deep LSTM model
.PHONY: train-deep-lstm
train-deep-lstm:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=deep_lstm

# Run feature importance analysis
.PHONY: feature-importance
feature-importance:
	$(PYTHON) $(SCRIPT) --mode=feature_importance --model=ensemble

# Evaluate the best model (uses the last trained model by default)
.PHONY: evaluate
evaluate:
	$(PYTHON) $(SCRIPT) --mode=evaluate

# Predict with a trained model
.PHONY: predict
predict:
	$(PYTHON) $(PREDICT_SCRIPT) --model=ensemble --limit=200

# Predict with a specific model
.PHONY: predict-ensemble predict-attention predict-bidirectional predict-deep-lstm
predict-ensemble:
	$(PYTHON) $(PREDICT_SCRIPT) --model=ensemble --limit=200
predict-attention:
	$(PYTHON) $(PREDICT_SCRIPT) --model=attention --limit=200
predict-bidirectional:
	$(PYTHON) $(PREDICT_SCRIPT) --model=bidirectional --limit=200
predict-deep-lstm:
	$(PYTHON) $(PREDICT_SCRIPT) --model=deep_lstm --limit=200

# Custom prediction
# Example: make custom-predict MODEL=attention LIMIT=300 OUTPUT=my_predictions.png
.PHONY: custom-predict
custom-predict:
	$(PYTHON) $(PREDICT_SCRIPT) --model=$(MODEL) --limit=$(LIMIT) --output=$(OUTPUT)

# Start WebSocket prediction service
.PHONY: start-service
start-service:
	$(PYTHON) ./src/prediction_service.py

# Clean generated files
.PHONY: clean
clean:
	rm -rf model_*.h5
	rm -rf *.png
	rm -rf model_evaluation/
	rm -rf __pycache__/
	rm -rf *.csv

# Advanced configuration using variables
# Example: make custom-train MODEL=attention EPOCHS=200 BATCH_SIZE=64 SEQUENCE_LEN=32
.PHONY: custom-train
custom-train:
	$(PYTHON) $(SCRIPT) --mode=evaluate --model=$(MODEL) --epochs=$(EPOCHS) --batch-size=$(BATCH_SIZE) --sequence-length=$(SEQUENCE_LEN)