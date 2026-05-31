# Cibles principales pour orchestrer le projet
# Usage · `make setup`, `make data`, `make train`, `make report`, etc.

PY := .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
else
  PY := .venv/bin/python
endif

.PHONY: help setup data train train-cnn train-resnet train-vit train-ae \
        train-vae train-mm test report demo compare eda clean lint

help:
	@echo "Cibles disponibles ·"
	@echo "  make setup        venv + dépendances"
	@echo "  make data         télécharger ChestMNIST + métadonnées NIH"
	@echo "  make eda          analyse exploratoire (figures dans artifacts/figures)"
	@echo "  make train        entraînement complet (CNN + ResNet + ViT + AE + VAE + multimodal)"
	@echo "  make train-cnn    juste le CNN scratch"
	@echo "  make train-ae     juste l'autoencodeur"
	@echo "  make train-vae    juste le VAE"
	@echo "  make train-mm     juste le multimodal"
	@echo "  make test         pytest sur la suite smoke"
	@echo "  make report       génère artifacts/RAPPORT.pdf via FPDF2"
	@echo "  make compare      tableau de comparaison MLflow"
	@echo "  make demo         lance le démonstrateur Streamlit"
	@echo "  make clean        supprime artifacts/ et mlruns/"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m scripts.download_chestmnist --sizes 64 128
	$(PY) -m scripts.download_nih_chestxray14 --metadata-only

eda:
	$(PY) -m scripts.eda

train:
	$(PY) -m scripts.train_all

train-cnn:
	$(PY) -m src.train.train_cnn --epochs 10

train-resnet:
	$(PY) -m src.train.train_resnet --epochs 10

train-vit:
	$(PY) -m src.train.train_vit --epochs 8

train-ae:
	$(PY) -m src.train.train_ae --epochs 15

train-vae:
	$(PY) -m src.train.train_vae --epochs 15

train-mm:
	$(PY) -m src.train.train_multimodal --epochs 8 --fusion late

test:
	$(PY) -m pytest tests/ -q

report:
	$(PY) -m scripts.generate_report

compare:
	$(PY) -m scripts.compare_runs

demo:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf mlruns/ artifacts/*.pt
