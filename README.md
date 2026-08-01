# Chest X-Ray Triage System

[![version](https://img.shields.io/badge/version-0.1.0-000091?style=flat-square)](https://github.com/Adam-Blf/chest-xray-triage-system/releases)

<!-- adam-badges:start -->
[![commits](https://img.shields.io/github/commit-activity/t/Adam-Blf/chest-xray-triage-system?color=001329&label=commits&style=flat-square)](https://github.com/Adam-Blf/chest-xray-triage-system/commits)
[![visites](https://hits.sh/github.com/Adam-Blf/chest-xray-triage-system.svg?style=flat-square&label=visites&color=001329)](https://hits.sh/github.com/Adam-Blf/chest-xray-triage-system/)
[![last commit](https://img.shields.io/github/last-commit/Adam-Blf/chest-xray-triage-system?color=D4A437&style=flat-square&label=dernier%20push)](https://github.com/Adam-Blf/chest-xray-triage-system/commits)
[![top language](https://img.shields.io/github/languages/top/Adam-Blf/chest-xray-triage-system?style=flat-square)](https://github.com/Adam-Blf/chest-xray-triage-system)
[![license](https://img.shields.io/github/license/Adam-Blf/chest-xray-triage-system?style=flat-square&color=D4A437)](LICENSE)
<!-- adam-badges:end -->

Système de triage de radiographies thoraciques basé sur le deep learning. Le projet regroupe la classification supervisée, la détection d’anomalies, la fusion image + texte, le suivi des expériences avec MLflow et une démonstration Streamlit.

## Architecture

```mermaid
flowchart TB
    D["Données<br/>ChestMNIST - NIH - OpenI - MIMIC-CXR"]
    P["Pipeline de préparation<br/>splits patient-disjoints - transforms - normalisation"]
    S["Modèles supervisés<br/>CNN - ResNet - EfficientNet - ViT"]
    A["Détection d’anomalies<br/>Autoencoder - VAE"]
    M["Multimodal<br/>image + texte - fusion early / intermediate / late"]
    T["MLflow<br/>paramètres - métriques - checkpoints"]
    U["Démo Streamlit<br/>upload - scores - rapport"]

    D --> P --> S --> T --> U
    P --> A --> T
    P --> M --> T
```

## Stack

PyTorch - torchvision - timm - medmnist - scikit-learn - MLflow - Streamlit

## Installation

```bash
git clone https://github.com/Adam-Blf/chest-xray-triage-system.git
cd chest-xray-triage-system

python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Démarrage rapide

```bash
python -m scripts.download_chestmnist --sizes 64
python -m scripts.train_all --smoke
mlflow ui --backend-store-uri mlruns
streamlit run app/streamlit_app.py
```

## Commandes utiles

```bash
python -m scripts.download_all
python -m src.train.train_cnn --epochs 10
python -m src.train.train_resnet --backbone resnet18 --epochs 10
python -m src.train.train_vit --backbone vit_tiny_patch16_224 --epochs 8
python -m src.train.train_ae --epochs 15
python -m src.train.train_vae --epochs 15 --beta 1.0
python -m src.train.train_multimodal --fusion late --epochs 8
```

## Structure

```text
src/        logique métier, modèles et entraînement
scripts/    téléchargement, EDA, rapport, orchestration
app/        démonstrateur Streamlit
docs/       documentation complémentaire
```

## Auteurs

- Adam Beloucif
- Emilien Morice
- Arnaud Dissongo

## Licence

MIT

 <picture>
 </picture>
</a>
