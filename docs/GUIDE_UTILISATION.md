# Guide d'utilisation · démarrage rapide

> Démarrer le projet de zéro · datasets, entraînements, démonstrateur,
> rapport PDF.

## 1. Installation (5 min)

```bash
git clone https://github.com/Adam-Blf/chest-xray-triage-system.git
cd chest-xray-triage-system

python -m venv .venv
.venv\Scripts\activate     # Windows
# . .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Vérification ·

```bash
pytest tests/ -q             # 6 tests, < 15 s
```

## 2. Téléchargement des données (1 - 30 min selon ce que tu veux)

```bash
# Minimum vital · ChestMNIST 64 px (~175 Mo)
python -m scripts.download_chestmnist --sizes 64

# Confort · ChestMNIST 64 + 128
python -m scripts.download_chestmnist --sizes 64 128

# Tout d'un coup (ChestMNIST + métadonnées NIH + Open-i extracté)
python -m scripts.download_all --openi-extract

# NIH ChestX-ray14 complet (~42 Go)
python -m scripts.download_nih_chestxray14 --images --extract

# MIMIC-CXR-JPG (credentialed) · vérifier les prérequis avant
python -m scripts.download_mimic_cxr --check
```

## 3. Analyse exploratoire (1 min)

```bash
python -m scripts.eda
```

Produit `artifacts/figures/label_distribution.png`,
`cooccurrence.png`, `examples_positive.png`, `label_stats.csv`.

Pour la version interactive · ouvrir `notebooks/01_eda.ipynb`.

## 4. Entraînement

### Tout d'un coup (smoke · 1 epoch chaque)

```bash
python -m scripts.train_all --smoke
```

### Architecture par architecture (recommandé pour 20/20)

```bash
python -m src.train.train_cnn           --epochs 10
python -m src.train.train_resnet        --backbone resnet18 --epochs 10
python -m src.train.train_vit           --backbone vit_tiny_patch16_224 --epochs 8
python -m src.train.train_ae            --epochs 15
python -m src.train.train_vae           --epochs 15 --beta 1.0
python -m src.train.train_multimodal    --fusion late --epochs 8
python -m src.train.train_multimodal    --fusion early --epochs 8
python -m src.train.train_multimodal    --fusion intermediate --epochs 8
```

Chaque commande logge dans MLflow (`mlruns/`) et dépose un checkpoint
dans `artifacts/<run_name>_best.pt`.

## 5. Comparaison des runs

```bash
python -m scripts.compare_runs
mlflow ui --backend-store-uri mlruns      # http://127.0.0.1:5000
```

`compare_runs` produit ·
- `artifacts/figures/runs_comparison.csv` · tableau complet
- `artifacts/figures/runs_comparison.png` · bar chart des AUROC test

## 6. Démonstrateur Streamlit

```bash
streamlit run app/streamlit_app.py
```

Trois panneaux ·
1. Prédictions supervisées · choisis l'architecture, vois la pathologie
   la plus probable + table triée
2. Score d'atypicité · AE/VAE + seuil p99
3. Fusion image + texte · activé si on saisit un finding text

## 7. Rapport PDF (livrable final)

```bash
python -m scripts.generate_report
```

Produit `artifacts/RAPPORT.pdf` (~50 Ko) à partir de `RAPPORT.md`.
Police Unicode (DejaVu/Segoe UI) pour les accents et le médiopoint.

## 8. Configuration matérielle recommandée

| modèle | CPU | GPU |
| --- | --- | --- |
| SimpleCNN @ 64 px | OK (~3-5 min/epoch) | ~30 s/epoch |
| ResNet18 @ 128 px | OK (~10 min/epoch) | ~1 min/epoch |
| ViT-Tiny @ 224 px | lent (~25 min/epoch) | ~2 min/epoch |
| ConvAE / VAE @ 64 px | OK (~2-3 min/epoch) | ~20 s/epoch |
| Multimodal @ 64 px | OK (~3 min/epoch) | ~30 s/epoch |

Sur CPU seul (16 cœurs Intel), prévoir ~3-4 h pour l'ensemble en 10
epochs chacun. Sur GPU (T4 Colab, RTX 3060), ~30 - 45 min.

## 9. Troubleshooting

**`RuntimeError: operator torchvision::nms does not exist`** ·
torch/torchvision désaccordés. Réinstaller avec les versions pinnées ·

```bash
pip install --force-reinstall --no-deps torch==2.5.1 torchvision==0.20.1
```

**`Not enough horizontal space to render a single character`** ·
FPDF2 sur une ligne trop longue · le générateur force déjà
`wrapmode="CHAR"`. Si ça crash sur ton ajout, casser les URL en
plusieurs lignes ou utiliser un texte plus court.

**Streamlit lent** · normal en CPU. Tester avec une seule image
chargée et désactiver les autres archis depuis la sidebar.
