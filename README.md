# Chest X-Ray Triage System

<!-- adam-badges:start -->
[![commits](https://img.shields.io/github/commit-activity/t/Adam-Blf/chest-xray-triage-system?color=001329&label=commits&style=flat-square)](https://github.com/Adam-Blf/chest-xray-triage-system/commits)
[![visites](https://hits.sh/github.com/Adam-Blf/chest-xray-triage-system.svg?style=flat-square&label=visites&color=001329)](https://hits.sh/github.com/Adam-Blf/chest-xray-triage-system/)
[![last commit](https://img.shields.io/github/last-commit/Adam-Blf/chest-xray-triage-system?color=D4A437&style=flat-square&label=dernier%20push)](https://github.com/Adam-Blf/chest-xray-triage-system/commits)
[![top language](https://img.shields.io/github/languages/top/Adam-Blf/chest-xray-triage-system?style=flat-square)](https://github.com/Adam-Blf/chest-xray-triage-system)
[![license](https://img.shields.io/github/license/Adam-Blf/chest-xray-triage-system?style=flat-square&color=D4A437)](LICENSE)
<!-- adam-badges:end -->

Deep-learning triage system for chest radiographs · EFREI M1 Data
Engineering & IA · projet trinôme **Adam Beloucif**, **Emilien Morice** &
**Arnaud Dissongo**.

Cadre · [Mastère Data Engineering & IA](https://www.efrei.fr/formation/mastere-data-engineering-ia/),
EFREI Villejuif · RNCP 40875.

The repo covers every requirement from the project brief ·

- **Supervised** · three image classifiers compared on 14-label ChestMNIST
  (CNN trained from scratch · ResNet18 / DenseNet / EfficientNet transfer
  learning · ViT-Tiny via timm).
- **Anomaly detection** · convolutional autoencoder and variational
  autoencoder producing reconstruction-error / KL-based anomaly scores.
- **Multimodal** · image + text fusion (early / intermediate / late)
  evaluating image-only, text-only and fused branches.
- **Experiment tracking** · every run is logged in MLflow with
  hyperparameters, metrics, checkpoints and per-class scores.
- **Demonstrator** · Streamlit app for radiograph upload, supervised
  predictions, anomaly gauge and optional finding-text fusion.

## Stack

PyTorch 2 · torchvision · timm · medmnist · scikit-learn · MLflow · Streamlit

## Datasets

| dataset | role | size | acquisition |
| --- | --- | --- | --- |
| **ChestMNIST / ChestMNIST+** | supervised + anomaly + multimodal proxy | ~100 MB @ 64px · ~600 MB @ 224px | `python -m scripts.download_chestmnist --sizes 64 128 224` |
| **NIH ChestX-ray14** | optional realistic comparator, metadata analysis | ~42 GB images, ~5 MB metadata | `python -m scripts.download_nih_chestxray14 --metadata-only` (metadata) · `--images --extract` (full) |
| **Open-i (NLM)** | multimodal image + free-text reports | ~1.6 GB | `python -m scripts.download_openi --extract` |
| **MIMIC-CXR-JPG** | advanced multimodal · credentialed | ~480 GB | `python -m scripts.download_mimic_cxr --check` after PhysioNet credentialing |

Run them all at once (defaults · ChestMNIST 64+128, NIH metadata only,
OpenI extracted, MIMIC checked) ·

```bash
python -m scripts.download_all
```

Add `--nih-images --nih-extract` to pull the full NIH archive.
MIMIC-CXR-JPG never auto-downloads · the script only checks for
PhysioNet credentials in `~/.netrc` / `_netrc`.

## Quickstart

```bash
git clone https://github.com/Adam-Blf/chest-xray-triage-system.git
cd chest-xray-triage-system

python -m venv .venv
. .venv/Scripts/activate                # Windows · .venv/bin/activate on Linux
pip install -r requirements.txt

python -m scripts.download_chestmnist --sizes 64
python -m scripts.train_all --smoke      # 1-epoch sanity pass over every model
mlflow ui --backend-store-uri mlruns     # browse the runs at http://127.0.0.1:5000
streamlit run app/streamlit_app.py       # open the demonstrator
```

## Training a single model

```bash
python -m src.train.train_cnn           --epochs 10
python -m src.train.train_resnet        --backbone resnet18 --epochs 10
python -m src.train.train_vit           --backbone vit_tiny_patch16_224 --epochs 8
python -m src.train.train_ae            --epochs 15
python -m src.train.train_vae           --epochs 15 --beta 1.0
python -m src.train.train_multimodal    --fusion late --epochs 8
```

Each command writes `artifacts/<run_name>_best.pt` and logs to MLflow.

## Layout

```
src/
  config.py                # paths, hyperparameters, label list
  data.py                  # ChestMNIST loader + transforms + class-balance helpers
  evaluation.py            # multi-label metrics (macro/micro AUROC, F1, per class)
  utils.py                 # seed, device, MLflow context manager
  models/
    cnn_scratch.py         # 4-stage conv baseline
    transfer.py            # ResNet / DenseNet / EfficientNet
    vit.py                 # timm ViT
    autoencoder.py         # conv AE for anomaly detection
    vae.py                 # conv VAE
    multimodal.py          # image+text fusion (early / intermediate / late)
  train/
    trainer.py             # shared supervised trainer (AMP, cosine LR, MLflow)
    train_cnn.py / train_resnet.py / train_vit.py
    train_ae.py / train_vae.py / train_multimodal.py
scripts/
  download_chestmnist.py
  download_nih_chestxray14.py
  download_openi.py
  download_mimic_cxr.py
  openi_loader.py          # parse OpenI XML reports + image pairing
  train_all.py             # one-shot orchestrator (every required model)
app/
  streamlit_app.py         # interactive demonstrator
docs/
  ARCHITECTURE.md
  ADR/
    0001-framework-choice.md
```

## Reproducibility · seeds, splits, anti-leakage

- Single `SEED = 42` propagated through Python `random`, NumPy, PyTorch
  CPU / CUDA, hash seed.
- ChestMNIST provides patient-disjoint train / val / test splits by
  construction; we never re-split.
- Mixed precision (`torch.amp`) is gated on CUDA availability.
- Best checkpoints are selected on **validation macro-AUROC** and stored
  in `artifacts/<run_name>_best.pt`; only the best run is exposed in the
  demonstrator.

## Compliance with the project brief

- §4.1 supervised image classification · CNN + ResNet + ViT (compared)
- §4.2 AE / VAE anomaly detection · convolutional AE and VAE with
  reconstruction-error and KL-aware scoring
- §4.3 multimodal · image-only, text-only and fused branches with
  early / intermediate / late strategies
- §4.4 MLflow tracking · `mlruns/` (override via `MLFLOW_TRACKING_URI`)
- §4.5 demonstrator · Streamlit (PNG / JPG upload + finding text)
- §4.6 clean pipeline · seed, no patient leakage, best-model promotion

## Hardware notes

The smoke pipeline runs on CPU in a few minutes. Full training of the
ViT at 224 px requires CUDA; reduce `--image-size 128` or use the ResNet18
backbone if no GPU is available.

## Auteurs

- **Adam Beloucif** · [adam.beloucif@efrei.net](mailto:adam.beloucif@efrei.net) · [github.com/Adam-Blf](https://github.com/Adam-Blf)
- **Emilien Morice** · binôme habituel des projets [[Mastère Data Engineering & IA]] M1
- **Arnaud Dissongo** · troisième membre de l'équipe

## License

MIT · see [LICENSE](LICENSE).
