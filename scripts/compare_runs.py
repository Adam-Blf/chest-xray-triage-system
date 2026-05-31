"""Compare les runs MLflow et produit un tableau récapitulatif.

Sortie ·
- ``artifacts/figures/runs_comparison.csv`` · une ligne par run, colonnes
  métriques + params clés
- ``artifacts/figures/runs_comparison.png`` · bar chart triée par
  `test_macro_auroc`
- impression console du tableau formaté
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from src.config import ARTIFACTS_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI   # noqa: E402

FIG_DIR = ARTIFACTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


METRIC_KEYS = [
    "test_macro_auroc",
    "test_micro_auroc",
    "test_macro_f1",
    "val_macro_auroc",
    "val_recon_err_mean",
    "val_anomaly_score_mean",
    "test_macro_auroc_fused_logits",
    "test_macro_auroc_image_logits",
    "test_macro_auroc_text_logits",
]

PARAM_KEYS = ["model_class", "epochs", "image_size", "batch_size", "lr",
              "n_params", "device", "fusion"]


def collect() -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        raise SystemExit(f"aucune expérience MLflow nommée {MLFLOW_EXPERIMENT!r}")

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    rows = []
    for r in runs:
        row = {
            "run_id": r.info.run_id[:8],
            "run_name": r.data.tags.get("mlflow.runName", r.info.run_id),
            "family": r.data.tags.get("family", "?"),
            "variant": r.data.tags.get("variant", ""),
            "backbone": r.data.tags.get("backbone", ""),
            "status": r.info.status,
        }
        for k in PARAM_KEYS:
            row[k] = r.data.params.get(k, None)
        for k in METRIC_KEYS:
            row[k] = r.data.metrics.get(k, None)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_supervised_auroc(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["test_macro_auroc"]).sort_values("test_macro_auroc")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(sub))))
    bars = ax.barh(sub["run_name"], sub["test_macro_auroc"], color="#001329")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in sub["test_macro_auroc"]],
                 fontsize=8, padding=4)
    ax.set_xlabel("Test macro-AUROC")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Comparaison des runs supervisés (ChestMNIST test)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "runs_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    df = collect()
    if df.empty:
        print("aucune run trouvée · lance d'abord python -m scripts.train_all")
        return

    cols_print = ["run_name", "family", "model_class", "epochs",
                  "test_macro_auroc", "test_macro_f1",
                  "val_recon_err_mean", "test_macro_auroc_fused_logits"]
    cols_print = [c for c in cols_print if c in df.columns]
    print(df[cols_print].to_string(index=False))

    out_csv = FIG_DIR / "runs_comparison.csv"
    df.to_csv(out_csv, index=False)
    plot_supervised_auroc(df)
    print(f"\nCSV  · {out_csv}")
    print(f"PNG  · {FIG_DIR / 'runs_comparison.png'}")


if __name__ == "__main__":
    main()
