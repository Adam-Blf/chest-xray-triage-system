"""OpenI dataset · pair de-identified XML reports with their PNG images.

Once ``scripts.download_openi`` has fetched and extracted the archives, the
report files live under ``data/openi/ecgen-radiology/*.xml`` and the images
under ``data/openi/NLMCXR_png/*.png``. Each XML carries one or more
``<parentImage id="..."/>`` references along with structured FINDINGS and
IMPRESSION sections.

This loader returns ``(image, report_text, label_vec)`` tuples. Labels are
heuristic · we match the 14 ChestMNIST pathology names against the report
text. The mapping is intentionally permissive · for production use replace
this with a CheXpert/NegBio NLP pipeline.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CHEST_LABELS, DATA_DIR
from src.data import build_transforms

OPENI_DIR = DATA_DIR / "openi"
REPORT_DIR_CANDIDATES = [OPENI_DIR / "ecgen-radiology", OPENI_DIR / "reports"]
IMAGE_DIR_CANDIDATES = [OPENI_DIR / "NLMCXR_png", OPENI_DIR / "images"]


KEYWORDS = {
    "atelectasis": ["atelectasis", "atelectatic"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "effusion": ["pleural effusion", "effusion"],
    "infiltration": ["infiltrate", "infiltration"],
    "mass": ["mass"],
    "nodule": ["nodule"],
    "pneumonia": ["pneumonia"],
    "pneumothorax": ["pneumothorax"],
    "consolidation": ["consolidation"],
    "edema": ["edema"],
    "emphysema": ["emphysema"],
    "fibrosis": ["fibrosis"],
    "pleural_thickening": ["pleural thickening"],
    "hernia": ["hernia"],
}
NEGATION_HINTS = ["no ", "without ", "negative for ", "no evidence of "]


@dataclass
class OpenIRecord:
    image_paths: list[Path]
    findings: str
    impression: str
    labels: np.ndarray


def _strip(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _label_from_text(text: str) -> np.ndarray:
    text_lc = text.lower()
    label = np.zeros(len(CHEST_LABELS), dtype=np.float32)
    for idx, key in enumerate(CHEST_LABELS):
        for kw in KEYWORDS[key]:
            occurrences = [m.start() for m in re.finditer(re.escape(kw), text_lc)]
            for pos in occurrences:
                window = text_lc[max(0, pos - 30): pos]
                if any(hint in window for hint in NEGATION_HINTS):
                    continue
                label[idx] = 1.0
                break
    return label


def _resolve_image(img_dir: Path, image_id: str) -> Path | None:
    cands = [img_dir / f"{image_id}.png", img_dir / image_id]
    for c in cands:
        if c.exists():
            return c
    matches = list(img_dir.rglob(f"{image_id}.png"))
    return matches[0] if matches else None


def parse_openi_reports(report_dir: Path | None = None,
                       image_dir: Path | None = None) -> list[OpenIRecord]:
    if report_dir is None:
        report_dir = next((p for p in REPORT_DIR_CANDIDATES if p.exists()), None)
    if image_dir is None:
        image_dir = next((p for p in IMAGE_DIR_CANDIDATES if p.exists()), None)
    if report_dir is None or image_dir is None:
        raise FileNotFoundError(
            f"OpenI not found. Run scripts.download_openi --extract first. "
            f"reports={report_dir} images={image_dir}"
        )

    records: list[OpenIRecord] = []
    for xml_path in sorted(report_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        findings, impression = "", ""
        for abstract in root.iter("AbstractText"):
            label = (abstract.get("Label") or "").strip().lower()
            if label == "findings":
                findings = _strip(abstract.text)
            elif label == "impression":
                impression = _strip(abstract.text)
        text = (findings + " " + impression).strip()
        if not text:
            continue
        image_ids = [pi.get("id") for pi in root.iter("parentImage") if pi.get("id")]
        image_paths = [p for p in (_resolve_image(image_dir, i) for i in image_ids) if p]
        if not image_paths:
            continue
        records.append(OpenIRecord(
            image_paths=image_paths,
            findings=findings,
            impression=impression,
            labels=_label_from_text(text),
        ))
    return records


class OpenIDataset(Dataset):
    """Real image+text dataset using parsed OpenI records.

    Returns ``(image_tensor, token_ids, attn_mask, label_vec)`` like the
    captioned ChestMNIST dataset, so the multimodal trainer can swap in
    without code changes.
    """

    def __init__(self, image_size: int = 64, max_len: int = 64,
                 split: str = "all", seed: int = 42):
        from src.train.train_multimodal import VOCAB, _tokenize
        self.vocab = VOCAB
        self._tokenize = _tokenize
        self.transform = build_transforms(image_size, train=False)
        self.records = parse_openi_reports()
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.records))
        rng.shuffle(idx)
        n = len(idx)
        if split == "train":
            self.idx = idx[: int(0.8 * n)]
        elif split == "val":
            self.idx = idx[int(0.8 * n): int(0.9 * n)]
        elif split == "test":
            self.idx = idx[int(0.9 * n):]
        else:
            self.idx = idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        rec = self.records[int(self.idx[i])]
        img = Image.open(rec.image_paths[0]).convert("L")
        x = self.transform(img)
        text = (rec.findings + " " + rec.impression).lower()
        ids, mask = self._tokenize(text, max_len=self.max_len)
        y = torch.as_tensor(rec.labels, dtype=torch.float32)
        return x, torch.tensor(ids, dtype=torch.long), \
            torch.tensor(mask, dtype=torch.long), y


__all__ = ["OpenIRecord", "OpenIDataset", "parse_openi_reports"]
