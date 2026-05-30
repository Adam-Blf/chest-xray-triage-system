"""Génère le rapport PDF final via FPDF2.

Pipeline · lit ``RAPPORT.md`` à la racine, le rend en PDF A4 avec ·
- police Unicode (DejaVu si dispo, fallback Segoe UI Windows)
- design système minimal · navy `#001329` pour les titres, brass `#D4A437`
  pour les accents, prose `#1F2937`
- en-tête + pied de page paginé
- table des matières automatique à partir des `## n. Titre`
- conservation des accents, médiopoints `·` et caractères spéciaux

Usage ·
    python -m scripts.generate_report                       # défaut · RAPPORT.md -> artifacts/RAPPORT.pdf
    python -m scripts.generate_report --src AUTRE.md --out out.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent

NAVY = (0, 19, 41)
BRASS = (212, 164, 55)
PROSE = (31, 41, 55)
SUBTLE = (107, 114, 128)


def _resolve_font_path() -> tuple[Path, Path] | None:
    """Cherche DejaVu (Linux/portable) puis Segoe UI (Windows)."""
    candidates = [
        (Path("C:/Windows/Fonts/DejaVuSans.ttf"),
         Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf")),
        (Path("C:/Windows/Fonts/segoeui.ttf"),
         Path("C:/Windows/Fonts/segoeuib.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold
    return None


class RapportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 22, 20)
        self._has_unicode = False
        fonts = _resolve_font_path()
        if fonts is not None:
            regular, bold = fonts
            self.add_font("Body", "", str(regular))
            self.add_font("Body", "B", str(bold))
            self._has_unicode = True
            self._family = "Body"
        else:
            self._family = "Helvetica"

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(self._family, "B", 9)
        self.set_text_color(*SUBTLE)
        self.cell(0, 7, "Système d'aide au tri radiologique · EFREI M1",
                  border=0, align="L")
        self.cell(0, 7, "Beloucif · Morice · Dissongo",
                  border=0, align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font(self._family, "", 8)
        self.set_text_color(*SUBTLE)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def cover(self) -> None:
        self.add_page()
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 60, "F")
        self.set_text_color(255, 255, 255)
        self.set_font(self._family, "B", 22)
        self.set_xy(20, 22)
        self.multi_cell(0, 9,
                        "Système d'aide au tri radiologique",
                        align="L")
        self.set_font(self._family, "", 12)
        self.set_text_color(*BRASS)
        self.set_xy(20, 44)
        self.cell(0, 6,
                  "Projet Deep Learning · M1 Data Engineering & IA · EFREI 2025-2026",
                  align="L")

        self.set_y(78)
        self.set_font(self._family, "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 7, "Auteurs", new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._family, "", 11)
        self.set_text_color(*PROSE)
        for line in (
            "Adam Beloucif · adam.beloucif@efrei.net",
            "Emilien Morice",
            "Arnaud Dissongo",
        ):
            self.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

        self.set_font(self._family, "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 7, "Repo", new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._family, "", 11)
        self.set_text_color(*PROSE)
        self.cell(0, 6, "https://github.com/Adam-Blf/chest-xray-triage-system",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

        self.set_font(self._family, "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 7, "Sujet", new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._family, "", 11)
        self.set_text_color(*PROSE)
        self.multi_cell(0, 6,
                        "Concevoir un système d'aide au tri radiologique capable "
                        "de prédire des pathologies thoraciques à partir de "
                        "radiographies, d'identifier les cas atypiques ou hors "
                        "distribution, d'exploiter le contexte textuel quand "
                        "il est disponible, et de rendre l'ensemble testable "
                        "via un démonstrateur applicatif.")

    def section_title(self, text: str) -> None:
        self.ln(4)
        self.set_text_color(*NAVY)
        self.set_font(self._family, "B", 14)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BRASS)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(),
                  self.l_margin + 28, self.get_y())
        self.ln(3)

    def subsection_title(self, text: str) -> None:
        self.ln(1)
        self.set_text_color(*NAVY)
        self.set_font(self._family, "B", 11)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def body_text(self, text: str) -> None:
        self.set_text_color(*PROSE)
        self.set_font(self._family, "", 10)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT", wrapmode="WORD")
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_text_color(*PROSE)
        self.set_font(self._family, "", 10)
        cur_x = self.get_x()
        self.cell(4, 5, "·")
        self.set_x(cur_x + 4)
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT", wrapmode="CHAR")


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
BULLET_RE = re.compile(r"^[-*]\s+(.+)$")
TABLE_RE = re.compile(r"^\|.+\|$")


def render_markdown(pdf: RapportPDF, md_text: str) -> None:
    pdf.add_page()
    lines = md_text.replace("\r\n", "\n").split("\n")
    in_code = False
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            pdf.body_text(" ".join(buffer).strip())
            buffer = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_buffer()
            in_code = not in_code
            continue
        if in_code:
            pdf.set_font(pdf._family, "", 9)
            pdf.set_text_color(*SUBTLE)
            pdf.multi_cell(0, 4.5, line if line else " ",
                           new_x="LMARGIN", new_y="NEXT", wrapmode="CHAR")
            continue
        if not line:
            flush_buffer()
            pdf.ln(2)
            continue

        h = HEADING_RE.match(line)
        if h:
            flush_buffer()
            level = len(h.group(1))
            txt = h.group(2).strip()
            if level == 1:
                pdf.set_font(pdf._family, "B", 16)
                pdf.set_text_color(*NAVY)
                pdf.ln(2)
                pdf.cell(0, 9, txt, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
            elif level == 2:
                pdf.section_title(txt)
            else:
                pdf.subsection_title(txt)
            continue

        b = BULLET_RE.match(line)
        if b:
            flush_buffer()
            pdf.bullet(b.group(1).strip())
            continue

        if TABLE_RE.match(line):
            flush_buffer()
            pdf.set_font(pdf._family, "", 8)
            pdf.set_text_color(*PROSE)
            pdf.multi_cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT",
                           wrapmode="CHAR")
            continue

        buffer.append(line.strip())

    flush_buffer()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "RAPPORT.md"))
    p.add_argument("--out", default=str(ROOT / "artifacts" / "RAPPORT.pdf"))
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"source introuvable · {src}")

    pdf = RapportPDF()
    pdf.cover()
    render_markdown(pdf, src.read_text(encoding="utf-8"))
    pdf.output(str(out))
    print(f"PDF généré · {out}  ({out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
