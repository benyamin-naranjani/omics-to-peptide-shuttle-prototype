#!/usr/bin/env python3
"""OmiPep v0.2 / Module 01 / Cell 14.1.

Create v2, a redesigned reproducible standalone publication figure of the complete mouse D02
stroke-response vs independent human brain endothelial RNA abundance landscape.

Inputs (read only; generated in earlier Module 01 cells):
    data/processed/review_outputs/EC_receptor_M01_12_4l_125_gate_audit_v1.csv
    data/processed/review_outputs/EC_receptor_M01_12_4l_six_receptor_selection_v1.csv

The five discovery candidates have post-hoc, evidence-adjudicated provisional
roles; plotting colors are NOT evidence of native BBB transcytosis or safety.
LRP1, lacking mouse D02 data, is presented as a separate comparator annotation.

Usage (from Jupyter, with current working directory notebooks/):
    %run -i 14_1_receptor_prioritization_landscape_v2.py
From the project root:
    python notebooks/14_1_receptor_prioritization_landscape_v2.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # deterministic, works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_FILE = "EC_receptor_M01_12_4l_125_gate_audit_v1.csv"
SELECTION_FILE = "EC_receptor_M01_12_4l_six_receptor_selection_v1.csv"
OUT_STEM = "M01_receptor_landscape_14_1_v2"
DATA_STEM = "EC_receptor_M01_14_1_receptor_landscape"

# OmiPep manuscript palette (keep consistent with previously committed figures).
COL = {
    "navy": "#1F4E79",
    "blue": "#4E84C4",
    "green": "#4FAE9D",
    "gold": "#C9A441",
    "coral": "#E58E73",
    "rose": "#F1C7D4",
    "ink": "#233747",
    "muted": "#647586",
    "faint": "#DCE4E9",
    "gray": "#97A5B3",
    "white": "#FFFFFF",
}

PRIMARY_STATUSES = {"provisional_primary_1", "provisional_primary_2"}
RESERVE_STATUSES = {"conditional_reserve_1", "reserve_2", "direction_mismatch_reserve"}
EXPECTED_STATUSES = PRIMARY_STATUSES | RESERVE_STATUSES | {"benchmark_only"}
EXPECTED_DISCOVERY = {"PROCR", "TM4SF1", "SELP", "ECSCR", "ACKR1"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_source(root: Path, name: str) -> Path:
    for directory in ("data/processed/review_outputs", "data/processed/review_inputs"):
        candidate = root / directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required source not found: {name} (checked review_outputs and review_inputs)")


def number_column(df: pd.DataFrame, name: str) -> pd.Series:
    original = df[name].astype(str).str.strip()
    converted = pd.to_numeric(original.replace("", np.nan), errors="coerce")
    malformed = original.ne("") & converted.isna()
    if malformed.any():
        raise ValueError(f"Non-numeric values found in {name}: {df.loc[malformed, name].unique()[:5]}")
    return converted


def load_and_validate(root: Path):
    apath, spath = (find_source(root, x) for x in (AUDIT_FILE, SELECTION_FILE))
    audit = pd.read_csv(apath, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    selected = pd.read_csv(spath, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    required = {
        "relationship_id", "mouse_gene", "human_gene", "D02_logFC", "D02_FDR",
        "human_brain_EC_CPM", "screen_gate", "pass2_role",
    }
    srequired = {"receptor", "role", "final_provisional_status", "D02_logFC", "human_brain_EC_CPM"}
    if not required.issubset(audit.columns):
        raise AssertionError(f"Audit missing fields: {sorted(required - set(audit.columns))}")
    if not srequired.issubset(selected.columns):
        raise AssertionError(f"Selection missing fields: {sorted(srequired - set(selected.columns))}")
    if len(audit) != 125 or audit.relationship_id.nunique() != 125:
        raise AssertionError("Expected exactly 125 unique mouse–human relationship IDs")
    if audit.mouse_gene.nunique() != 114:
        raise AssertionError("Expected exactly 114 distinct mouse genes")
    if len(selected) != 6 or selected.receptor.nunique() != 6:
        raise AssertionError("Expected six selected/contextual receptor rows")
    if set(selected.receptor) != EXPECTED_DISCOVERY | {"LRP1"}:
        raise AssertionError("The selected receptor universe differs from the frozen six-row audit")
    if set(selected.final_provisional_status) != EXPECTED_STATUSES:
        raise AssertionError("One or more selection status codes have changed; review manually")

    for column in ("D02_logFC", "D02_FDR", "human_brain_EC_CPM"):
        audit[column] = number_column(audit, column)
    if (audit.human_brain_EC_CPM.dropna() < 0).any():
        raise AssertionError("CPM cannot be negative")
    if (~audit.D02_FDR.dropna().between(0, 1)).any():
        raise AssertionError("D02 FDR outside [0, 1]")
    mismatched_d02_missing = audit.D02_logFC.isna() != audit.D02_FDR.isna()
    if mismatched_d02_missing.any():
        raise AssertionError("D02 fold-change and FDR are not jointly measured for all records")

    audit["stroke_category"] = np.select(
        [
            audit.D02_logFC.isna(),
            (audit.D02_FDR < 0.05) & (audit.D02_logFC > 0),
            (audit.D02_FDR < 0.05) & (audit.D02_logFC < 0),
        ],
        ["D02 unavailable", "Significantly upregulated", "Significantly downregulated"],
        default="Not significant",
    )
    audit["plotted"] = audit.D02_logFC.notna() & audit.human_brain_EC_CPM.notna()
    audit["plot_y_log10_1plus_CPM"] = np.log10(1.0 + audit.human_brain_EC_CPM)
    audit["plot_exclusion_reason"] = ""
    audit.loc[audit.D02_logFC.isna(), "plot_exclusion_reason"] = "D02 log2FC/FDR unavailable"
    audit.loc[audit.human_brain_EC_CPM.isna(), "plot_exclusion_reason"] = "Human brain EC median CPM unavailable"
    both_missing = audit.D02_logFC.isna() & audit.human_brain_EC_CPM.isna()
    audit.loc[both_missing, "plot_exclusion_reason"] = "Both D02 and human brain EC measurements unavailable"

    counts = Counter(audit.stroke_category)
    required_counts = {
        "Significantly upregulated": 94,
        "Significantly downregulated": 9,
        "Not significant": 18,
        "D02 unavailable": 4,
    }
    if dict(counts) != required_counts:
        raise AssertionError(f"Stroke categories changed: {dict(counts)}; expected {required_counts}")
    if int(audit.plotted.sum()) != 110 or int(audit.human_brain_EC_CPM.isna().sum()) != 11:
        raise AssertionError("Previously verified complete-case or human-expression missingness changed")
    audit["visual_group"] = "Background relationship"
    sel_idx = selected.set_index("receptor")
    if not sel_idx.index.is_unique:
        raise AssertionError("Selected receptor index contains duplicates")
    for gene in EXPECTED_DISCOVERY:
        matching = audit[audit.human_gene.str.upper() == gene]
        if len(matching) != 1:
            raise AssertionError(f"Expected one audit relationship for {gene}, found {len(matching)}")
        r, s = matching.iloc[0], sel_idx.loc[gene]
        if not bool(r.plotted):
            raise AssertionError(f"Discovery receptor {gene} lacks plot coordinates")
        if not math.isclose(r.D02_logFC, float(s.D02_logFC), rel_tol=1e-11):
            raise AssertionError(f"D02 mismatch between selection and audit for {gene}")
        if not math.isclose(r.human_brain_EC_CPM, float(s.human_brain_EC_CPM), rel_tol=1e-11):
            raise AssertionError(f"CPM mismatch between selection and audit for {gene}")
        group = "Provisional primary" if s.final_provisional_status in PRIMARY_STATUSES else "Provisional reserve"
        audit.loc[matching.index, "visual_group"] = group
    lrp = audit[audit.human_gene.str.upper() == "LRP1"]
    if len(lrp) != 1 or not pd.isna(lrp.iloc[0].D02_logFC):
        raise AssertionError("LRP1 must have one entry and no mouse D02 estimate")
    if sel_idx.loc["LRP1", "final_provisional_status"] != "benchmark_only":
        raise AssertionError("LRP1 benchmark status changed")
    audit.loc[lrp.index, "visual_group"] = "Independent benchmark (unplotted)"
    return audit, selected, {"audit": apath, "selection": spath}


def build_figure(df: pd.DataFrame, selected: pd.DataFrame, stem: Path) -> None:
    """Compact editorial scatter: raw data positions; no broken axes, jitter or scoring."""
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",    # leave SVG text editable
        "pdf.fonttype": 42,       # font-based vector PDF
        "savefig.facecolor": "white",
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#AEBBC6",
        "axes.labelcolor": COL["ink"],
        "xtick.color": "#536476",
        "ytick.color": "#536476",
        "text.color": COL["ink"],
    })

    # Intended as a complete standalone panel, with dimensions suited to
    # later reflow into a manuscript multipanel figure. No headline or
    # explanatory paragraphs are embedded in the scientific artwork.
    fig, ax = plt.subplots(figsize=(8.35, 6.52), facecolor="white")
    fig.subplots_adjust(left=0.138, right=0.966, top=0.922, bottom=0.215)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    # y is transformed ONLY for positioning; the original CPM values appear
    # on the ticks. This handles true CPM==0 without silently dropping it.
    tick_cpm = np.array([0, 1, 10, 100, 1000, 10000], dtype=float)
    tick_y = np.log10(1.0 + tick_cpm)
    ax.yaxis.set_major_locator(FixedLocator(tick_y))
    ax.yaxis.set_major_formatter(FixedFormatter(["0", "1", "10", "100", "1,000", "10,000"]))
    ax.yaxis.grid(True, which="major", lw=0.67, color="#DCE4E9", alpha=0.67)
    ax.axvline(0, color="#8294A4", lw=0.87, ls=(0, (3, 4)), alpha=0.75, zorder=0)

    plotted = df[df.plotted].copy()
    selected_data = plotted[plotted.visual_group.isin(["Provisional primary", "Provisional reserve"])]
    ordinary = plotted[plotted.visual_group == "Background relationship"]

    # Larger, distinct symbols than v1; one marker per relationship.
    styles = {
        "Significantly upregulated": dict(marker="o", s=57, fc=COL["blue"], alpha=0.77),
        "Significantly downregulated": dict(marker="v", s=69, fc=COL["coral"], alpha=0.87),
        "Not significant": dict(marker="D", s=49, fc="#A1ABB5", alpha=0.77),
    }
    for category in ("Not significant", "Significantly downregulated", "Significantly upregulated"):
        part = ordinary[ordinary.stroke_category == category]
        sty = styles[category]
        ax.scatter(part.D02_logFC, part.plot_y_log10_1plus_CPM,
                   s=sty["s"], marker=sty["marker"],
                   color=sty["fc"], alpha=sty["alpha"],
                   edgecolor="white", linewidth=0.58,
                   zorder=2, rasterized=False)

    # Unambiguous visual status: two provisional primaries (teal rings),
    # three reserves (gold diamonds). Symbols do NOT establish transport.
    for _, row in selected_data.iterrows():
        is_primary = row.visual_group == "Provisional primary"
        color = COL["green"] if is_primary else COL["gold"]
        shape = "o" if is_primary else "D"
        ax.scatter([row.D02_logFC], [row.plot_y_log10_1plus_CPM],
                   marker=shape, s=212 if is_primary else 196,
                   color=color, edgecolor="white", linewidth=2.2,
                   zorder=5, rasterized=False)
        ax.scatter([row.D02_logFC], [row.plot_y_log10_1plus_CPM],
                   marker=shape, s=219 if is_primary else 205,
                   facecolor="none", edgecolor=COL["ink"], linewidth=0.62,
                   zorder=6, rasterized=False)

    # Label text is deliberately gene-only. Role, limitations, and evidence
    # belong in the legend and source-linked caption, not callout boxes.
    # These positions use the data coordinate system and are fixed for the
    # previously reviewed five receptors; leaders identify exact markers.
    label_positions = {
        "TM4SF1": (1.78, 3.28, "left", "bottom"),
        "PROCR": (0.56, 1.74, "right", "top"),
        "ECSCR": (2.65, 2.61, "left", "bottom"),
        "ACKR1": (3.06, 1.63, "left", "top"),
        "SELP": (6.62, 0.54, "right", "bottom"),
    }
    for gene in ("TM4SF1", "PROCR", "ECSCR", "ACKR1", "SELP"):
        target = selected_data.loc[selected_data.human_gene.str.upper().eq(gene)]
        if len(target) != 1:
            raise AssertionError(f"Exactly one plotted receptor expected: {gene}")
        row = target.iloc[0]
        primary = row.visual_group == "Provisional primary"
        color = COL["green"] if primary else "#A38229"  # darker gold for text contrast
        x, y, ha, va = label_positions[gene]
        annotation = ax.annotate(
            gene,
            xy=(row.D02_logFC, row.plot_y_log10_1plus_CPM), xycoords="data",
            xytext=(x, y), textcoords="data",
            ha=ha, va=va, fontsize=11.05, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.94, alpha=0.82,
                            shrinkA=3.2, shrinkB=10,
                            connectionstyle="arc3,rad=0"),
            zorder=10,
        )
        annotation.set_path_effects([patheffects.withStroke(linewidth=3.3, foreground="white")])

    # Include the real extreme right tail; intentionally NO broken x-axis.
    ax.set_xlim(-2.43, 8.08)
    ax.set_ylim(-0.09, 4.08)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.tick_params(axis="both", labelsize=10, length=4.2, width=0.72, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B0BDC8")
    ax.set_xlabel("Mouse EC response, D02 vs Sham (log₂FC)", fontsize=11.45, labelpad=11)
    ax.set_ylabel("Human brain EC median expression (CPM)", fontsize=11.45, labelpad=10)

    # Actual class totals INCLUDE highlighted receptors; all are points, not
    # additional entities. Layout deliberately uses short readable labels.
    n = Counter(plotted.stroke_category)
    def leg(marker, color, label, size=8.4):
        return Line2D([], [], marker=marker, linestyle="None", markersize=size,
                      markerfacecolor=color, markeredgecolor="white",
                      markeredgewidth=0.7, label=label)
    handles = [
        leg("o", COL["blue"], f"D02 up  ·  {n['Significantly upregulated']}", 8.4),
        leg("v", COL["coral"], f"D02 down  ·  {n['Significantly downregulated']}", 8.6),
        leg("D", "#A1ABB5", f"Not significant  ·  {n['Not significant']}", 8.0),
        leg("o", COL["green"], "Provisional primary  ·  2", 9),
        leg("D", COL["gold"], "Reserve  ·  3", 8.5),
    ]
    # Two unobtrusive legend rows avoid dense 5-column microtype.
    first = ax.legend(handles=handles[:3], ncol=3,
                      bbox_to_anchor=(0, -0.158), loc="upper left", frameon=False,
                      fontsize=9.0, handletextpad=0.46, columnspacing=1.45,
                      borderaxespad=0)
    ax.add_artist(first)
    ax.legend(handles=handles[3:], ncol=2,
              bbox_to_anchor=(0, -0.229), loc="upper left", frameon=False,
              fontsize=9.0, handletextpad=0.46, columnspacing=1.55,
              borderaxespad=0)
    # Compact, low-contrast note: all methodological details remain caption-only.
    fig.text(0.965, 0.055, "LRP1 benchmark omitted: D02 unavailable",
             ha="right", va="center", fontsize=8.4, color=COL["muted"])

    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 450})):
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight",
                    pad_inches=0.14, **kwargs)
    plt.close(fig)


def write_outputs(root: Path, audit: pd.DataFrame, selection: pd.DataFrame, paths: dict[str, Path]):
    figure_dir = root / "results/figures/module_01"
    evidence_dir = root / "data/processed/review_outputs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figure_dir / OUT_STEM
    out_csv = evidence_dir / (DATA_STEM + "_source_v2.csv")
    out_json = evidence_dir / (DATA_STEM + "_validation_v2.json")
    out_caption = evidence_dir / (DATA_STEM + "_caption_v2.txt")

    columns = [
        "relationship_id", "mouse_gene", "human_gene", "D02_logFC", "D02_FDR",
        "human_brain_EC_CPM", "plot_y_log10_1plus_CPM", "stroke_category",
        "visual_group", "plotted", "plot_exclusion_reason", "screen_gate", "pass2_role",
    ]
    source = audit[columns].copy().sort_values("relationship_id", kind="stable")
    source.to_csv(out_csv, index=False, float_format="%.15g", na_rep="")
    build_figure(audit, selection, out_fig)

    categories = {k: int(v) for k, v in sorted(Counter(audit.stroke_category).items())}
    shown_categories = {k: int(v) for k, v in sorted(Counter(audit.loc[audit.plotted, "stroke_category"]).items())}
    manifest = {
        "figure_id": "OmiPep_Module01_Cell14.1_v2",
        "visualization_revision": "v2; 4:3 compact editorial layout, continuous x-axis, minimal labels",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {k: sha256(p) for k, p in paths.items()},
        "input_paths": {k: str(p.relative_to(root)) for k, p in paths.items()},
        "relationship_total": len(audit),
        "unique_mouse_genes": int(audit.mouse_gene.nunique()),
        "stroke_categories_full_125": categories,
        "plotted_relationships": int(audit.plotted.sum()),
        "plotted_stroke_categories": shown_categories,
        "missing_D02": int(audit.D02_logFC.isna().sum()),
        "missing_human_CPM": int(audit.human_brain_EC_CPM.isna().sum()),
        "missing_both": int((audit.D02_logFC.isna() & audit.human_brain_EC_CPM.isna()).sum()),
        "excluded_from_coordinates": int((~audit.plotted).sum()),
        "provisional_primaries": ["PROCR", "TM4SF1"],
        "provisional_reserves": ["SELP", "ECSCR", "ACKR1"],
        "benchmark_only": "LRP1 (D02 unavailable)",
        "numeric_y_transform": "log10(1 + human_brain_EC_CPM)",
        "displayed_y_ticks": [0, 1, 10, 100, 1000, 10000],
        "interpretation_limitations": [
            "Post-hoc manually adjudicated receptor roles; no prospective automated optimization claimed",
            "Relationship records sharing one mouse gene do not constitute independent mouse DE observations",
            "No claim of native human stroke-luminal protein expression, human intact cargo transcytosis, or receptor safety",
            "Independent benchmark LRP1 excluded from D02 quantification without imputation",
        ],
        "outputs": [str(out_csv.relative_to(root)), str(out_caption.relative_to(root)),
                    *[str(out_fig.with_suffix('.' + ext).relative_to(root)) for ext in ("svg", "pdf", "png")]],
        "input_files_unchanged": True,
        "no_broken_axis": True,
        "no_jitter_or_imputation": True,
        "selection_approval_claimed": False,
    }
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    caption = (
        "OmiPep Module 01 / Cell 14.1 (v2) — Mouse stroke and human endothelial expression. "
        "Each point represents one mouse–human relationship from the frozen 125-row screen "
        "(114 distinct mouse genes), positioned by replicate-aware mouse endothelial "
        "D02-versus-Sham log2 fold change (x) and independent human brain endothelial "
        "median CPM (y; log10[1+CPM] transformation, axis labels in original CPM units). "
        "The complete-case plot contains 110 relationships. Colors/shapes distinguish "
        "D02 FDR<0.05 upregulation and downregulation from nonsignificant changes. "
        "PROCR and TM4SF1 are highlighted as provisional design targets; SELP, ECSCR and "
        "ACKR1 are provisional alternatives, based on separate literature-informed, "
        "post-hoc biological adjudication, not a model-derived threshold. "
        "Eleven relationships lack human CPM and four lack D02 measurements; none were "
        "imputed. LRP1 is described separately as the independent benchmark because "
        "its D02 data are unavailable. Expression associations do not establish native "
        "luminal protein accessibility, receptor-mediated transcytosis, intact engineered "
        "peptide delivery, or clinical safety. Underlying per-relationship plotted and "
        "unplotted records are provided in the accompanying source CSV."
    )
    out_caption.write_text(caption + "\n", encoding="utf-8")
    print(f"PASS | Cell 14.1: 125 relationships / 114 mouse genes; {int(audit.plotted.sum())} plotted; 15 excluded without imputation")
    print(f"PASS | D02 full categories: {categories}")
    print(f"PASS | Five provisional discovery candidates highlighted; LRP1 benchmark separate")
    print(f"FIG  | {out_fig}.{{svg,pdf,png}}")
    print(f"DATA | {out_csv.relative_to(root)}")
    print(f"META | {out_json.relative_to(root)}; {out_caption.relative_to(root)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Repo root, inferred by default from script in notebooks/")
    opt = parser.parse_args()
    root = (opt.root or Path(__file__).resolve().parent.parent).resolve()
    audit, selection, input_paths = load_and_validate(root)
    before = {name: sha256(path) for name, path in input_paths.items()}
    write_outputs(root, audit, selection, input_paths)
    after = {name: sha256(path) for name, path in input_paths.items()}
    assert before == after, "A read-only source file changed unexpectedly"


if __name__ == "__main__":
    main()
