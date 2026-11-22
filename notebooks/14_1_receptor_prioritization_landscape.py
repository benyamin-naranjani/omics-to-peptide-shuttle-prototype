#!/usr/bin/env python3
"""OmiPep v0.2 / Module 01 / Cell 14.1.

Create a reproducible, standalone publication figure of the complete mouse D02
stroke-response vs independent human brain endothelial RNA abundance landscape.

Inputs (read only; generated in earlier Module 01 cells):
    data/processed/review_outputs/EC_receptor_M01_12_4l_125_gate_audit_v1.csv
    data/processed/review_outputs/EC_receptor_M01_12_4l_six_receptor_selection_v1.csv

The five discovery candidates have post-hoc, evidence-adjudicated provisional
roles; plotting colors are NOT evidence of native BBB transcytosis or safety.
LRP1, lacking mouse D02 data, is presented as a separate comparator annotation.

Usage (from Jupyter, with current working directory notebooks/):
    %run -i 14_1_receptor_prioritization_landscape.py
From the project root:
    python notebooks/14_1_receptor_prioritization_landscape.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

AUDIT_FILE = "EC_receptor_M01_12_4l_125_gate_audit_v1.csv"
SELECTION_FILE = "EC_receptor_M01_12_4l_six_receptor_selection_v1.csv"
OUT_STEM = "M01_receptor_landscape_14_1_v1"
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

# Fixed label positions for the reviewed five; coordinates are annotations,
# not measured data. They can be adjusted when composing the full manuscript.
LABEL_POS = {
    "TM4SF1": (1.56, 3.43),
    "PROCR": (-0.45, 2.43),
    "ECSCR": (2.88, 2.76),
    "ACKR1": (3.30, 1.57),
    "SELP": (5.58, 0.64),
}


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
    sel_idx = selected.set_index("receptor", verify_integrity=True)
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
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.4,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.2,
        "axes.edgecolor": "#B5C5CF",
        "axes.linewidth": 0.8,
        "xtick.color": COL["muted"],
        "ytick.color": COL["muted"],
        "text.color": COL["ink"],
        "axes.labelcolor": COL["ink"],
        "svg.fonttype": "none",   # keep text labels editable
        "pdf.fonttype": 42,     # embed TrueType fonts, avoid path outlines
        "savefig.facecolor": COL["white"],
    })

    fig = plt.figure(figsize=(11.3, 7.15), facecolor=COL["white"])
    fig.text(0.092, 0.961, "OMIPEP  /  MODULE 01  /  CROSS-SPECIES RECEPTOR LANDSCAPE",
             color=COL["navy"], weight="bold", fontsize=9.1, va="top")
    fig.text(0.092, 0.926, "Stroke responsiveness meets human brain endothelial expression",
             color=COL["ink"], weight="bold", fontsize=16.3, va="top")
    fig.text(0.092, 0.887,
             "125 mouse–human relationships from 114 genes  ·  110 with both quantitative measurements",
             color=COL["muted"], fontsize=10.3, va="top")
    fig.text(0.092, 0.849, "LRP1: independent benchmark, not plotted (D02 unavailable)",
             fontsize=9.4, color=COL["navy"], va="center", weight="medium",
             bbox={"boxstyle": "round,pad=0.45,rounding_size=0.2",
                   "facecolor": "#EEF3F8", "edgecolor": "#CDDBE6", "linewidth": 0.65})

    ax = fig.add_axes([0.102, 0.267, 0.841, 0.543])
    ax.set_facecolor(COL["white"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color=COL["faint"], linewidth=0.67, alpha=0.78)
    ax.xaxis.grid(False)
    ax.axvline(0, linewidth=0.92, color=COL["muted"], linestyle=(0, (3, 3)), alpha=0.64, zorder=1)

    base = df[df.plotted].copy()
    category_style = (
        ("Not significant", "D", COL["gray"], 32, 0.55),
        ("Significantly downregulated", "v", COL["coral"], 38, 0.76),
        ("Significantly upregulated", "o", COL["blue"], 39, 0.53),
    )
    for category, marker, color, size, alpha in category_style:
        part = base[base.stroke_category == category]
        ax.scatter(part.D02_logFC, part.plot_y_log10_1plus_CPM,
                   s=size, marker=marker, facecolors=color, edgecolors="white",
                   linewidths=0.50, alpha=alpha, zorder=2, rasterized=False)

    active = base[base.visual_group.isin(["Provisional primary", "Provisional reserve"])]
    for _, r in active.iterrows():
        primary = r.visual_group == "Provisional primary"
        col = COL["green"] if primary else COL["gold"]
        marker = "o" if primary else "D"
        ax.scatter([r.D02_logFC], [r.plot_y_log10_1plus_CPM], s=365,
                   c=[col], alpha=0.13, linewidths=0, zorder=4)
        ax.scatter([r.D02_logFC], [r.plot_y_log10_1plus_CPM], s=120,
                   marker=marker, c=[col], edgecolors="white", linewidths=1.7, zorder=6)
        ax.scatter([r.D02_logFC], [r.plot_y_log10_1plus_CPM], s=128,
                   marker=marker, facecolors="none", edgecolors=COL["ink"], linewidths=0.38, zorder=7)

    # Annotation positions deliberately avoid portraying selection as a ranked score.
    for gene in ["TM4SF1", "PROCR", "ECSCR", "ACKR1", "SELP"]:
        r = active.loc[active.human_gene.str.upper() == gene].iloc[0]
        primary = r.visual_group == "Provisional primary"
        color = COL["green"] if primary else COL["gold"]
        status = "PRIMARY · PROVISIONAL" if primary else "RESERVE · PROVISIONAL"
        pos = LABEL_POS[gene]
        ax.annotate(f"{gene}\n{status}",
                    xy=(r.D02_logFC, r.plot_y_log10_1plus_CPM), xycoords="data",
                    xytext=pos, textcoords="data", ha="left", va="center",
                    fontsize=9.05, fontweight="bold", color=COL["ink"],
                    linespacing=1.42, zorder=9,
                    bbox={"boxstyle": "round,pad=0.36,rounding_size=0.18",
                          "fc": "white", "ec": color, "lw": 1.03, "alpha": 0.98},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.05,
                                "shrinkA": 6, "shrinkB": 9,
                                "connectionstyle": "arc3,rad=0.0"})

    ax.set_xlim(-2.45, 8.55)
    ax.set_ylim(-0.08, 4.21)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.set_xlabel("Mouse endothelial D02 vs Sham  ·  log₂ fold change", labelpad=10)
    ax.set_yticks(np.log10(1 + np.array([0, 1, 10, 100, 1000, 10000], dtype=float)))
    ax.set_yticklabels(["0", "1", "10", "100", "1,000", "10,000"])
    ax.set_ylabel("Human brain EC median CPM  ·  log₁₀(1 + CPM) scale", labelpad=10)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, labelsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    counts = Counter(base.stroke_category)
    handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=7.2,
               markerfacecolor=COL["blue"], markeredgecolor="white",
               label=f"D02 up (n={counts['Significantly upregulated']})"),
        Line2D([], [], marker="v", linestyle="None", markersize=7.2,
               markerfacecolor=COL["coral"], markeredgecolor="white",
               label=f"D02 down (n={counts['Significantly downregulated']})"),
        Line2D([], [], marker="D", linestyle="None", markersize=6.4,
               markerfacecolor=COL["gray"], markeredgecolor="white",
               label=f"D02 nonsig. (n={counts['Not significant']})"),
        Line2D([], [], marker="o", linestyle="None", markersize=8.7,
               markerfacecolor=COL["green"], markeredgecolor="white",
               label="Primary (n=2)"),
        Line2D([], [], marker="D", linestyle="None", markersize=7.3,
               markerfacecolor=COL["gold"], markeredgecolor="white",
               label="Reserve (n=3)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.091, 0.149),
               ncol=5, frameon=False, handlelength=0.9, columnspacing=1.32,
               handletextpad=0.48, fontsize=8.45)
    fig.text(0.092, 0.111,
             "15 unplotted relationships: 11 lack human brain EC CPM; 4 lack D02 data (including LRP1). No missing value was set to zero.",
             fontsize=9.05, color=COL["muted"], ha="left", va="center")
    fig.text(0.092, 0.080,
             "Categories use D02 FDR < 0.05. Each point is a relationship, not an independent receptor or validated transport mechanism.",
             fontsize=9.05, color=COL["muted"], ha="left", va="center")
    fig.text(0.092, 0.049,
             "Discovery roles are evidence-informed, post-hoc and provisional; native luminal accessibility and intact peptide transport remain unproven.",
             fontsize=8.9, color=COL["muted"], ha="left", va="center")

    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 450})):
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", pad_inches=0.16, **kwargs)
    plt.close(fig)


def write_outputs(root: Path, audit: pd.DataFrame, selection: pd.DataFrame, paths: dict[str, Path]):
    figure_dir = root / "results/figures/module_01"
    evidence_dir = root / "data/processed/review_outputs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figure_dir / OUT_STEM
    out_csv = evidence_dir / (DATA_STEM + "_source_v1.csv")
    out_json = evidence_dir / (DATA_STEM + "_validation_v1.json")
    out_caption = evidence_dir / (DATA_STEM + "_caption_v1.txt")

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
        "figure_id": "OmiPep_Module01_Cell14.1",
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
        "selection_approval_claimed": False,
    }
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    caption = (
        "OmiPep Module 01 / Cell 14.1 — Cross-species endothelial receptor landscape. "
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
