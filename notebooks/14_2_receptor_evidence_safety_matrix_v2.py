#!/usr/bin/env python3
"""OmiPep Module 01 Cell 14.2 v2: compact receptor evidence / liability panel.

Creates a compact, publication-oriented summary for the five provisional discovery
receptors plus the independent LRP1 benchmark. This version is deliberately
minimal so it can later become a small panel within an integrated multi-panel
main figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from textwrap import fill

INPUTS = {
    "pub_table": "EC_receptor_M01_12_4m_six_receptor_publication_table_v1.csv",
    "selection": "EC_receptor_M01_12_4l_six_receptor_selection_v1.csv",
    "ledger": "EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv",
}
ORDER = ["PROCR", "TM4SF1", "SELP", "ECSCR", "ACKR1", "LRP1"]
STATUS = {
    "provisional_primary_1": ("PRIMARY 01", "#67AFA0"),
    "provisional_primary_2": ("PRIMARY 02", "#67AFA0"),
    "conditional_reserve_1": ("COND. RESERVE", "#C8A23F"),
    "reserve_2": ("RESERVE", "#C8A23F"),
    "direction_mismatch_reserve": ("DIRECTIONAL RESERVE", "#C8A23F"),
    "benchmark_only": ("BENCHMARK", "#5D86C7"),
}
# Compact scientific summaries for the main-figure style panel.
COMPACT = {
    "PROCR": {
        "delivery": "Native APC blood→brain\n(mouse BBB)",
        "liability": "APC / PAR1 pathway",
        "peripheral_site": "Thymus",
        "peripheral_ntpm": 339.6,
        "evidence_class": "direct",
    },
    "TM4SF1": {
        "delivery": "Antibody uptake only\n(no BBB transport)",
        "liability": "Vascular repair;\nglycan/species epitope",
        "peripheral_site": "Skin",
        "peripheral_ntpm": 5464.6,
        "evidence_class": "limited",
    },
    "SELP": {
        "delivery": "Fucoidan NP blood→brain\n(tumor BBB)",
        "liability": "Thrombo-inflammation;\nleak context",
        "peripheral_site": "Lung",
        "peripheral_ntpm": 158.2,
        "evidence_class": "contextual",
    },
    "ECSCR": {
        "delivery": "KDR-linked trafficking only\n(no shuttle shown)",
        "liability": "VEGF / KDR signaling",
        "peripheral_site": "Breast",
        "peripheral_ntpm": 1202.2,
        "evidence_class": "contextual",
    },
    "ACKR1": {
        "delivery": "Reverse chemokine flux\n(brain→blood)",
        "liability": "Chemokine axis;\nerythroid sink",
        "peripheral_site": "Fallopian tube",
        "peripheral_ntpm": 2761.6,
        "evidence_class": "reverse",
    },
    "LRP1": {
        "delivery": "Angiopep-2 / L57\nbenchmark comparator",
        "liability": "Broad ligand clearance",
        "peripheral_site": "Liver",
        "peripheral_ntpm": 49.1,
        "evidence_class": "contextual",
    },
}
OUT_PREFIX = "EC_receptor_M01_14_2_evidence_safety_matrix"
FIG_STEM = "M01_receptor_evidence_safety_matrix_14_2_v2"


def locate(root: Path, name: str) -> Path:
    candidates = [
        root / "data/processed/review_outputs" / name,
        root / "data/processed/review_inputs" / name,
        root / "data/processed" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing required source: {name}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        out = list(csv.DictReader(f))
    if not out:
        raise ValueError(f"Empty input: {path}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def wrap(s: str, n: int) -> str:
    return fill(str(s), width=n, break_long_words=False, break_on_hyphens=False)


def infer_font() -> str:
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Arial", "Liberation Sans", "DejaVu Sans"]:
        if cand in names:
            return cand
    return "sans-serif"


def create_figure(rows: list[dict[str, str]], path_stem: Path, font_name: str) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": font_name,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    navy = "#244C84"
    text = "#223648"
    muted = "#6A7A89"
    grid = "#D8E0E7"
    light = "#F6F8FB"
    pill_fill = "#F4F6F9"
    no_fill = "#EEF2F6"
    no_edge = "#B7C1CC"
    direct = "#67AFA0"
    gold = "#C8A23F"
    reverse = "#D98E6B"
    blue = "#5D86C7"

    fig = plt.figure(figsize=(12.6, 5.3), facecolor='white')
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Layout
    x_name = 0.03
    x_del = 0.255
    x_liab = 0.545
    x_peri = 0.812
    w_name = 0.17
    w_del = 0.24
    w_liab = 0.215
    w_peri = 0.15
    y_top = 0.86
    row_h = 0.11

    ax.text(0.03, 0.955, "ENDOTHELIAL RECEPTOR EVIDENCE", fontsize=14.5, fontweight='bold', color=navy, va='bottom')
    ax.text(x_del, 0.91, "DELIVERY PRECEDENT", fontsize=10.8, fontweight='bold', color=navy, va='bottom', ha='center')
    ax.text(x_liab, 0.91, "KEY LIABILITY", fontsize=10.8, fontweight='bold', color=navy, va='bottom', ha='center')
    ax.text(x_peri, 0.91, "PERIPHERAL EC RNA", fontsize=10.8, fontweight='bold', color=navy, va='bottom', ha='left')
    ax.text(x_del, 0.875, "what is actually shown", fontsize=8.6, color=muted, va='bottom', ha='center')
    ax.text(x_liab, 0.875, "main translational constraint", fontsize=8.6, color=muted, va='bottom', ha='center')
    ax.text(x_peri, 0.875, "HPA peak nTPM · log scale", fontsize=8.6, color=muted, va='bottom', ha='left')

    ax.plot([x_del - 0.11, x_del + 0.11], [0.895, 0.895], lw=2.2, color=direct)
    ax.plot([x_liab - 0.09, x_liab + 0.09], [0.895, 0.895], lw=2.2, color=reverse)
    ax.plot([x_peri, x_peri + 0.15], [0.895, 0.895], lw=2.2, color=blue)

    values = [COMPACT[g]["peripheral_ntpm"] for g in ORDER]
    logmin = math.log10(min(values))
    logmax = math.log10(max(values))

    def ntpm_to_x(v: float) -> float:
        f = (math.log10(v) - logmin) / (logmax - logmin)
        return x_peri + 0.01 + f * 0.09

    text_nodes = 0
    for i, gene in enumerate(ORDER):
        r = next(rr for rr in rows if rr['receptor'] == gene)
        yy = y_top - i * row_h
        ymid = yy - row_h/2 + 0.003
        status_lbl, status_color = STATUS[r['status_code']]
        # row divider
        if i == 5:
            ax.plot([0.03, 0.97], [yy+0.02, yy+0.02], color=grid, lw=1.0, ls=(0,(2.5,3.5)))

        # name block
        ax.add_patch(Rectangle((x_name, yy-row_h+0.013), 0.0035, row_h-0.026, color=status_color, lw=0))
        ax.text(x_name + 0.01, ymid + 0.014, gene, fontsize=14.2, fontweight='bold', color=navy, va='center'); text_nodes += 1
        ax.text(x_name + 0.01, ymid - 0.028, status_lbl, fontsize=8.7, color=status_color, va='center'); text_nodes += 1

        # delivery pill
        del_color = {
            'direct': direct,
            'limited': muted,
            'contextual': gold,
            'reverse': reverse,
        }[COMPACT[gene]['evidence_class']]
        pill = FancyBboxPatch((x_del-0.095, yy-row_h+0.025), 0.19, row_h-0.05,
                              boxstyle="round,pad=0.006,rounding_size=0.008",
                              facecolor=pill_fill, edgecolor=grid, linewidth=0.8)
        ax.add_patch(pill)
        ax.add_patch(Rectangle((x_del-0.093, yy-row_h+0.028), 0.0022, row_h-0.056, color=del_color, lw=0))
        marker = {'direct':'●', 'limited':'—', 'contextual':'◌', 'reverse':'◆'}[COMPACT[gene]['evidence_class']]
        ax.text(x_del-0.084, ymid+0.002, marker, fontsize=15, color=del_color, va='center', ha='left'); text_nodes += 1
        ax.text(x_del-0.066, ymid+0.002, wrap(COMPACT[gene]['delivery'], 18), fontsize=10.4, color=text, va='center', ha='left'); text_nodes += 1

        # liability pill
        lp = FancyBboxPatch((x_liab-0.08, yy-row_h+0.025), 0.16, row_h-0.05,
                            boxstyle="round,pad=0.006,rounding_size=0.008",
                            facecolor="#F8F4F1", edgecolor=grid, linewidth=0.8)
        ax.add_patch(lp)
        ax.add_patch(Rectangle((x_liab-0.078, yy-row_h+0.028), 0.0022, row_h-0.056, color=reverse, lw=0))
        ax.text(x_liab-0.068, ymid+0.002, wrap(COMPACT[gene]['liability'], 18), fontsize=10.4, color=text, va='center', ha='left'); text_nodes += 1

        # peripheral track
        ax.text(x_peri, ymid-0.028, COMPACT[gene]['peripheral_site'], fontsize=9.4, color=muted, va='center', ha='left'); text_nodes += 1
        ax.plot([x_peri, x_peri+0.10], [ymid+0.012, ymid+0.012], color=grid, lw=5.5, solid_capstyle='round')
        xdot = ntpm_to_x(COMPACT[gene]['peripheral_ntpm'])
        ax.plot([x_peri, xdot], [ymid+0.012, ymid+0.012], color=blue, lw=5.5, solid_capstyle='round')
        ax.scatter([xdot], [ymid+0.012], s=120, color=navy, edgecolor='white', linewidth=0.8, zorder=3)
        ax.text(x_peri+0.118, ymid+0.012, f"{COMPACT[gene]['peripheral_ntpm']:,.1f}", fontsize=11, fontweight='bold', color=navy, va='center', ha='left'); text_nodes += 1

    # small axis for log track
    tick_vals = [10, 100, 1000, 10000]
    for tv in tick_vals:
        xx = ntpm_to_x(tv)
        ax.plot([xx, xx], [0.145, 0.155], color=grid, lw=0.8)
        ax.text(xx, 0.125, f"{tv:g}" if tv < 1000 else ("1k" if tv == 1000 else "10k"), fontsize=8.4, color=muted, ha='center', va='top'); text_nodes += 1

    # Legend / note
    yleg = 0.06
    legend_items = [
        (direct, '●', 'direct / model-specific precedent'),
        (gold, '◌', 'contextual / indirect precedent'),
        (reverse, '◆', 'opposite-direction precedent'),
        (muted, '—', 'no BBB transport shown'),
    ]
    x = 0.035
    for color, marker, label in legend_items:
        ax.text(x, yleg, marker, fontsize=16, color=color, va='center', ha='left'); text_nodes += 1
        ax.text(x+0.02, yleg, label, fontsize=9.6, color=text, va='center', ha='left'); text_nodes += 1
        x += 0.21
    ax.text(0.03, 0.015,
            "Binder safety and intact engineered-peptide BBB delivery remain unverified.",
            fontsize=9.6, color=navy, fontweight='bold', va='bottom'); text_nodes += 1

    for ext, kwargs in [(".svg", {}), (".pdf", {}), (".png", {"dpi": 450})]:
        fig.savefig(path_stem.with_suffix(ext), bbox_inches='tight', pad_inches=0.06, **kwargs)
    plt.close(fig)
    return text_nodes


def build_caption(rows: list[dict[str, str]]) -> str:
    return (
        "Compact source-linked comparison of the five provisional discovery receptors and the "
        "independent LRP1 benchmark. The delivery-precedent column summarizes the strongest "
        "reviewed evidence actually shown in the primary literature, rather than claiming that "
        "any receptor is an approved peptide shuttle. The liability column summarizes the main "
        "pathway or targeting constraint that would need to be addressed during binder design. "
        "Peripheral endothelial RNA is shown as the peak Human Protein Atlas endothelial-cluster "
        "nTPM on a log scale, providing exposure context rather than a validated safety score. "
        "No receptor is claimed here to have verified intact engineered-peptide blood-to-brain delivery."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--figure-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    root = (args.root or Path(os.environ.get("OMIPEP_ROOT", Path(__file__).resolve().parent.parent))).resolve()
    fig_dir = (args.figure_dir or root / "results/figures/module_01").resolve()
    out_dir = (args.out_dir or root / "data/processed/review_outputs").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {k: locate(root, v) for k, v in INPUTS.items()}
    hashes_before = {k: sha256(v) for k, v in input_paths.items()}
    pub = load_csv(input_paths['pub_table'])
    selection = load_csv(input_paths['selection'])
    ledger = load_csv(input_paths['ledger'])

    if len(pub) != 6 or {r['receptor'] for r in pub} != set(ORDER):
        raise AssertionError('Expected six receptor rows from 12.4m publication table')
    if len(selection) != 6 or {r['receptor'] for r in selection} != set(ORDER):
        raise AssertionError('Expected six receptor rows from 12.4l selection table')
    if len(ledger) != 19:
        raise AssertionError(f'Expected 19 full-text paper rows, found {len(ledger)}')

    sel_map = {r['receptor']: r for r in selection}
    pub_map = {r['receptor']: r for r in pub}
    source_rows = []
    for gene in ORDER:
        if sel_map[gene]['final_provisional_status'] not in STATUS:
            raise AssertionError(f'Unexpected status code for {gene}')
        row = {
            'receptor': gene,
            'role': sel_map[gene]['role'],
            'status_code': sel_map[gene]['final_provisional_status'],
            'delivery_precedent_short': COMPACT[gene]['delivery'],
            'liability_short': COMPACT[gene]['liability'],
            'peripheral_site': COMPACT[gene]['peripheral_site'],
            'peripheral_ntpm': COMPACT[gene]['peripheral_ntpm'],
            'evidence_class': COMPACT[gene]['evidence_class'],
            'reviewed_dois': pub_map[gene]['reviewed_dois'],
            'paper_count': pub_map[gene]['paper_count'],
            'evidence_source_relationship_id': pub_map[gene]['evidence_source_relationship_id'],
        }
        source_rows.append(row)

    source_csv = out_dir / f"{OUT_PREFIX}_source_v2.csv"
    write_csv(source_csv, source_rows, list(source_rows[0].keys()))
    caption_path = out_dir / f"{OUT_PREFIX}_caption_v2.txt"
    caption_path.write_text(build_caption(source_rows), encoding='utf-8')

    font_name = infer_font()
    text_nodes = create_figure(source_rows, fig_dir / FIG_STEM, font_name)

    validation = {
        'figure_id': 'Cell 14.2 v2',
        'inputs': {k: str(v) for k, v in input_paths.items()},
        'input_sha256': hashes_before,
        'receptor_count': len(source_rows),
        'paper_count': len(ledger),
        'font_used': font_name,
        'svg_text_nodes': text_nodes,
        'notes': [
            'Compact redesign intended as a small main-figure panel candidate.',
            'Delivery precedent is summarized conservatively and does not imply approved shuttle function.',
            'Peripheral HPA endothelial RNA provides exposure context only.',
            'No approved engineered peptide blood-to-brain shuttle is claimed for any receptor.'
        ],
        'outputs': {
            'figure_svg': str(fig_dir / (FIG_STEM + '.svg')),
            'figure_pdf': str(fig_dir / (FIG_STEM + '.pdf')),
            'figure_png': str(fig_dir / (FIG_STEM + '.png')),
            'source_csv': str(source_csv),
            'caption_txt': str(caption_path),
        }
    }
    val_path = out_dir / f"{OUT_PREFIX}_validation_v2.json"
    val_path.write_text(json.dumps(validation, indent=2), encoding='utf-8')

    print(f"PASS | Cell 14.2 v2: {len(source_rows)} receptors; {len(ledger)} full-text papers")
    print("PASS | Compact main-figure panel redesign; no receptor approved as peptide shuttle")
    print(f"FONT | {font_name} (editable SVG, sans-serif fallback; {text_nodes} text nodes)")
    print(f"FIG  | {fig_dir / (FIG_STEM + '.{svg,pdf,png}')}")
    print(f"DATA | {source_csv}")
    print(f"META | {val_path}; {caption_path}")


if __name__ == '__main__':
    main()
