#!/usr/bin/env python3
"""OmiPep Module 01 Cell 12.4m: source-linked six-receptor manuscript table.

No re-ranking, no new receptor approval, and no modification of input CSVs.
Dependencies: Python 3.11+, matplotlib. Source data from Cells 12.4k/12.4l.
Outputs: 6-row publication CSV, 19-DOI reference map, SVG/PDF/PNG figure,
figure caption and a provenance/validation JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

INPUTS = {
    "selection": "EC_receptor_M01_12_4l_six_receptor_selection_v1.csv",
    "paper_ledger": "EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv",
    "gate_audit": "EC_receptor_M01_12_4l_125_gate_audit_v1.csv",
    "master": "EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv",
}
ORDER = ["PROCR", "TM4SF1", "SELP", "ECSCR", "ACKR1", "LRP1"]
STATUS = {
    "provisional_primary_1": ("PROVISIONAL 1", "#4FAE9D"),
    "provisional_primary_2": ("PROVISIONAL 2", "#4FAE9D"),
    "conditional_reserve_1": ("RESERVE 1", "#C9A441"),
    "reserve_2": ("RESERVE 2", "#C9A441"),
    "direction_mismatch_reserve": ("DIRECTION MISMATCH", "#E58E73"),
    "benchmark_only": ("BENCHMARK", "#4E84C4"),
}
# Text in these columns is an explicitly labelled scientific interpretation of
# the FULL-TEXT source-backed review, not a quantified or trained score.
CURATED = {
    "PROCR": {
        "evidence": "Mouse blood-to-brain transport of native APC; EPCR-dependent. No engineered peptide tested.",
        "safety": "APC/protein C competition; anticoagulant and PAR1-linked cytoprotective signalling.",
        "constraint": "Noncompeting, glycan-aware epitope; soluble EPCR; human stroke luminal access.",
    },
    "TM4SF1": {
        "evidence": "Endothelial surface antibody uptake; polarized BBB peptide transcytosis untested.",
        "safety": "Angiogenesis and vessel repair; species-dependent glycosylated EL2 epitope.",
        "constraint": "Human stroke BBB surface access, safe epitope and intact basolateral cargo recovery.",
    },
    "SELP": {
        "evidence": "CAV1-dependent fucoidan nanocarrier transport in a brain-tumour model; not a free peptide.",
        "safety": "Platelet binding and thromboinflammation; extremely low healthy human brain EC RNA.",
        "constraint": "Patient-stroke endothelial luminal protein and nonperturbing peptide transport.",
    },
    "ECSCR": {
        "evidence": "VEGFR2-associated endosomal redistribution; no ECSCR-bound cargo BBB transport.",
        "safety": "VEGF/PI3K/eNOS signalling and systemic metabolic effects; low ECD conservation.",
        "constraint": "Native luminal epitope and ECSCR-specific internalization/transport.",
    },
    "ACKR1": {
        "evidence": "Intact chemokines move abluminal-to-luminal; opposite required drug influx.",
        "safety": "Leukocyte recruitment, inflammation and genotype-dependent erythrocyte sink.",
        "constraint": "Direct blood-to-brain short-peptide transport, not shown.",
    },
    "LRP1": {
        "evidence": "Angiopep-2: partial LRP1-linked BBB influx. L57: cluster-IV binding and brain tracer uptake.",
        "safety": "Broad ligand clearance; BBB polarity and peripheral uptake; non-LRP1 routes possible.",
        "constraint": "Independent peptide-score comparator; no claim of clinical validation.",
    },
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required source: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        out = list(csv.DictReader(f))
    if not out:
        raise ValueError(f"Empty input: {path}")
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def numerical_equal(a: str, b: str) -> bool:
    if a.strip() == "" or b.strip() == "":
        return a.strip() == b.strip()
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def format_fdr(s: str) -> str:
    if not s:
        return "not available"
    x = float(s)
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"


def fmt(s: str, digits: int = 1) -> str:
    return f"{float(s):,.{digits}f}" if s.strip() else "not available"


def locate(name: str, root: Path) -> Path:
    candidates = [
        root / "data/processed/review_outputs" / name,
        root / "data/processed/review_inputs" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"{name} not found in review_outputs or review_inputs under {root}")


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def create_figure(rows: list[dict[str, str]], path_stem: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from textwrap import wrap

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",  # editable vector labels
        "pdf.fonttype": 42,    # preserve fonts rather than turning into paths
        "savefig.facecolor": "white",
    })
    widths = [2.05, 2.20, 2.00, 1.00, 3.55, 3.06, 2.50, 3.38]
    headers = ["Receptor / role", "Mouse stroke (D02)", "Human brain EC", "ECD identity", "Transport endpoint", "Safety / structural liability", "Peripheral EC RNA peak", "Reviewed primary DOIs"]
    total = sum(widths)
    mx = 0.43
    header_bottom = 7.36
    row_h = 1.08
    top = header_bottom + 0.62
    bottom = header_bottom - row_h * len(rows)
    fig = plt.figure(figsize=(total + 2*mx, 9.55), facecolor="white")
    ax = fig.add_axes([0.025, 0.018, .95, .965]); ax.set_xlim(0, total + 2*mx); ax.set_ylim(0, 9.55); ax.axis("off")
    navy = "#1F4E79"; ink = "#243649"; soft = "#5F6F7D"
    ax.text(mx, 9.14, "STROKE ENDOTHELIAL RECEPTOR PRIORITIZATION", fontsize=16.0, fontweight="bold", color=navy, va="center")
    ax.text(mx, 8.72, "Five exploratory discovery candidates and one independent peptide-shuttle benchmark", fontsize=10.3, color=soft, va="center")
    ax.text(mx, 8.37, "Decisions are manually adjudicated; the quantitative audit is a sensitivity analysis, not a trained selection model.", fontsize=8.8, color=soft, va="center")
    ax.add_patch(Rectangle((mx, header_bottom), total, .62, color=navy, linewidth=0))
    cur = mx
    for i,(w,h) in enumerate(zip(widths,headers)):
        ax.text(cur + .10, top - .32, h, fontsize=8.6 if i!=7 else 8.1, color="white", fontweight="bold", va="center")
        cur += w
    colx = [mx]
    for w in widths: colx.append(colx[-1]+w)

    def lines(s: str, maxlen: int, max_lines: int = 6) -> str:
        # Validation below prevents silent content truncation in publication outputs.
        pieces = wrap(str(s), width=maxlen, break_long_words=False, break_on_hyphens=False)
        if len(pieces)>max_lines:
            raise ValueError("Figure cell would overflow: "+repr(s))
        return "\n".join(pieces)

    for idx,row in enumerate(rows):
        ytop = header_bottom - idx*row_h
        ybottom = ytop-row_h
        base = "#FFFFFF" if idx%2 == 0 else "#F4F7FA"
        ax.add_patch(Rectangle((mx,ybottom),total,row_h,facecolor=base,edgecolor="#DEE4EA",lw=.36))
        status,accent=STATUS[row["status_code"]]
        ax.add_patch(Rectangle((mx,ybottom),.065,row_h,color=accent,lw=0))
        for xx in colx[1:-1]: ax.plot([xx,xx],[ybottom,ytop],lw=.28,color="#DAE2E9")
        targetx = colx[0]+.14
        ax.text(targetx, ytop-.245, row["receptor"], color=navy,fontweight="bold",fontsize=12.2,va="top")
        ax.text(targetx, ytop-.64, status, color=accent,fontweight="bold",fontsize=7.5,va="top")
        x=colx[1]+.10
        if row["D02_logFC"]:
            stroke=f"log2FC {float(row['D02_logFC']):+.2f}\nFDR {format_fdr(row['D02_FDR'])}\nLOSO {fmt(row['LOSO_pct'],0)}%"
        else: stroke="D02 unavailable\nBenchmark (not ranked)"
        ax.text(x,ytop-.155,stroke,color=ink,fontsize=8.9,va="top",linespacing=1.5)
        hum = f"{fmt(row['human_brain_EC_CPM'])} CPM\nEC enrichment {float(row['human_EC_nonEC_log2']):+.2f} log2"
        ax.text(colx[2]+.10,ytop-.20,hum,color=ink,fontsize=8.7,va="top",linespacing=1.65)
        ax.text(colx[3]+.10,ytop-.20,f"{fmt(row['ECD_identity_pct'])}%",color=ink,fontsize=9.3,va="top")
        ax.text(colx[4]+.10,ytop-.19,lines(row["evidence_short"],41,5),color=ink,fontsize=8.55,va="top",linespacing=1.36)
        ax.text(colx[5]+.10,ytop-.19,lines(row["safety_short"],34,5),color=ink,fontsize=8.4,va="top",linespacing=1.36)
        ax.text(colx[6]+.10,ytop-.19,lines(row["peripheral_short"],28,4),color=ink,fontsize=8.4,va="top",linespacing=1.5)
        doi_lines = row["reviewed_dois"].split("; ")
        ax.text(colx[7]+.10,ytop-.17,"\n".join(doi_lines),color="#2A6091",fontsize=7.75,va="top",linespacing=1.38)

    foot = bottom-.20
    ax.plot([mx,mx+total],[foot,foot],lw=.8,color="#B8C5D1")
    notes=[
        "Provisional design targets only: PROCR and TM4SF1. SELP, ECSCR and ACKR1 remain documented alternatives; LRP1 is a separate benchmark.",
        "ECD = extracellular domain; EC = endothelial cell; CPM = counts per million; LOSO = leave-one-sample-out direction stability.",
        "Human brain atlas CPM and peripheral HPA nTPM use different datasets/scales and must NOT be interpreted as a selectivity ratio.",
        "Published binding or uptake does not prove intact engineered peptide delivery; native luminal access, safety and epitope approval remain open.",
    ]
    for j,n in enumerate(notes):ax.text(mx,foot-.20-j*.205,n,fontsize=8.0,color=soft,va="top")
    for ext,kwargs in [(".pdf",{}),(".svg",{}),(".png",{"dpi":210})]:
        fig.savefig(path_stem.with_suffix(ext),bbox_inches="tight",pad_inches=.16,**kwargs)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=None,help="OmiPep project root (otherwise inferred from script in notebooks/)")
    p.add_argument("--out",type=Path,default=None,help="Override CSV output folder")
    p.add_argument("--figure-dir",type=Path,default=None,help="Override figure output folder")
    a=p.parse_args()
    root = (a.root or Path(os.environ.get("OMIPEP_ROOT", Path(__file__).resolve().parent.parent))).resolve()
    paths={key:locate(value,root) for key,value in INPUTS.items()}
    hashes_before={key:sha256(value) for key,value in paths.items()}
    selection=load_csv(paths["selection"])
    papers=load_csv(paths["paper_ledger"])
    master=load_csv(paths["master"])
    audit=load_csv(paths["gate_audit"])
    if len(selection)!=6 or {r["receptor"] for r in selection}!=set(ORDER):
        raise AssertionError("Expected exactly six distinct receptors from Cell 12.4l")
    if len(master)!=125 or len({r["relationship_id"] for r in master})!=125 or len({r["mouse_gene"] for r in master})!=114:
        raise AssertionError("Source 125-relationship/114-mouse-gene universe changed")
    if len(audit)!=125 or {r["relationship_id"] for r in audit}!={r["relationship_id"] for r in master}:
        raise AssertionError("Cell 12.4l audit is not aligned with master relationship IDs")
    sig_up=sum(bool(r["mouse_D02_logFC"] and r["mouse_D02_FDR"] and float(r["mouse_D02_logFC"])>0 and float(r["mouse_D02_FDR"])<.05) for r in master)
    if sig_up!=94: raise AssertionError(f"Expected 94 D02 significant upregulated relationships, found {sig_up}")
    if len(papers)!=19 or len({r["citation_key"] for r in papers})!=19:
        raise AssertionError("Expected 19 unique full-text paper records")
    keymap={r["citation_key"]:r for r in papers}
    by_symbol={r["human_gene"].upper():r for r in master}
    by_sel={r["receptor"]:r for r in selection}
    rows=[]; refrows=[]
    for gene in ORDER:
        s=by_sel[gene]; m=by_symbol[gene]
        if s["final_provisional_status"] not in STATUS: raise AssertionError(f"Unexpected selection status: {gene}")
        if gene=="LRP1":
            if s["D02_logFC"]!="" or s["D02_FDR"]!="": raise AssertionError("Benchmark must not be presented as D02 significant up")
        else:
            if s["final_provisional_status"]=="benchmark_only":raise AssertionError("Discovery target mislabelled benchmark")
        for k,master_key in [("D02_logFC","mouse_D02_logFC"),("D02_FDR","mouse_D02_FDR"),("LOSO_pct","mouse_D02_LOSO_direction_stability_pct"),("human_brain_EC_CPM","human_brain_EC_median_CPM_atlas"),("human_EC_nonEC_log2","human_EC_nonEC_log2_ratio_atlas"),("ECD_identity_pct","mouse_human_extracellular_identity_pct")]:
            if not numerical_equal(s[k],m[master_key]):
                raise AssertionError(f"Numerical source mismatch {gene}: {k} vs {master_key}")
        keys=[x.strip() for x in s["literature_references"].split(";") if x.strip()]
        if len(keys)<3 or len(set(keys))!=len(keys):raise AssertionError(f"Bad citation keys for {gene}")
        if any(k not in keymap for k in keys):raise AssertionError(f"Missing reviewed full text for {gene}")
        doilist=[]
        for key in keys:
            pp=keymap[key]
            if gene not in pp["receptor"].upper() and gene not in pp["relationship_id"].upper():
                raise AssertionError(f"Paper source mismatch {gene}: {key}")
            doi=pp["doi"].strip().lower()
            if not doi.startswith("10."):raise AssertionError(f"Missing invalid DOI: {key}")
            doilist.append(doi)
            refrows.append({"receptor":gene,"role":s["role"],"citation_key":key,"year":pp["year"],"doi":doi,"doi_url":"https://doi.org/"+doi,"evidence_grade":pp["evidence_grade"],"review_status":pp["review_status"]})
        bulkval=m["HPA_peripheral_bulk_RNA_max_nTPM"]
        ecval=m["HPA_peripheral_EC_RNA_max_nTPM"]
        ecsite=m["HPA_peripheral_EC_peak_organ_cluster"].split("|")[0].strip()
        short=CURATED[gene]
        row={
            "receptor":gene,
            "role":s["role"],"status_code":s["final_provisional_status"],
            "D02_logFC":s["D02_logFC"],"D02_FDR":s["D02_FDR"],"LOSO_pct":s["LOSO_pct"],
            "human_brain_EC_CPM":s["human_brain_EC_CPM"],"human_EC_nonEC_log2":s["human_EC_nonEC_log2"],
            "ECD_identity_pct":s["ECD_identity_pct"],
            "evidence_short":short["evidence"],"safety_short":short["safety"],
            "peripheral_short":f"{ecsite} EC: {fmt(ecval)} nTPM (HPA)",
            "structural_and_transport_gate":short["constraint"],
            "peripheral_bulk_peak_tissue":m["HPA_peripheral_bulk_peak_tissue"],
            "peripheral_bulk_max_nTPM":bulkval,
            "peripheral_EC_peak_organ_cluster":m["HPA_peripheral_EC_peak_organ_cluster"],
            "peripheral_EC_max_nTPM":ecval,
            "evidence_source_relationship_id":m["relationship_id"],
            "reviewed_citation_keys":"; ".join(keys),
            "reviewed_dois":"; ".join(doilist),
            "paper_count":str(len(keys)),
            "approval_state":"NOT_EXPERT_OR_STRUCTURALLY_APPROVED",
        }
        rows.append(row)
    if len(refrows)!=19:raise AssertionError("Expected 19 receptor-linked citations")
    out=(a.out or root/"data/processed/review_outputs").resolve()
    figdir=(a.figure_dir or root/"results/figures/module_01").resolve()
    out.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    maincsv=out/"EC_receptor_M01_12_4m_six_receptor_publication_table_v1.csv"
    refcsv=out/"EC_receptor_M01_12_4m_reviewed_DOI_map_v1.csv"
    figstem=figdir/"M01_receptor_selection_table_12_4m_v1"
    write_csv(maincsv,rows,list(rows[0]))
    write_csv(refcsv,refrows,list(refrows[0]))
    create_figure(rows,figstem)
    caption=(
       "OmiPep Module 01 | Standalone candidate receptor selection table (provisional).\n\n"
       "From 125 distinct mouse-human relationships representing 114 mouse genes, 94 met the D02 "
       "upregulation flag (log2FC > 0; FDR < 0.05). Five receptors were explored by literature-informed "
       "manual adjudication after quantitative and topological screening. The independent LRP1 benchmark "
       "was not included among D02 significant-upregulated discovery targets. Columns describe mouse stroke "
       "DE, independent human brain endothelial RNA, mouse-human extracellular-domain conservation, "
       "transport endpoint, noncompensatory biological liabilities and peripheral HPA context. The latter "
       "is shown in nTPM and must not be divided by brain-atlas CPM. 19 distinct uploaded primary papers "
       "were assessed by full-text figure-level review across the six targets, with DOI provenance in the table. "
       "The initial five-candidate selection was exploratory, and rank-sensitivity analysis was retrospective; "
       "it does not demonstrate that these five uniquely emerged from a pretrained or preregistered scoring rule. "
       "PROCR and TM4SF1 are provisional computational design arms, SELP/ECSCR/ACKR1 are reserves, and "
       "LRP1 is a published-peptide comparator. Native stroke-human luminal epitopes, safe nonblocking "
       "binding, intact peptide-cargo delivery and expert approval remain unverified.\n"
    )
    (out/"EC_receptor_M01_12_4m_table_caption_v1.txt").write_text(caption,encoding="utf-8")
    assert {k:sha256(v) for k,v in paths.items()}==hashes_before,"Source bytes changed"
    auditinfo={
        "cell":"12.4m","scope":"publication table generated from prior completed review; NO receptor re-ranking",
        "input_sha256":hashes_before,"source_files":{k:str(v.relative_to(root)) if v.is_relative_to(root) else str(v) for k,v in paths.items()},
        "source_counts":{"relationships":len(master),"distinct_mouse_genes":114,"D02_significantly_upregulated":sig_up,"paper_fulltexts":len(papers),"manuscript_rows":len(rows)},
        "source_selection":"human-led manual post-hoc adjudication, with LLM-assisted evidence organization; see Cell 12.4l method and limitations",
        "selection_status":{r["receptor"]:r["status_code"] for r in rows},
        "output_sha256":{"publication_table":sha256(maincsv),"reviewed_doi_map":sha256(refcsv),"figure_pdf":sha256(figstem.with_suffix(".pdf")),"figure_svg":sha256(figstem.with_suffix(".svg"))},
        "open_gates":["expert approval","native human ischemic BBB luminal target protein","membrane-context epitope qualification","intact engineered peptide-cargo luminal-to-abluminal recovery","functional and systemic safety"],
    }
    (out/"EC_receptor_M01_12_4m_validation_v1.json").write_text(json.dumps(auditinfo,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"PASS | 125 relationships, 114 genes, 94 D02-up; 6 table rows; 19 DOI-linked primary papers; source SHA-256 preserved")
    print(f"CSV: {maincsv}\nReferences: {refcsv}\nFigure: {figstem}.pdf/.svg/.png")
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            from IPython.display import display, Image
            display(Image(filename=str(figstem.with_suffix('.png')),width=1300))
    except ImportError:
        pass


if __name__=="__main__":
    main()
