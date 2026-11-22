#!/usr/bin/env python3
"""OmiPep Module 01 Cell 14.2 | Mechanistic evidence and translational liabilities.

This is a source-linked, manually adjudicated categorical figure, NOT a safety
score, a transport-performance ranking, or evidence of peptide clinical safety.

Inputs: Cell 12.4m publication table, Cell 12.4l six-receptor selection and
125-relationship audit, Cell 12.4k 19-paper full-text ledger, and an explicit
42-cell manually reviewed decision table included in this installation bundle.
All input files remain untouched. Run via notebook ``%run -i ...`` or CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg") if __name__ == "__main__" and "ipykernel" not in __import__("sys").modules else None
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

ORDER = ("PROCR", "TM4SF1", "SELP", "ECSCR", "ACKR1", "LRP1")
DIMS = ("access", "uptake", "brainward", "peptide", "function", "barrier", "epitope")
EVIDENCE = DIMS[:4]
LIABILITY = DIMS[4:]
FNAME = "EC_receptor_M01_14_2_endpoint_risk_adjudication_v1.csv"
NAME = "M01_receptor_evidence_safety_matrix_14_2_v1"
DATA_NAME = "EC_receptor_M01_14_2_evidence_safety_matrix_source_v1.csv"
CAPTION_NAME = "EC_receptor_M01_14_2_evidence_safety_matrix_caption_v1.txt"
VALIDATION_NAME = "EC_receptor_M01_14_2_evidence_safety_matrix_validation_v1.json"

INPUTS = {
 "table": "EC_receptor_M01_12_4m_six_receptor_publication_table_v1.csv",
 "selected": "EC_receptor_M01_12_4l_six_receptor_selection_v1.csv",
 "audit": "EC_receptor_M01_12_4l_125_gate_audit_v1.csv",
 "ledger": "EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv",
 "adjudication": FNAME,
}
COL = {
 "navy": "#1F4E79", "blue": "#4E84C4", "teal": "#4FAE9D", "gold": "#C9A441",
 "coral": "#E58E73", "rose": "#F1C7D4", "ink": "#233747", "muted": "#687988",
 "slate": "#A9B6C2", "faint": "#E3E9ED", "light": "#F4F7F9", "white": "#FFFFFF",
}
FILL = {
 "observed": ("#E4F3F0", COL["teal"]),
 "contextual": ("#FAF3DF", COL["gold"]),
 "reverse": ("#FBEDE7", COL["coral"]),
 "unestablished": ("#F2F5F7", COL["slate"]),
 "potential": ("#FAF3DF", COL["gold"]),
 "unknown": ("#F2F5F7", COL["slate"]),
}
STATUSES = {
 "PROCR":"provisional_primary_1",
 "TM4SF1":"provisional_primary_2",
 "SELP":"conditional_reserve_1",
 "ECSCR":"reserve_2",
 "ACKR1":"direction_mismatch_reserve",
 "LRP1":"benchmark_only",
}
ROLES = {
 "PROCR":("PRIMARY 01",COL["teal"]),
 "TM4SF1":("PRIMARY 02",COL["teal"]),
 "SELP":("CONDITIONAL RESERVE",COL["gold"]),
 "ECSCR":("RESERVE",COL["gold"]),
 "ACKR1":("DIRECTIONAL RESERVE",COL["gold"]),
 "LRP1":("BENCHMARK ONLY",COL["blue"]),
}
DISPLAY_COLS = (
 ("access", "Blood-facing\naccess"),
 ("uptake", "Uptake /\ntrafficking"),
 ("brainward", "Blood → brain\ntransport"),
 ("peptide", "Intact peptide\ndelivery"),
 ("function", "Physiological\ninterference"),
 ("barrier", "BBB\nintegrity"),
 ("epitope", "Epitope /\nformat"),
)


def read_csv(p: Path) -> list[dict[str,str]]:
    if not p.is_file():
        raise FileNotFoundError(f"Required source file not found: {p}")
    with p.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or len(rd.fieldnames) != len(set(rd.fieldnames)):
            raise AssertionError(f"Missing / repeated headers in {p}")
        return list(rd)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def find_input(root:Path, name:str) -> Path:
    for sub in ("data/processed/review_outputs", "data/processed/review_inputs"):
        p = root/sub/name
        if p.is_file(): return p
    raise FileNotFoundError(f"Missing {name}. Install the Cell 14.2 bundle from the project root, and preserve 12.4k/l/m outputs.")


def file_hash(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def configure_font() -> tuple[str,str]:
    # Use existing user's Windows font installation via WSL when available;
    # NEVER package / redistribute font files.
    win=Path("/mnt/c/Windows/Fonts")
    for n in ("arial.ttf","arialbd.ttf","ariali.ttf","arialbi.ttf",
              "Arial.ttf","Arialbd.ttf"):
        p=win/n
        if p.is_file():
            try: fm.fontManager.addfont(str(p))
            except (ValueError,OSError,RuntimeError): pass
    for name in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            path=fm.findfont(fm.FontProperties(family=[name]),fallback_to_default=False)
            matplotlib.rcParams.update({
                "font.family":name,
                "font.sans-serif":["Arial", "Liberation Sans", "DejaVu Sans"],
                "font.weight":"normal", "svg.fonttype":"none", "pdf.fonttype":42,
                "savefig.facecolor":"white", "mathtext.fontset":"dejavusans",
            })
            return name,str(path)
        except ValueError: pass
    raise RuntimeError("No usable sans-serif font")


def css_svg_stack(path:Path)->int:
    text=path.read_text(encoding="utf-8")
    stack="Arial, Helvetica, 'Liberation Sans', 'DejaVu Sans', sans-serif"
    count=0
    def replace(m):
        nonlocal count
        count+=1
        tag=m.group(0)
        if 'style="' in tag:
            return re.sub(r'style="([^"]*)"',lambda t:'style="'+t.group(1).rstrip('; ')+'; font-family: '+stack+';"',tag,count=1)
        return tag[:-1]+' style="font-family: '+stack+';">'
    text=re.sub(r'<text\b[^>]*>',replace,text)
    path.write_text(text,encoding="utf-8")
    ET.parse(path)
    return count


def build_data(root:Path):
    paths={k:find_input(root,v) for k,v in INPUTS.items()}
    hashes_before={k:file_hash(v) for k,v in paths.items()}
    table=read_csv(paths['table']); selected=read_csv(paths['selected'])
    audit=read_csv(paths['audit']);ledger=read_csv(paths['ledger'])
    decision=read_csv(paths['adjudication'])
    assert len(table)==6 and {r['receptor'] for r in table}==set(ORDER),"Expected 6 Cell 12.4m publication rows"
    assert len(selected)==6 and {r['receptor'] for r in selected}==set(ORDER),"Expected 6 Cell 12.4l rows"
    assert len(audit)==125 and len({r['relationship_id'] for r in audit})==125 and len({r['mouse_gene'] for r in audit})==114,"125/114 audit failed"
    assert len(ledger)==19 and len({r['citation_key'] for r in ledger})==19,"Expected 19 unique full-text papers"
    assert len(decision)==42,"Expected 6 x 7 curated evidence cells"
    assert len({(r['receptor'],r['dimension']) for r in decision})==42,"Duplicated manually curated cell"
    assert {(r,d) for r in ORDER for d in DIMS}=={(x['receptor'],x['dimension']) for x in decision},"Incomplete matrix"
    tb={r['receptor']:r for r in table}; sel={r['receptor']:r for r in selected}
    paper_by_key={r['citation_key']:r for r in ledger}
    audit_id={r['relationship_id'] for r in audit}
    cells=[]
    counts=Counter()
    for receptor in ORDER:
        t=tb[receptor];s=sel[receptor]
        assert t['status_code']==STATUSES[receptor] and s['final_provisional_status']==STATUSES[receptor]
        assert t['approval_state']=='NOT_EXPERT_OR_STRUCTURALLY_APPROVED'
        assert t['evidence_source_relationship_id'] in audit_id
        for k in ('D02_logFC','D02_FDR','human_brain_EC_CPM','ECD_identity_pct'):
            assert t[k]==s[k],f"Unexpected 12.4l/m divergence {receptor}.{k}"
        if receptor=='LRP1': assert not t['D02_logFC'] and not t['D02_FDR']
        else: assert float(t['D02_FDR'])<.05 and float(t['D02_logFC'])>0
        x=float(t['peripheral_EC_max_nTPM']);assert math.isfinite(x) and x>=0
        for dim in DIMS:
            d=next(r for r in decision if r['receptor']==receptor and r['dimension']==dim)
            if dim in EVIDENCE:
                assert d['level'] in ('observed','contextual','reverse','unestablished')
            else:
                assert d['level'] in ('observed','potential','unknown')
            assert d['adjudication_status']=="MANUAL_SOURCE_LINKED_PROVISIONAL_NOT_EXPERT_APPROVED"
            assert d['visible_label'] and d['scientific_boundary']
            refs=[k.strip() for k in d['citation_keys'].split(';') if k.strip()]
            assert refs and len(set(refs))==len(refs),"Every cell requires nonduplicate ledger citations"
            assert all(k in paper_by_key for k in refs),f"Missing full-text source {receptor}.{dim}"
            assert all(receptor.upper() in paper_by_key[k]['receptor'].upper() for k in refs),f"Cross-receptor paper error {receptor}.{dim}"
            cites=[paper_by_key[k]['doi'].strip().lower() for k in refs]
            assert all(v.startswith('10.') for v in cites),f"Bad DOI {receptor}.{dim}"
            codes=dict(d)
            codes.update({
              'role':t['role'], 'status_code':t['status_code'],
              'reviewed_dois':'; '.join(cites),
              'HPA_peripheral_EC_max_nTPM':t['peripheral_EC_max_nTPM'],
              'HPA_peripheral_EC_peak_organ_cluster':t['peripheral_EC_peak_organ_cluster'],
              'HPA_peripheral_bulk_max_nTPM':t['peripheral_bulk_max_nTPM'],
              'human_brain_EC_CPM_independent_atlas':t['human_brain_EC_CPM'],
              'ECD_identity_pct_source':t['ECD_identity_pct'],
              'source_relationship_id':t['evidence_source_relationship_id'],
              'source_approval_state':t['approval_state'],
            })
            cells.append(codes)
            counts[f"{dim}:{d['level']}"]+=1
    return paths,hashes_before,tb,cells,counts


def short_lines(s:str)->str:
    # Labels are display strings from a manually curated, auditable source.
    # Fail instead of silently clipping to the figure panel.
    parts=textwrap.wrap(s,width=12,break_long_words=True,break_on_hyphens=True)
    if len(parts)>2: raise AssertionError(f"Curated display label too long: {s!r} -> {parts!r}")
    return '\n'.join(parts)


def round_tile(ax,x,y,w,h,level,dim):
    bg, accent=FILL[level]
    if dim in LIABILITY and level == "observed":
        bg, accent = "#FAECE6", COL["coral"]
    tile=FancyBboxPatch((x,y-h/2),w,h,boxstyle="round,pad=0.002,rounding_size=0.068",
                         edgecolor="#E4E9ED",facecolor=bg,lw=.52,zorder=2)
    ax.add_patch(tile)
    ax.add_patch(Rectangle((x+.025,y-h/2+.075),.032,h-.15,color=accent,lw=0,zorder=3))
    # Distinct glyph shapes with meaning explained in legend: no safety grades.
    cx=x+.207; cy=y+.007
    if level=="observed":
        color=COL['coral'] if dim in LIABILITY else COL['teal']
        ax.add_patch(Circle((cx,cy),.071,fc=color,ec='white',lw=.9,zorder=5))
        if dim in LIABILITY:
            ax.text(cx,cy-.004,'!',ha='center',va='center',color='white',fontsize=7.3,fontweight='bold',zorder=6)
    elif level in ('contextual','potential'):
        ax.add_patch(Circle((cx,cy),.070,fc='white',ec=COL['gold'],lw=1.8,zorder=5))
        ax.add_patch(Circle((cx,cy),.030,fc=COL['gold'],ec='none',zorder=6))
    elif level=='reverse':
        ax.add_patch(Polygon([[cx,cy+.092],[cx+.092,cy],[cx,cy-.092],[cx-.092,cy]],closed=True,
                             fc=COL['coral'],ec='white',lw=.6,zorder=5))
    else:
        ax.plot([cx-.066,cx+.066],[cy,cy],color=COL['slate'],lw=1.7,solid_capstyle='round',zorder=5)


def draw_chart(tables:dict,decisions:list,figstem:Path)->None:
    chosen={(r['receptor'],r['dimension']):r for r in decisions}
    W,H=13.65,7.70
    fig=plt.figure(figsize=(W,H),facecolor='white')
    ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(0,H));ax.set_axis_off()
    nav=COL['navy']; grey=COL['muted']
    left=0.32; start=2.07
    widths=(1.20,1.22,1.27,1.26,1.25,1.23,1.28)
    xstarts=[start]
    for w in widths: xstarts.append(xstarts[-1]+w)
    assert xstarts[-1]<10.9
    bar_x0=11.18; bar_x1=12.49; num_x=13.36
    # Header is intentionally editorially quiet: the figure is scientifically self-explanatory.
    ax.text(left,7.42,"ENDOTHELIAL RECEPTOR EVIDENCE",fontsize=10.7,fontweight='bold',color=nav,va='top')
    ax.text(13.35,7.42,"19 reviewed papers · conditional, post-hoc assessment",
            fontsize=8.0,color=grey,va='top',ha='right')
    groups=[(start,xstarts[4],"DELIVERY EVIDENCE",COL['teal']),
            (xstarts[4],xstarts[7],"TRANSLATIONAL LIABILITY",COL['coral']),
            (bar_x0,13.39,"PERIPHERAL EXPOSURE",COL['blue'])]
    for x0,x1,label,c in groups:
        ax.text(x0+.045,7.045,label,color=nav,fontsize=9.2,fontweight='bold',ha='left',va='center')
        ax.plot([x0+.04,x1-.045],[6.915,6.915],color=c,lw=2.1,solid_capstyle='round')
    for j,(_,label) in enumerate(DISPLAY_COLS):
        ax.text(xstarts[j]+widths[j]/2,6.70,label,color=nav,fontsize=8.5,
                fontweight='bold',ha='center',va='center',linespacing=1.17)
    ax.text(12.29,6.70,"Peak peripheral EC RNA\nHPA · nTPM (log axis)",
            color=nav,fontsize=8.4,fontweight='bold',ha='center',va='center',linespacing=1.17)
    ys=(5.98,5.18,4.34,3.54,2.74,1.66)
    # Fine separator between discovery group and separate benchmark arm.
    ax.plot([left,13.38],[2.205,2.205],color="#CCD6DE",lw=.8,linestyle=(0,(3,4)))
    # Subtle group separation.
    for xx in (xstarts[4]-.025,xstarts[7]+.05):
        ax.plot([xx,xx],[1.26,6.40],lw=.6,color='#E8EDF0')
    for i,rec in enumerate(ORDER):
        y=ys[i]; groupname,rolecol=ROLES[rec]
        ax.text(left+.08,y+.104,rec,fontweight='bold',fontsize=13.3,color=nav,va='center')
        ax.text(left+.08,y-.184,groupname,fontweight='bold',fontsize=7.45,color=rolecol,va='center')
        # Thin role accent avoids using role colors as categorical safety evidence.
        ax.plot([left,left],[y-.27,y+.27],lw=3.4,color=rolecol,solid_capstyle='round')
        for j,(dim,_) in enumerate(DISPLAY_COLS):
            dat=chosen[(rec,dim)];level=dat['level'];x=xstarts[j]+.052;w=widths[j]-.104
            round_tile(ax,x,y,w,.60,level,dim)
            label=short_lines(dat['visible_label'])
            ax.text(x+.342,y+.004,label,ha='left',va='center',color=COL['ink'],fontsize=8.2,
                    linespacing=1.12, zorder=8)
        # Quantified peripheral target RNA: *same* HPA EC dataset, no cross-atlas ratios.
        v=float(tables[rec]['peripheral_EC_max_nTPM'])
        organ=tables[rec]['peripheral_EC_peak_organ_cluster'].split('|')[0].strip()
        t=min(1.,math.log10(1+v)/4.0)  # log(1+nTPM) with displayed reference extent 10,000
        ax.plot([bar_x0,bar_x1],[y+.09,y+.09],color='#E6EBF0',lw=5.3,solid_capstyle='round')
        ax.plot([bar_x0,bar_x0+(bar_x1-bar_x0)*t],[y+.09,y+.09],
                color=COL['blue'],lw=4.8,solid_capstyle='round')
        ax.add_patch(Circle((bar_x0+(bar_x1-bar_x0)*t,y+.09),.060,fc=COL['navy'],ec='white',lw=.65,zorder=8))
        ax.text(num_x,y+.098,f'{v:,.1f}',fontsize=9.5,color=nav,ha='right',va='center',fontweight='bold')
        ax.text(bar_x0,y-.18,organ.capitalize(),fontsize=8.0,color=grey,va='center',ha='left')
    # Peripheral log-axis reference not a statistic: avoid suggesting brain specificity.
    for v,tick in [(0,'0'),(10,'10'),(100,'100'),(1000,'1k'),(10000,'10k')]:
        x=bar_x0+(bar_x1-bar_x0)*math.log10(1+v)/4.0
        ax.plot([x,x],[1.235,1.30],lw=.60,color='#BAC5CF')
        ax.text(x,1.13,tick,ha='center',va='top',fontsize=7.1,color=grey)
    # Two lean legend lines (evidence vs liability are deliberately NOT conflated).
    ly=.68
    def legend_dot(x,c,outline=False,diamond=False,dash=False,exclaim=False):
        if dash: ax.plot([x-.07,x+.07],[ly,ly],color=COL['slate'],lw=2)
        elif diamond:ax.add_patch(Polygon([[x,ly+.08],[x+.08,ly],[x,ly-.08],[x-.08,ly]],fc=c,ec='none',closed=True))
        else:
            ax.add_patch(Circle((x,ly),.074,fc='white' if outline else c,
                                ec=c if outline else 'none',lw=1.4))
            if outline:ax.add_patch(Circle((x,ly),.029,fc=c,ec='none'))
            if exclaim:ax.text(x,ly-.006,'!',color='white',fontsize=7.5,fontweight='bold',va='center',ha='center')
    specs=[(left+.10,COL['teal'],'Model-specific evidence',False,False,False,False),
           (3.15,COL['gold'],'Contextual / indirect',True,False,False,False),
           (5.86,COL['coral'],'Opposite-direction finding',False,True,False,False),
           (9.24,COL['slate'],'Not established / unassessed',False,False,True,False)]
    for x,c,lab,o,d,ds,e in specs:
        legend_dot(x,c,o,d,ds,e)
        ax.text(x+.19,ly,lab,fontsize=9.1,color=COL['ink'],va='center')
    ax.text(left,.30,'In liability columns: coral = documented pathway/constraint; gold = plausible concern; grey = binder-specific effect unassessed.',
            fontsize=8.2,color=grey,va='center')
    ax.text(13.35,.30,'Binder safety and intact engineered-peptide BBB delivery remain unverified.',
            fontsize=8.0,color=nav,fontweight='bold',ha='right',va='center')
    for ext,kw in (('svg',{}),('pdf',{}),('png',{'dpi':360})):
        fig.savefig(figstem.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.02,**kw)
    plt.close(fig)


def make_caption()->str:
    return ("Figure M01-14.2 | Mechanistic evidence and translational liabilities across five manually adjudicated provisional "
            "discovery receptors and an independent LRP1 benchmark. Cell annotations are model-/cargo-specific interpretations "
            "of the 19-paper full-text ledger: teal marks an endpoint demonstrated in the cited setting (not necessarily human "
            "ischemic BBB or short peptide), ochre denotes contextual/indirect evidence, coral diamond denotes the tested "
            "opposite-direction chemokine transport, and grey denotes a requested endpoint not established in the reviewed evidence. "
            "Liability columns separately encode pathway or structural concerns, not experimentally proven peptide toxicity; "
            "a grey entry is unassessed, never evidence of safety. A separate HPA peripheral endothelial maximum RNA track "
            "(log10[1+nTPM] display; labels are original nTPM) provides on-target exposure context across peripheral organs, "
            "not a measured peripheral drug sink or organ toxicity. Its nTPM values must not be divided by the independent human "
            "brain atlas CPM. PROCR mouse APC influx, SELP tumor-model fucoidan nanocarrier delivery and LRP1-linked Angiopep-2 "
            "model transport cannot be equated to intact engineered-peptide transport in healthy or ischemic human BBB. "
            "ACKR1 literature demonstrates native brain-to-blood chemokine export, not general impossibility of brainward "
            "transport for other ligands. No target has been expert-approved or structurally approved; no intact engineered-peptide "
            "BBB delivery or safety is demonstrated. Detailed per-cell citations, model limitations, and provenance appear in the "
            "adjacent source-data CSV. Biological evidence status is post-hoc and must not be presented as preregistered selection.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=None,help='Project root; default parent of notebooks script')
    parser.add_argument('--figure-dir',type=Path,default=None)
    parser.add_argument('--out-dir',type=Path,default=None)
    args=parser.parse_args(argv)
    root=(args.root if args.root else Path(__file__).resolve().parent.parent).resolve()
    paths,hashes,table,src,counts=build_data(root)
    out=args.out_dir.resolve() if args.out_dir else root/'data/processed/review_outputs'
    figdir=args.figure_dir.resolve() if args.figure_dir else root/'results/figures/module_01'
    out.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    fstack,fpath=configure_font()
    stem=figdir/NAME
    draw_chart(table,src,stem)
    nsvg=css_svg_stack(stem.with_suffix('.svg'))
    # Validate all three delivered exports and absence of changes to the input universe.
    assert all(stem.with_suffix('.'+ext).stat().st_size>5000 for ext in ('svg','pdf','png'))
    assert nsvg>=25,f'Vector SVG lost text nodes ({nsvg})'
    fields=['receptor','dimension','level','visible_label','scientific_boundary','citation_keys',
            'adjudication_status','role','status_code','reviewed_dois','HPA_peripheral_EC_max_nTPM',
            'HPA_peripheral_EC_peak_organ_cluster','HPA_peripheral_bulk_max_nTPM',
            'human_brain_EC_CPM_independent_atlas','ECD_identity_pct_source','source_relationship_id','source_approval_state']
    write_csv(out/DATA_NAME,src,fields)
    (out/CAPTION_NAME).write_text(make_caption()+'\n',encoding='utf-8')
    assert {k:file_hash(v) for k,v in paths.items()}==hashes,"Source files modified during rendering"
    validation={
      'module':'M01','cell':'14.2','output_version':'v1','relationship_universe':125,
      'mouse_genes':114,'reviewed_paper_count':19,'n_receptors':6,'n_cells':42,
      'provisional_primaries':['PROCR','TM4SF1'],
      'provisional_reserves':['SELP','ECSCR','ACKR1'],
      'benchmark':['LRP1'],
      'all_receptor_approvals_pending':True,
      'manual_cell_interpretation_not_machine_score':True,
      'level_counts':dict(counts),
      'font_family':fstack,'font_resolved_path':fpath,
      'SVG_editable_text_nodes':nsvg,
      'input_sha256':{str(v.relative_to(root)):hashes[k] for k,v in paths.items()},
      'output_sha256':{p.name:file_hash(p) for p in (
          stem.with_suffix('.svg'),stem.with_suffix('.pdf'),stem.with_suffix('.png'),out/DATA_NAME,out/CAPTION_NAME)},
      'safe_scope_statement':'Contextual receptor/pathway evidence only; does not show clinical or engineered-peptide safety',
    }
    (out/VALIDATION_NAME).write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PASS | Cell 14.2: 6 receptors; 42 source-linked cells; 19 full-text papers')
    print('PASS | 125 relationships / 114 mouse genes unchanged; all source hashes preserved')
    print('PASS | Evidence endpoints, model caveats and mechanistic liability separate; no approved peptide shuttle')
    print(f'FONT | {fstack} (editable SVG, sans-serif fallback; {nsvg} text nodes)')
    print(f'FIG  | {stem}.{{svg,pdf,png}}')
    print(f'DATA | {out/DATA_NAME}')
    print(f'META | {out/VALIDATION_NAME}; {out/CAPTION_NAME}')

if __name__=='__main__':
    main()
