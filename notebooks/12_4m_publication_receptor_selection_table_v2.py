#!/usr/bin/env python3
"""OmiPep M01 / Cell 12.4m v2: journal-style editable receptor table.

Inputs: frozen 12.4l/12.4m outputs and 19-study full-text ledger.
Outputs: black-and-white editable Word manuscript table, an internal working
version retaining DOIs, six-row source table, reference map and validation.
Also exports publication-layout SVG/PDF/PNG previews without embedding data as
an image in Word. Does not modify any input or historic v1 output.

Run: python notebooks/12_4m_publication_receptor_selection_table_v2.py
Notebook (cwd=notebooks): %run -i 12_4m_publication_receptor_selection_table_v2.py
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import math

INPUT_FILES = {
    'publication_v1': 'EC_receptor_M01_12_4m_six_receptor_publication_table_v1.csv',
    'selection': 'EC_receptor_M01_12_4l_six_receptor_selection_v1.csv',
    'gate_audit': 'EC_receptor_M01_12_4l_125_gate_audit_v1.csv',
    'ledger': 'EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv',
}
ORDER = ['PROCR','TM4SF1','SELP','ECSCR','ACKR1','LRP1']
ROLE = {
    'PROCR': 'Primary 1 (provisional)',
    'TM4SF1': 'Primary 2 (provisional)',
    'SELP': 'Conditional reserve',
    'ECSCR': 'Reserve',
    'ACKR1': 'Directional reserve',
    'LRP1': 'Independent benchmark',
}
# Reviewed, receptor-specific syntheses: each row maps to the 19-paper source ledger.
# These are *scientific interpretations*, not machine-generated safety labels.
SUMMARY = {
    'PROCR': {
        'access': 'Functional access at mouse BBB; human stroke luminal epitope unverified.',
        'transport': 'Native APC: blood-to-brain transport in mouse; no engineered peptide.',
        'liability': 'APC/protein C competition; coagulation and PAR1 signalling.',
        'gate': 'Noncompeting human luminal epitope; intact peptide transport and coagulation safety.'
    },
    'TM4SF1': {
        'access': 'Human peripheral-vessel surface binding; glycan-sensitive EL2 epitope.',
        'transport': 'Antibody internalization in human EC; BBB transcytosis not demonstrated.',
        'liability': 'Vascular repair and angiogenesis; epitope- and payload-dependent risks.',
        'gate': 'Native human brain luminal epitope; glycan-aware binder and intact peptide transport.'
    },
    'SELP': {
        'access': 'Activated tumour/inflamed brain vessels; human stroke luminal level unknown.',
        'transport': 'Fucoidan nanoparticles: tumour-BBB delivery; no free-peptide evidence.',
        'liability': 'Platelet/leukocyte adhesion and inflammatory barrier leakage.',
        'gate': 'Human stroke luminal abundance; monovalent peptide transport and platelet safety.'
    },
    'ECSCR': {
        'access': 'Cultured endothelial surface; native human brain luminal status unknown.',
        'transport': 'VEGFR2/KDR-associated trafficking; no cargo BBB passage.',
        'liability': 'VEGF/KDR signalling, vascular repair and systemic metabolic effects.',
        'gate': 'Human brain luminal access; receptor-specific uptake and signalling-neutral binder.'
    },
    'ACKR1': {
        'access': 'Inflamed CNS venules; native human stroke luminal level unknown.',
        'transport': 'Native chemokines: brain-to-blood transport (opposite direction).',
        'liability': 'Chemokine-mediated leukocyte recruitment; erythrocyte sink.',
        'gate': 'Demonstrate blood-to-brain peptide delivery and acceptable erythroid/immune targeting.'
    },
    'LRP1': {
        'access': 'Human brain capillary protein detected; native luminal epitope unresolved.',
        'transport': 'Angiopep-2 BBB precedent; L57 brain tracer does not establish intact parenchymal cargo.',
        'liability': 'Broad ligand clearance, BBB polarity and peripheral ligand uptake.',
        'gate': 'Peptide-specific receptor dependence and intact abluminal cargo recovery.'
    },
}
SOURCE_PREFIX = 'EC_receptor_M01_12_4m'
FIG_STEM = 'M01_receptor_selection_table_12_4m_v2'
TITLE = 'Biological evidence and translational constraints of candidate receptors'
HEADERS = [
    'Receptor / role',
    'Luminal and structural context',
    'Transport precedent',
    'Functional liability',
    'Peripheral EC RNA peak',
    'Critical unresolved requirement',
]
DOC_KEYS = ['receptor_role','access_and_structure','transport_precedent','functional_liability', 'peripheral_EC_RNA_peak','critical_unresolved_requirement']


def read_csv(path: Path) -> list[dict[str,str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Required input absent: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AssertionError(f'Empty CSV {path}')
    return rows


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def locate(root:Path, filename:str)->Path:
    for directory in ('data/processed/review_outputs','data/processed/review_inputs'):
        p=root/directory/filename
        if p.is_file(): return p
    raise FileNotFoundError(f'{filename} not found under review_outputs or review_inputs')


def num_equal(a:str,b:str)->bool:
    if not str(a).strip() or not str(b).strip(): return not str(a).strip() and not str(b).strip()
    return math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10)


def write_csv(path:Path, rows:list[dict[str,object]], headers:list[str]) -> None:
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore');w.writeheader();w.writerows(rows)


def prepare(root:Path):
    ip={key:locate(root,name) for key,name in INPUT_FILES.items()}
    hashes={key:sha(p) for key,p in ip.items()}
    v1=read_csv(ip['publication_v1']);sel=read_csv(ip['selection']);audit=read_csv(ip['gate_audit']);papers=read_csv(ip['ledger'])
    assert len(v1)==6 and {r['receptor'] for r in v1}==set(ORDER),'Expected six 12.4m table rows'
    assert len(sel)==6 and {r['receptor'] for r in sel}==set(ORDER),'Expected six 12.4l selected rows'
    assert len(audit)==125 and len({r['relationship_id'] for r in audit})==125 and len({r['mouse_gene'] for r in audit})==114,'125 / 114 audit altered'
    assert len(papers)==19 and len({r['citation_key'] for r in papers})==19,'Expected 19 distinct fulltext papers'
    vb={r['receptor']:r for r in v1};sb={r['receptor']:r for r in sel};by_id={r['relationship_id']:r for r in audit};pb={r['citation_key']:r for r in papers}
    assert sum(r['screen_gate']=='D02_sig_up' for r in audit)==94
    count_status={r['final_provisional_status'] for r in sel}
    assert {'provisional_primary_1','provisional_primary_2','conditional_reserve_1','reserve_2','direction_mismatch_reserve','benchmark_only'}==count_status
    out=[];doi_records=[]
    for g in ORDER:
        v=vb[g];s=sb[g];a=by_id[v['evidence_source_relationship_id']]
        assert a['human_gene'].upper()==g, ('wrong relationship link',g)
        for left,right in [('D02_logFC','D02_logFC'),('D02_FDR','D02_FDR'),('human_brain_EC_CPM','human_brain_EC_CPM'),('ECD_identity_pct','ECD_identity_pct')]:
            assert num_equal(v[left],s[left]) and num_equal(v[left],a[right]),f'Cross-source mismatch: {g} {left}'
        assert v['status_code']==s['final_provisional_status']
        assert v['approval_state']=='NOT_EXPERT_OR_STRUCTURALLY_APPROVED'
        maxval=float(v['peripheral_EC_max_nTPM'])
        expected_organ=v['peripheral_EC_peak_organ_cluster'].split('|')[0].strip()
        assert maxval>=0 and expected_organ,'Invalid HPA data'
        keys=[x.strip() for x in v['reviewed_citation_keys'].split(';') if x.strip()]
        dois=[x.strip().lower() for x in v['reviewed_dois'].split(';') if x.strip()]
        assert len(keys)==len(dois)==int(v['paper_count']) and len(keys)>=3,(g,'citations')
        for k,d in zip(keys,dois):
            assert k in pb and pb[k]['doi'].lower()==d and d.startswith('10.'),('DOI/source error',g,k,d)
            doi_records.append({
                'receptor':g,'relationship_id':v['evidence_source_relationship_id'],
                'citation_key':k,'year':pb[k]['year'],'doi':d,'doi_url':'https://doi.org/'+d,
                'model':pb[k]['species_models'],'experimental_endpoint':pb[k]['main_endpoint'],
                'evidence_grade':pb[k]['evidence_grade'],'review_status':pb[k]['review_status']
            })
        access=f"{SUMMARY[g]['access']} ECD identity: {float(v['ECD_identity_pct']):.1f}%."
        r={
            'receptor':g,
            'role':ROLE[g],
            'receptor_role': f'{g}\n{ROLE[g]}',
            'access_and_structure':access,
            'transport_precedent':SUMMARY[g]['transport'],
            'functional_liability':SUMMARY[g]['liability'],
            'peripheral_EC_RNA_peak':f"{expected_organ.capitalize()}: {maxval:,.1f} nTPM",
            'critical_unresolved_requirement':SUMMARY[g]['gate'],
            'D02_logFC_source':v['D02_logFC'],
            'D02_FDR_source':v['D02_FDR'],
            'LOSO_pct_source':v['LOSO_pct'],
            'human_brain_EC_CPM_source':v['human_brain_EC_CPM'],
            'human_EC_nonEC_log2_source':v['human_EC_nonEC_log2'],
            'ECD_identity_pct_source':v['ECD_identity_pct'],
            'peripheral_EC_peak_organ_cluster_source':v['peripheral_EC_peak_organ_cluster'],
            'peripheral_EC_max_nTPM_source':v['peripheral_EC_max_nTPM'],
            'peripheral_bulk_peak_tissue_source':v['peripheral_bulk_peak_tissue'],
            'peripheral_bulk_max_nTPM_source':v['peripheral_bulk_max_nTPM'],
            'supporting_citation_keys':'; '.join(keys),
            'supporting_dois':'; '.join(dois),
            'evidence_source_relationship_id':v['evidence_source_relationship_id'],
            'approval_state':v['approval_state'],
            'selection_status_source':v['status_code'],
        }
        out.append(r)
    assert len(doi_records)==19
    assert {r['receptor'] for r in out[:2]}=={'PROCR','TM4SF1'}
    assert out[-1]['receptor']=='LRP1'
    return out,doi_records,ip,hashes


def set_cell_border(cell, bottom=True, top=False, color='C8CED4', sz=5):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tc_pr=cell._tc.get_or_add_tcPr()
    borders=tc_pr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders=OxmlElement('w:tcBorders');tc_pr.append(borders)
    for edge,use in [('top',top),('bottom',bottom)]:
        name='w:'+edge
        el=borders.find(qn(name))
        if el is None:
            el=OxmlElement(name);borders.append(el)
        el.set(qn('w:val'),'single' if use else 'nil')
        if use:
            el.set(qn('w:sz'),str(sz));el.set(qn('w:color'),color)


def cell_margin(cell, top=90, start=90, bottom=90, end=80):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tcPr=cell._tc.get_or_add_tcPr()
    tcMar=tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar=OxmlElement('w:tcMar');tcPr.append(tcMar)
    for m,v in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        e=tcMar.find(qn('w:'+m))
        if e is None:e=OxmlElement('w:'+m);tcMar.append(e)
        e.set(qn('w:w'),str(v));e.set(qn('w:type'),'dxa')


def write_docx(rows:list[dict[str,str]], outpath:Path, working:bool=False):
    try:
        from docx import Document
        from docx.shared import Cm, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
        from docx.enum.section import WD_ORIENTATION
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except ImportError as exc:
        raise SystemExit('python-docx is required: python -m pip install python-docx') from exc
    d=Document(); sec=d.sections[0]
    sec.orientation=WD_ORIENTATION.LANDSCAPE
    if working:
        sec.page_height=Cm(29.7);sec.page_width=Cm(42.0) # A3 landscape, internal DOI-rich copy
        sec.left_margin=sec.right_margin=Cm(1.35)
        widths=[2.8,6.05,6.25,5.55,3.5,7.0,7.2]
    else:
        sec.page_height=Cm(21.0);sec.page_width=Cm(29.7) # A4 landscape, journal-facing
        sec.left_margin=sec.right_margin=Cm(1.15)
        widths=[2.8,5.05,5.45,4.6,3.65,5.85]
    sec.top_margin=Cm(1.05);sec.bottom_margin=Cm(1.0)
    sec.header_distance=Cm(.4);sec.footer_distance=Cm(.45)
    styles=d.styles
    normal=styles['Normal']; normal.font.name='Arial';normal.font.size=Pt(9 if not working else 9.2)
    normal.font.color.rgb=RGBColor(30,30,30)
    normal.paragraph_format.space_after=Pt(0)
    normal.paragraph_format.space_before=Pt(0)
    normal.paragraph_format.line_spacing=1.06
    title=d.add_paragraph()
    title.paragraph_format.space_after=Pt(9)
    t=title.add_run('Table 1. '+TITLE+('.' if working else ''))
    t.font.name='Arial';t.bold=True;t.font.size=Pt(10.8 if not working else 11)
    tb=d.add_table(rows=1, cols=len(widths))
    tb.autofit=False
    tb.alignment=WD_TABLE_ALIGNMENT.CENTER
    headers=HEADERS+(['Supporting DOIs (working copy)'] if working else [])
    for i,(h,w) in enumerate(zip(headers,widths)):
        c=tb.rows[0].cells[i];c.width=Cm(w);cell_margin(c,70,60,110,60)
        p=c.paragraphs[0];p.paragraph_format.space_after=Pt(0)
        run=p.add_run(h);run.font.name='Arial';run.font.size=Pt(8.5 if not working else 8.7);run.bold=True
        set_cell_border(c,top=True,bottom=True,color='424242',sz=8)
    # header marked repeating
    trPr=tb.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');trPr.append(repeat)
    for ri,r in enumerate(rows):
        cells=tb.add_row().cells
        vals=[r[k] for k in DOC_KEYS]
        if working:vals.append('\n'.join(x.strip() for x in r['supporting_dois'].split(';') if x.strip()))
        for i,(c,value,w) in enumerate(zip(cells,vals,widths)):
            c.width=Cm(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;cell_margin(c)
            p=c.paragraphs[0]
            p.paragraph_format.keep_together=True;p.paragraph_format.space_after=Pt(0)
            run=p.add_run(value);run.font.name='Arial';run.font.size=Pt(8.65 if not working else (8 if i==6 else 8.7))
            set_cell_border(c,bottom=True,top=(ri==5),color='A8B0B6' if ri==5 else 'D0D5D8',sz=6 if ri==5 else 4)
        rowprop=tb.rows[-1]._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');rowprop.append(cant)
    cap=d.add_paragraph()
    cap.paragraph_format.space_before=Pt(9)
    cap.paragraph_format.line_spacing=1.02
    notes=[
        'ECD, extracellular domain; EC, endothelial cell; BBB, blood-brain barrier; APC, activated protein C; nTPM, normalized transcripts per million.',
        'Roles are provisional, evidence-informed post-hoc choices; LRP1 is a benchmark outside the stroke-based discovery comparison.',
        'Human brain EC CPM and peripheral HPA nTPM are derived from different datasets and are not a quantitative selectivity ratio.',
        'Accessibility, uptake and ligand transport do not establish intact engineered-peptide delivery or binder safety; no receptor or fragment is approved.',
    ]
    if working: notes.append('Internal working copy: supporting DOIs are retained for reference verification and may be replaced with numbered references in the manuscript.')
    for j,note in enumerate(notes):
        p=d.add_paragraph()
        p.paragraph_format.space_after=Pt(2)
        run=p.add_run(note);run.font.name='Arial';run.font.size=Pt(8.1 if not working else 8.5)
    d.save(outpath)


def render_preview(rows:list[dict[str,str]], outstem:Path):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    from textwrap import wrap
    # The SVG is a proof/rendering convenience; editable Word is the submission table.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((x for x in ('Arial','Liberation Sans','DejaVu Sans') if x in installed),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.unicode_minus':False,'savefig.facecolor':'white'})
    widths=[1.55,3.02,3.22,2.74,2.16,3.5]
    total=sum(widths); gap=.18
    height=6.95
    fig=plt.figure(figsize=(total+2*gap,height),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,total+2*gap);ax.set_ylim(0,height);ax.axis('off')
    ink='#202020';muted='#555555'
    ax.text(gap, height-.42, 'Table 1  '+TITLE,fontsize=12.0,color=ink,weight='bold',va='center')
    cols=[gap];
    for w in widths: cols.append(cols[-1]+w)
    yheader=height-1.06
    ax.plot([gap,cols[-1]],[yheader+.31,yheader+.31],color=ink,lw=.9)
    ax.plot([gap,cols[-1]],[yheader-.33,yheader-.33],color=ink,lw=.65)
    for i,h in enumerate(HEADERS):
        ax.text(cols[i]+.07,yheader+0.12,'\n'.join(wrap(h,max(14,int(widths[i]*8.5)),break_long_words=False)),
                fontsize=9.0,weight='bold',color=ink,va='center')
    row_h=.73
    for i,r in enumerate(rows):
        top=yheader-.33-i*row_h
        if i==5: ax.plot([gap,cols[-1]],[top,top],lw=.8,color='#808080')
        vals=[r[k] for k in DOC_KEYS]
        for j,(val,w) in enumerate(zip(vals,widths)):
            textval=val
            # small text strings to stay in regular table columns
            maxlen=max(15,int(w*10))
            ls=[]
            for line in textval.split('\n'):
                ls.extend(wrap(line,maxlen,break_long_words=False,break_on_hyphens=False))
            if len(ls)>5:
                raise ValueError(f'Preview text overflow: {r["receptor"]}, {HEADERS[j]}: {ls}')
            ax.text(cols[j]+.07, top-.13,'\n'.join(ls),fontsize=8.6 if j!=0 else 8.75,
                    color=ink,va='top',linespacing=1.33)
        ax.plot([gap,cols[-1]],[top-row_h,top-row_h],color='#D0D0D0',lw=.38)
    foot_y=yheader-.33-len(rows)*row_h-.22
    notes=[
        'Roles are provisional post-hoc; LRP1 is an independent benchmark. ECD: extracellular domain; EC: endothelial cell.',
        'Human brain atlas CPM and HPA nTPM are not directly comparable. No binder or intact engineered-peptide BBB shuttle is approved.',
    ]
    for k,n in enumerate(notes): ax.text(gap,foot_y-k*.23,n,fontsize=8.1,color=muted,va='top')
    for ext,kwargs in [('.svg',{}),('.pdf',{}),('.png',{'dpi':300})]:
        fig.savefig(outstem.with_suffix(ext),bbox_inches='tight',pad_inches=.08,**kwargs)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=None)
    parser.add_argument('--out-dir',type=Path,default=None)
    parser.add_argument('--figure-dir',type=Path,default=None)
    args=parser.parse_args()
    root=(args.root or Path(os.environ.get('OMIPEP_ROOT',Path(__file__).resolve().parent.parent))).resolve()
    outdir=(args.out_dir or root/'data/processed/review_outputs').resolve()
    figdir=(args.figure_dir or root/'results/figures/module_01').resolve()
    outdir.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    rows,refs,inputs,hashes=prepare(root)
    source=outdir/(SOURCE_PREFIX+'_six_receptor_publication_table_v2.csv')
    write_csv(source,rows,list(rows[0]))
    refp=outdir/(SOURCE_PREFIX+'_DOI_evidence_map_v2.csv')
    write_csv(refp,refs,list(refs[0]))
    caption=outdir/(SOURCE_PREFIX+'_table_caption_v2.txt')
    caption.write_text('Table 1. '+TITLE+'.\n\n'+
        'The table summarizes source-bound luminal/structural evidence, cargo-specific transport precedents, functional liabilities, and peak HPA peripheral endothelial RNA expression for five provisional discovery receptors plus the independent LRP1 benchmark. Roles derive from a post-hoc interpretation of the frozen 125-relationship evidence audit and 19 decision-critical full-text studies, not a prospective automated selection model. ECD identity refers to compiled mouse-human extracellular sequence identity. Human brain EC CPM and peripheral HPA endothelial nTPM come from distinct datasets, and neither demonstrates luminal protein accessibility or transport. No receptor has expert/structural approval and no intact engineered-peptide BBB delivery is established. Reviewed DOIs and per-paper model limitations are retained in the accompanying source-linked DOI map.\n',encoding='utf-8')
    docx_main=figdir/(FIG_STEM+'.docx')
    docx_work=figdir/(FIG_STEM+'_working_DOIs.docx')
    write_docx(rows,docx_main,working=False)
    write_docx(rows,docx_work,working=True)
    render_preview(rows,figdir/FIG_STEM)
    hashes_after={k:sha(p) for k,p in inputs.items()}
    assert hashes_after==hashes, 'Source files modified'
    val={
      'version':'Cell 12.4m v2','relationship_universe':125,'mouse_genes':114,
      'six_receptor_order':ORDER,'provisional_primary':['PROCR','TM4SF1'],'independent_benchmark':'LRP1',
      'fulltext_ledger_publications':19,'receptor_evidence_links':len(refs),
      'DOIs_retained_in_source_and_working_document':True,
      'quantitative_stroke_and_human_EC_values_retained_in_source_not_table':True,
      'approval': 'NONE',
      'source_sha256_before':hashes,'source_sha256_after':hashes_after,
      'source_paths':{k:str(p) for k,p in inputs.items()},
      'exports':{'manuscript_docx':str(docx_main),'working_DOI_docx':str(docx_work),'source_csv':str(source),'evidence_map':str(refp),'caption':str(caption),'preview':str(figdir/FIG_STEM)},
    }
    valp=outdir/(SOURCE_PREFIX+'_validation_v2.json')
    valp.write_text(json.dumps(val,indent=2),encoding='utf-8')
    print('PASS | Cell 12.4m v2: 125 relationships / 114 mouse genes; 6 receptor rows; 19 source-linked DOIs')
    print('PASS | Source hashes preserved; no receptor/peptide approved')
    print('DOCX |',docx_main)
    print('WORK |',docx_work)
    print('FIG  |',str(figdir/(FIG_STEM+'.{svg,pdf,png}')))
    print('DATA |',source)
    print('META |',refp,caption,valp)

if __name__ == '__main__':
    main()
