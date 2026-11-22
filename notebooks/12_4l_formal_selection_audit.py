#!/usr/bin/env python3
"""OmiPep v0.2 Module 01 Cell 12.4l: conservative, reproducible, post-hoc discovery audit.

Inputs are untouched. All 125 relationship rows retained. Preliminary mechanistic
ratings for five targets are literature-adjudication labels, never auto-verified.
Run via `%run -i 12_4l_formal_selection_audit.py` or `python notebooks/12_4l_formal_selection_audit.py`.
"""
from __future__ import annotations
import csv, hashlib, json, math, argparse, sys
from collections import Counter
from pathlib import Path
from datetime import date

MASTER='EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv'
SIX='EC_receptor_M01_12_4k_shortlist_WORKING_v8.csv'
PREFIX='EC_receptor_M01_12_4l'
WEIGHTS={
 'balanced': {'fc':.25,'brain':.30,'enrichment':.25,'identity':.20},
 'stroke_emphasis': {'fc':.45,'brain':.20,'enrichment':.15,'identity':.20},
 'human_emphasis': {'fc':.10,'brain':.45,'enrichment':.30,'identity':.15},
 'orthology_emphasis': {'fc':.15,'brain':.20,'enrichment':.15,'identity':.50},
}
KEYS={
 'fc':'mouse_D02_logFC', 'brain':'human_brain_EC_median_CPM_atlas',
 'enrichment':'human_EC_nonEC_log2_ratio_atlas',
 'identity':'mouse_human_extracellular_identity_pct',
}
# These are transparent *qualitative* judgements informed by the source-linked
# 19-paper full-text ledger, not outputs of a mathematical calculation.
FULLTEXT={
 'PROCR':('native ligand blood-to-brain evidence','APC-dependent in-vivo influx and FVIIa cell transport, not new peptide',
          'coagulation/APC competition; soluble EPCR; peripheral sink','provisional_primary_1',
          'non-competing accessible human luminal epitope; intact engineered peptide transcytosis; coagulation safety'),
 'TM4SF1':('peripheral endothelial uptake only','surface-binding antibody internalization, no BBB basolateral release',
           'species-specific glycan recognition; repair/angiogenesis; skin EC sink','provisional_primary_2',
           'human stroke luminal protein; species-compatible glycan-aware epitope; barrier transport; binder-only safety'),
 'SELP':('brain disease-model nanocarrier transport','fucoidan nanoparticle-CAV1 tumor-model transport, not monovalent peptide',
         'platelet and vascular inflammation; 0.303 healthy-human EC CPM','conditional_reserve_1',
         'human stroke luminal abundance; monovalent peptide transport; platelet binding and anti-thrombotic safety'),
 'ECSCR':('indirect KDR-linked trafficking','VEGFR2 signalling/endosomal colocalization, no ECSCR shuttle',
          'VEGF/PI3K/eNOS, insulin availability, 46% extracellular identity','reserve_2',
          'human luminal protein; receptor-specific peptide uptake/transcytosis; signalling nonperturbation'),
 'ACKR1':('opposite-direction native chemokine transport','brain/abluminal to blood/luminal intact chemokine export',
          'RBC Duffy sink, leukocyte migration, inflammatory pathways','direction_mismatch_reserve',
          'prove blood-to-brain cargo transport; erythroid genotype-specific off-target and chemokine safety'),
 'LRP1':('peptide benchmark independent of stroke gate','A2/L57 evidence; L57 brain radioactivity and CL4 binding; partial LRP1 contribution',
         'wide tissue distribution, clearance/amyloid pathways, membrane polarity','benchmark_only',
         'exact peptide-sequence scoring; receptor-specific transport and intact cargo; binding epitope, ligand competition'),
}

def read_csv(p):
 with open(p, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def write_csv(p,rows,columns):
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
def num(v):
 try:
  if v is None or str(v).strip()=='':return None
  x=float(v);return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(v,n=5):return '' if v is None else round(v,n)
def ranked_pct(rows,key):
 vals=sorted((num(r.get(key)),r['relationship_id']) for r in rows if num(r.get(key)) is not None)
 n=len(vals);out={}
 for i,(x,rid) in enumerate(vals):
  count_lower=sum(y<x for y,_ in vals)
  count_eq=sum(y==x for y,_ in vals)
  out[rid]=(count_lower+(count_eq-1)/2)/(n-1) if n>1 else 0.5
 return out

def locate(root,name):
 for p in (root/'data/processed/review_inputs'/name,root/'data/processed/review_outputs'/name,root/'data/processed'/name,root/name,Path('/mnt/data')/name):
  if p.exists():return p
 raise FileNotFoundError(f'Missing {name}; put in data/processed/review_outputs/ first')

def run(root, outdir):
 mp=locate(root,MASTER); sp=locate(root,SIX)
 master=read_csv(mp); six=read_csv(sp)
 # Pin the version-1 evidence baseline to avoid silently generating a changed result.
 assert sha(mp)=='b15905ba0c237fccd83347ce31b901746ff94fc1b59f39b4e79ba11141030329','Master changed: version a new 12.4l audit'
 assert sha(sp)=='0310c829a20bfa2e170f4ea105c8c6beb071e2619c89740eebdfcc05107edede','Six-row source changed: version a new 12.4l audit'
 assert len(master)==125,(len(master),'relationships')
 assert len(six)==6,(len(six),'shortlist')
 assert len({r['relationship_id'] for r in master})==125
 assert len({r['mouse_gene'] for r in master})==114
 assert len({r['relationship_id'] for r in six})==6
 lookup={r['relationship_id']:r for r in master}
 for r in six:
  assert r['relationship_id'] in lookup
  t=lookup[r['relationship_id']]
  for a,b in [('D02_logFC','mouse_D02_logFC'),('D02_FDR','mouse_D02_FDR'),
       ('LOSO_direction_stability_pct','mouse_D02_LOSO_direction_stability_pct'),
       ('human_brain_EC_median_CPM','human_brain_EC_median_CPM_atlas'),
       ('human_EC_nonEC_log2_ratio','human_EC_nonEC_log2_ratio_atlas'),
       ('extracellular_sequence_identity_pct','mouse_human_extracellular_identity_pct')]:
   assert r[a]==t[b],(r['relationship_id'],a,r[a],t[b])
 positive=[r for r in master if (num(r['mouse_D02_FDR']) is not None and num(r['mouse_D02_FDR'])<.05 and num(r['mouse_D02_logFC']) is not None and num(r['mouse_D02_logFC'])>0)]
 assert len(positive)==94,len(positive)
 pr={k:ranked_pct(positive,source) for k,source in KEYS.items()}
 audited=[]
 for r in master:
  rid=r['relationship_id']; hg=r['human_gene'].upper(); fdr=num(r['mouse_D02_FDR']); fc=num(r['mouse_D02_logFC']);
  sig=(fdr is not None and fdr<.05 and fc is not None and fc>0)
  topo=r['mouse_protein_topology_context']; br=num(r['human_brain_EC_median_CPM_atlas']);loso=num(r['mouse_D02_LOSO_direction_stability_pct'])
  sr=[x for x in six if x['relationship_id']==rid]
  designated=sr[0]['candidate_role'].lower() if sr else 'not_selected_for_fulltext_pass2'
  if fdr is None: gate='D02_not_measured'
  elif not sig: gate='not_significant_up_at_D02'
  else: gate='D02_sig_up'
  topoflag=('intracellular_or_organelle_screen_defer' if topo=='intracellular_or_organelle' else
    'secreted_not_membrane_receptor_screen_defer' if topo=='secreted_or_extracellular' else
    'surface_or_membrane_plausible_screen_only' if topo in {'surface_accessible','membrane_external_topology','surface_mature_chain_supported','surface_isoform_dependent'} else
    'topology_unresolved_screen_only')
  metrics={k:pr[k].get(rid) for k in KEYS}
  covered={k:v for k,v in metrics.items() if v is not None}
  scores={}
  for scheme,ws in WEIGHTS.items():
   cov=sum(ws[k] for k in covered)
   scores[scheme]=(sum(ws[k]*covered[k] for k in covered)/cov if cov else None)
  for field,val in metrics.items():
   pass
  fulltext=FULLTEXT.get(hg)
  # Many-to-many orthology: relationship-specific records remain distinct.
  status=('benchmark_outside_D02_discovery_screen' if hg=='LRP1' else
   'fulltext_adjudicated_conditional' if fulltext and hg!='LRP1' else
   'pass1_literature_screen_only_NOT_fulltext_negative')
  audit={
    'relationship_id':rid,'mouse_gene':r['mouse_gene'],'human_gene':r['human_gene'],
    'screen_gate':gate,'topology_screen_flag':topoflag,
    'human_brain_CPM_missing':br is None,'LOSO_below_80pct':loso is not None and loso<80,
    'human_extracellular_identity_missing':num(r['mouse_human_extracellular_identity_pct']) is None,
    'D02_logFC':r['mouse_D02_logFC'],'D02_FDR':r['mouse_D02_FDR'],'LOSO_pct':r['mouse_D02_LOSO_direction_stability_pct'],
    'human_brain_EC_CPM':r['human_brain_EC_median_CPM_atlas'],
    'human_EC_nonEC_log2':r['human_EC_nonEC_log2_ratio_atlas'],
    'ECD_identity_pct':r['mouse_human_extracellular_identity_pct'],
    'quantile_FC':fmt(metrics['fc']),'quantile_human_brain_CPM':fmt(metrics['brain']),
    'quantile_endothelial_enrichment':fmt(metrics['enrichment']),
    'quantile_ECD_identity':fmt(metrics['identity']),
    'quant_score_balanced_partial':fmt(scores['balanced']),
    'quant_score_stroke_emphasis_partial':fmt(scores['stroke_emphasis']),
    'quant_score_human_emphasis_partial':fmt(scores['human_emphasis']),
    'quant_score_orthology_emphasis_partial':fmt(scores['orthology_emphasis']),
    'quant_metric_count':len(covered),'quant_full_4_metric_coverage':len(covered)==4,
    'pass2_role':designated,'biological_evidence_status':status,
    'direction_and_endpoint_fulltext':fulltext[1] if fulltext else 'NOT_FULLTEXT_ADJUDICATED; screened literature only',
    'physiological_safety_fulltext':fulltext[2] if fulltext else 'NOT_FULLTEXT_ADJUDICATED; screened literature only',
    'screened_luminal_source':r['screened_native_luminal_access'],
    'screened_transport_source':r['screened_directional_BBB_transport'],
    'screened_peripheral_safety_source':r['screened_peripheral_and_soluble_risks'],
    'pass1_citation_keys':r['screened_literature_reference_keys'],
    'source_topology_discordance_warning':any(x in r['screened_native_luminal_access'].lower() for x in ('predominantly intracellular','mainly intracellular','native er membrane','er-resident','er membrane component','not demonstrated as native blood-facing')),
    'explicit_decision':'PROVISIONAL '+fulltext[3] if fulltext else 'NO FINAL DECISION; no absence-of-evidence exclusion',
    'critical_experiment':fulltext[4] if fulltext else 'Full-text evaluation only if promoted; confirm target access/directional intact peptide transport/safety'
   }
  audited.append(audit)
 assert len(audited)==125
 complete=[r for r in audited if r['screen_gate']=='D02_sig_up' and r['quant_full_4_metric_coverage']]
 for scheme in WEIGHTS:
  key='quant_score_'+scheme+'_partial'
  ranked=sorted(complete,key=lambda r:-float(r[key]))
  lookup_rank={r['relationship_id']:i+1 for i,r in enumerate(ranked)}
  for r in audited:r['complete_case_rank_'+scheme]=lookup_rank.get(r['relationship_id'],'')
 human_counts=Counter(r['human_gene'] for r in master)
 for r in audited:r['human_symbol_relationship_count']=human_counts[r['human_gene']]
 sel={r['human_gene'].upper():r for r in audited if r['human_gene'].upper() in FULLTEXT}
 outdir.mkdir(parents=True,exist_ok=True)
 auditp=outdir/(PREFIX+'_125_gate_audit_v1.csv');write_csv(auditp,audited,list(audited[0]))
 sixrows=[]
 for r in six:
  hg=r['mouse_human_gene_pair'].split('→')[-1].strip().upper()
  sr=sel[hg]; sub=FULLTEXT[hg]
  sixrows.append({
   'receptor':hg,'role':'Benchmark' if hg=='LRP1' else 'Discovery',
   'final_provisional_status':sub[3],
   'D02_logFC':r['D02_logFC'],'D02_FDR':r['D02_FDR'],'LOSO_pct':r['LOSO_direction_stability_pct'],
   'human_brain_EC_CPM':r['human_brain_EC_median_CPM'],
   'human_EC_nonEC_log2':r['human_EC_nonEC_log2_ratio'],
   'ECD_identity_pct':r['extracellular_sequence_identity_pct'],
   'endpoint_level':sub[0], 'endpoint_scientific_scope':sub[1],
   'functional_peripheral_safety':sub[2],
   'numeric_balanced_score_exploratory':sr['quant_score_balanced_partial'] if hg!='LRP1' else '',
   'quant_metric_count':sr['quant_metric_count'] if hg!='LRP1' else '',
   'balanced_rank_among_56_complete_case':sr['complete_case_rank_balanced'] if hg!='LRP1' else '',
   'stroke_emphasis_rank_among_56_complete_case':sr['complete_case_rank_stroke_emphasis'] if hg!='LRP1' else '',
   'human_emphasis_rank_among_56_complete_case':sr['complete_case_rank_human_emphasis'] if hg!='LRP1' else '',
   'orthology_emphasis_rank_among_56_complete_case':sr['complete_case_rank_orthology_emphasis'] if hg!='LRP1' else '',
   'binding_epitope_and_transport_gates':sub[4],
   'literature_references':r['pass2_fulltext_reviewed_keys'],
   'fulltext_verification_status':r['pass2_fulltext_review_status'],
   'not_approved_reason':'No validated native human stroke luminal epitope, safe binder and intact engineered peptide transport',
  })
 sixp=outdir/(PREFIX+'_six_receptor_selection_v1.csv');write_csv(sixp,sixrows,list(sixrows[0]))
 pairs=[r for r in sixrows if r['final_provisional_status'].startswith('provisional_primary')]
 assert {r['receptor'] for r in pairs}=={'PROCR','TM4SF1'}
 two=outdir/(PREFIX+'_provisional_two_plus_LRP1_v1.csv')
 write_csv(two,[r for r in sixrows if r['receptor'] in {'PROCR','TM4SF1','LRP1'}],list(sixrows[0]))
 # Empirical expression/inflammation-only signals do not 'prove' target merit.
 # Stress-test high-scoring membrane candidates not in our original five.
 alternative=[]
 for r in audited:
  if r['human_gene'].upper() in FULLTEXT or r['screen_gate']!='D02_sig_up':continue
  if r['topology_screen_flag'] not in ('surface_or_membrane_plausible_screen_only',):continue
  if not r['quant_full_4_metric_coverage']:continue
  alternative.append(r)
 alternative.sort(key=lambda r:-float(r['quant_score_balanced_partial']))
 altp=outdir/(PREFIX+'_counterfactual_surface_candidates_top20_v1.csv')
 altcols=['relationship_id','mouse_gene','human_gene','D02_logFC','D02_FDR','human_brain_EC_CPM','human_EC_nonEC_log2','ECD_identity_pct','quant_score_balanced_partial','quant_score_stroke_emphasis_partial','quant_score_human_emphasis_partial','quant_score_orthology_emphasis_partial','screened_luminal_source','screened_transport_source','screened_peripheral_safety_source','pass1_citation_keys','biological_evidence_status','source_topology_discordance_warning','human_symbol_relationship_count']
 write_csv(altp,alternative[:20],altcols)
 dedup=[]; seen=set()
 for a in alternative:
  if a['human_gene'] not in seen:
   dedup.append(a);seen.add(a['human_gene'])
 uniquealt=outdir/(PREFIX+'_counterfactual_unique_human_targets_top15_v1.csv')
 write_csv(uniquealt,dedup[:15],altcols)
 summary={
  'method':'POST-HOC TRANSPARENT AUDIT; not preregistered, not intended as calibrated clinical efficacy probability',
  'n_relationships':len(master),'n_unique_mouse_genes':len({r['mouse_gene'] for r in master}),
  'n_sig_D02_up':len(positive),
  'n_sig_up_full_4_metric':sum(r['screen_gate']=='D02_sig_up' and r['quant_full_4_metric_coverage'] for r in audited),
  'comparison_complete_case_size':len(complete),
  'numerical_score_instability':{receptor:{scheme:next(r['complete_case_rank_'+scheme] for r in audited if r['human_gene']==receptor) for scheme in WEIGHTS} for receptor in ['PROCR','TM4SF1','SELP','ECSCR','ACKR1']},
  'n_sig_up_topology_screen_defer':sum(r['screen_gate']=='D02_sig_up' and 'screen_defer' in r['topology_screen_flag'] for r in audited),
  'n_fulltext_discovery':5,'n_benchmark':1,
  'selected_provisional':['PROCR','TM4SF1'],'conditional_high_interest_reserve':'SELP',
  'additional_reserves':['ECSCR','ACKR1'],
  'benchmark':['LRP1','Angiopep-2','L57','Angiopep-7 as comparison-only'],
  'data_integrity':{'master_sha256':sha(mp),'shortlist_v8_sha256':sha(sp)},
  'gate_definitions':{'DE':'D02 FDR<0.05 AND D02 log2FC>0; no magnitude threshold','LOSO':'reported only; <80 flagged, not excluded','human':'missing or low CPM flagged, NEVER presumed no stroke expression','topology':'source screening categories; NOT measured blood-facing access','mechanism':'full-text adjudicated only for five + benchmark; experimental direction and cargo identity explicitly separated','safety':'noncompensatory review and experiments; no toxicity probability estimated'},
  'quant_priority':{'population':'94 significant-up relationships','formula':'weighted mean of empirical percentiles of logFC, human brain EC CPM, EC/nonEC log2, extracellular identity; missing metrics omitted and coverage reported; not a final rank', 'weight_sets':WEIGHTS,'caveat':'No paper-count popularity boost; no compensation for wrong transport direction, nonluminal protein or serious safety'},
  'selection_caveat':'PROCR and TM4SF1 are working choices for computational feasibility; expert/luminal binding/wet-lab gate required before experimental or clinical selection',
  'all_candidates_fulltext':'Only 5/125 discovery candidates were reviewed at full-text level; counterfactual candidates from pass1 must not be described as experimentally rejected',
  'generated_files':{},
 }
 methods=outdir/(PREFIX+'_selection_method_and_limitations_v1.md')
 methods.write_text('''# Module 01: auditable receptor selection — provisional, post-hoc (v1)\n\n## Exact workflow\n\n1. **Universe:** Preserve all 125 mouse→human relationships (114 mouse genes), including many-to-many mappings; never replace missing observations by zero.\n2. **Stroke evidence gate:** Flag D02 log2FC > 0 and FDR < 0.05. Among the uploaded records this gives 94 relationships. Record LOSO direction stability without manufacturing a binary robustness distinction when most are 100%.\n3. **Human and topology context:** Record human brain EC abundance/enrichment, extracellular conservation, membrane localization and peripheral expression. Do NOT equate RNA abundance with native luminal receptor density. A zero/low healthy brain reference does not rule out stroke-induced luminal expression. Explicitly mark organelle-resident and secreted-only source classifications as provisional screening deferrals, not as full-text validated negatives.\n4. **Numerical triage, NOT target selection:** Within 94 D02-up relationships, calculate empirical percentile ranks for four independent quantities: D02 logFC, human brain EC median CPM, human EC/non-EC enrichment, and mouse-human extracellular identity. Balanced weights = 0.25/0.30/0.25/0.20. Three alternative weightings stress-test the dependence on chosen weights. Missing fields are omitted from the partial score, and coverage is reported; never rank a missing-data row as though complete. All weights are analytic conventions selected AFTER seeing the cohort, not preregistered or validated therapeutic cutoffs.\n5. **Manual endpoint-level adjudication:** Use 19 source-reviewed papers/213 experiment records to separately evaluate the *five already shortlisted* receptor mechanisms (native ligand vs nanocarrier vs antibody uptake vs indirect trafficking; transport direction; intact peptide or only isotope signal); **do not extrapolate those endpoint judgments to the other 120 relationships**. Count screening-only alternative hits independently to identify possible omissions. No quantity of citations substitutes for an endpoint.\n6. **Noncompensatory constraints:** A documented opposite-direction transport result, physiologic ligand competition, major vascular safety issue, human luminal uncertainty or erythrocyte/platelet binding cannot be cancelled by high expression or weighted score. Safety must be tested with the actual *nonblocking* binder and cargo, not inferred from knockouts/ADCs.\n7. **Decision:** PROCR + TM4SF1 are provisional computational design arms, distinct in mechanism and quantitative expression; SELP is the highest-priority conditional reserve; ECSCR is a reserve; ACKR1 has an opposite-direction warning. LRP1 is a separate benchmark arm (Angiopep-2, L57; Angiopep-7 for published comparison), **not one of the stroke-upregulated discovery receptors**. No final receptor/epitope or clinical approval.\n\n## Why these two?\n\n- **PROCR:** native mouse blood-to-brain APC precedent, moderate human EC expression/enrichment; must demonstrate short-peptide transport and avoid APC/protein C anticoagulant interference and soluble-EPCR sink.\n- **TM4SF1:** high human brain EC expression/enrichment and surface antibody internalization, but **no proven BBB peptide transcytosis**; glycan-dependent species difference, peripheral skin EC exposure and angiogenesis/repair liability are major translational hurdles.\n- **SELP:** nanocarrier transport in a brain tumour model and strong mouse stroke induction but 0.303 healthy-human brain EC CPM; patient-stroke protein abundance and platelet binding/safety are gating experiments.\n- **ECSCR:** human EC enrichment but no demonstrated receptor-dependent shuttle; 46% ECD identity and vascular signalling liabilities.\n- **ACKR1:** intact native chemokine transport is brain/abluminal→blood/luminal, the reverse of the desired brain-influx route; erythrocyte sink depends on genotype.\n- **LRP1:** positive peptide benchmark evidence is not a validated clinical endpoint; L57 in-situ 0.220±0.052% input vs Angiopep-7 0.053±0.019%, with IV differences not established; Angiopep-2 vs Angiopep-7 are different sequences.\n\n## Reviewer disclosures\n\n- Initial five-receptor shortlist was exploratory/literature-informed, not derived uniquely by the retrospective quantitative rule. Describe it honestly as *two-stage evidence triage with decision audit*.\n- This implementation is a documented reproducibility artifact, not a machine-learned classifier or clinically calibrated scoring model. LLM-assisted note extraction does not mean independent validation; claims must retain paper and experiment IDs.\n- The 20 highest quantitative **alternative** membrane-annotated relationships, and a deduplicated human-target list, are retained as counterfactual stress tests, not described as negative without full-text adjudication. Some top-scoring IFITM mappings are mostly intracellular despite their screening topology label, highlighting the need to cross-check source narratives.
- Numerical weights change ranks markedly, including PROCR, underscoring why the numerical diagnostic alone cannot justify the final target decision.\n- No numerical score is allowed to override unresolved native luminal epitope, actual blood→brain intact peptide transfer, or essential physiologic safety.\n- The selected targets are **provisional computational design arms**; expert confirmation and experiment design remain mandatory.\n\n## Planned minimum experiments for claims beyond computational selection\n\nNative human ischemic brain microvascular immunoluminal access; receptor-specific binder affinity/competition; polarized human BBB integrity plus directional mass-balanced intact peptide basolateral recovery (LC-MS, not just fluorescence or radiotracer); matched knockout/competition and paracellular controls; platelet/coagulation/angiogenesis tests; human cross-reactivity, membrane, glycan and off-organ tissue panel.\n''',encoding='utf-8')
 for p in (auditp,sixp,two,altp,uniquealt,methods):summary['generated_files'][p.name]=sha(p)
 summary_path=outdir/(PREFIX+'_validation_and_provenance_v1.json'); summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('PASS | original relationship count=125, unique mouse genes=114, sig D02-up=94')
 print('PASS | original quantitative values match six receptor v8 exactly')
 print('PASS | five discovery + benchmark separated; no full-text claims invented for other rows')
 print('PASS | produced',len(summary['generated_files']),'audited output files')
 print('Alternative comparison top 10:',[(r['human_gene'],r['quant_score_balanced_partial']) for r in alternative[:10]])
 print('Output directory',outdir)
 return summary

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=None);ap.add_argument('--out',default=None)
 a,_=ap.parse_known_args()
 # If run inside notebook folder, relative parent normally points to repo root.
 here=Path.cwd().resolve()
 root=Path(a.root).resolve() if a.root else (here.parent if here.name=='notebooks' else here)
 out=Path(a.out).resolve() if a.out else root/'data/processed/review_outputs'
 MODULE01_SELECTION_124L=run(root,out)
