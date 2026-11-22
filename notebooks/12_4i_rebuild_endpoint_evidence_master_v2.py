"""OmiPep v0.2 | Module 01 | Cell 12.4i

Cumulative endpoint-level literature importer (screening evidence v2).
Run from existing Jupyter notebook, without upstream reanalysis:

  %run -i 12_4i_rebuild_endpoint_evidence_master_v2.py

Inputs: immutable 12.4d 125-row master and 541-field source dossier, plus
versioned LIT-*-REVIEWED_vN.zip packages in data/processed/review_packets/.
For each batch, chooses highest available revision (v2 supersedes v1) and
asserts the declared v1 hash when both files are present.

Outputs to data/processed/review_outputs/ only, *v2 filenames*. Never edits:
  original dossier, original 29-column expert master, review ZIPs, manual
  adjudication, the citation Google Sheet, previous *v1 derived outputs,
  receptor selection or any protein fragment approval.

The derived claims remain screened, NOT completely full-text/expert-adjudicated.
Their observed results refer exclusively to measured study models. Native
luminal access and intact human BBB transcytosis cannot be inferred.
"""
from __future__ import annotations
import csv, io, hashlib, json, os, re, tempfile, zipfile
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

ROOT_124i = Path(os.environ.get('OMIPEP_REPO_ROOT',str(Path(__file__).resolve().parent.parent))).resolve()
PROC_124i = ROOT_124i/'data'/'processed'
PACKS_124i = PROC_124i/'review_packets'
OUTPUT_124i = PROC_124i/'review_outputs'
SOURCE_124i = PROC_124i/'EC_receptor_12_4_full_relationship_docket_v1.csv'
BASE_124i = PROC_124i/'EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv'
PATTERN_124i = re.compile(r'^(LIT-[PR]\d{2}-\d{2})_REVIEWED_v(\d+)\.zip$')

# Additive, explicit screening-only columns; the upstream 29 are untouched.
EXTRA_COLUMNS_124i = {
 'review_batch_id':'screened_literature_batch_id',
 'review_status':'screened_literature_review_state',
 'reviewed_primary_sources':'screened_literature_reference_keys',
 'n_screened_papers':'screened_literature_n_publications_NOT_adjudicated',
 'n_endpoint_claim_rows':'screened_literature_n_endpoint_claims_NOT_validated_transport',
 'native_luminal_access_summary':'screened_native_luminal_access',
 'binding_internalization_summary':'screened_binding_and_internalization',
 'trafficking_verdict':'screened_intracellular_trafficking',
 'intact_BBB_transcytosis_summary':'screened_directional_BBB_transport',
 'abluminal_intact_generated_peptide_cargo_summary':'screened_abluminal_intact_cargo',
 'surface_class_assessment':'screened_receptor_vs_secreted_surface_class',
 'topology_epitope_unknowns':'screened_epitope_and_isoform_uncertainties',
 'structural_glycan_membrane_constraints':'screened_structure_glycan_membrane_constraints',
 'peripheral_and_soluble_sink':'screened_peripheral_and_soluble_risks',
 'contradictions_and_specificity':'screened_conflicts_and_specificity',
 'necessary_tests':'screened_required_next_experiments',
 'final_expert_adjudication_complete':'literature_final_expert_adjudication_complete',
 'selection_approved':'literature_receptor_approval_flag',
}
# Optional v1 old format can join the same cumulative ledger as a screened record.
LEGACY_FALLBACK_124i={
 'trafficking_verdict':'not_yet_structured_in_legacy_batch',
 'surface_class_assessment':'not_yet_structured_in_legacy_batch',
 'topology_epitope_unknowns':'not_yet_structured_in_legacy_batch',
 'structural_glycan_membrane_constraints':'not_yet_structured_in_legacy_batch',
 'peripheral_and_soluble_sink':'not_yet_structured_in_legacy_batch',
 'contradictions_and_specificity':'not_yet_structured_in_legacy_batch',
 'necessary_tests':'not_yet_structured_in_legacy_batch',
 'final_expert_adjudication_complete':'NO',
 'selection_approved':'NO'
}
CLAIM_RECOMMENDED_V2_124i = {
 'figure_or_table_locator','claim_source_depth','assay_or_readout',
 'observed_result_status','biological_evidence_stage','tested_direction',
 'cargo_format_and_intactness','native_luminal_access_in_tested_model',
 'supports_shuttle_candidate','structural_or_glycan_context',
 'safety_or_peripheral_context','full_text_adjudication_state',
}

def insist_124i(ok, msg):
 if not bool(ok):raise AssertionError('FAIL | '+msg)
 print('PASS | '+msg)

def digest_124i(b): return hashlib.sha256(b).hexdigest()

def read_124i(data, name):
 s=data.decode('utf-8-sig')
 first=next(csv.reader(io.StringIO(s)),[])
 insist_124i(bool(first) and len(first)==len(set(first)),name+' has distinct CSV headers')
 return pd.read_csv(io.StringIO(s),dtype=str,keep_default_na=False,low_memory=False)

def atomic_124i(path, b):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.is_file() and path.read_bytes()==b:
  print('UNCHANGED |', path.relative_to(ROOT_124i));return
 with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.124i-',delete=False) as f:
  temp=Path(f.name)
  try:
   f.write(b);f.flush();os.fsync(f.fileno())
  except Exception:
   temp.unlink(missing_ok=True);raise
 temp.replace(path)
 print('REBUILT |',path.relative_to(ROOT_124i))

def csvbytes_124i(df):return df.to_csv(index=False,lineterminator='\n').encode('utf8')

def normdoi_124i(x):
 s=str(x).strip().lower()
 s=re.sub(r'^https?://(?:dx\.)?doi\.org/','',s)
 return re.sub(r'^doi:\s*','',s).strip(' /')

def extract_packet_124i(path, ids):
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  insist_124i(len(names)==len(set(names)) and not any('..' in Path(n).parts or n.startswith('/') for n in names),path.name+': safe unique ZIP paths')
  need={'review_manifest.json','references.csv','evidence_claims.csv','relationship_review.csv'}
  insist_124i(need.issubset(names),path.name+': required members')
  m=json.loads(z.read('review_manifest.json'))
  hashed=m['sha256_of_members']
  insist_124i(all(k in names and digest_124i(z.read(k))==v for k,v in hashed.items()) and (need-{'review_manifest.json'}).issubset(hashed),path.name+': all declared member hashes')
  refs=read_124i(z.read('references.csv'),path.name+'/references')
  claims=read_124i(z.read('evidence_claims.csv'),path.name+'/claims')
  rels=read_124i(z.read('relationship_review.csv'),path.name+'/relationships')
 match=PATTERN_124i.fullmatch(path.name)
 assert match is not None
 b,ver=match.group(1),int(match.group(2))
 insist_124i(m.get('review_batch_id')==b and int(m['bundle_version'])==ver,path.name+': ID and version agree')
 for frame,key in [(refs,'citation_key'),(claims,'claim_id'),(rels,'relationship_id')]:
  insist_124i(key in frame and frame[key].ne('').all() and frame[key].is_unique,path.name+': unique '+key)
 insist_124i(set(rels.relationship_id)==set(m['relationship_ids']) and set(rels.relationship_id)<=ids,path.name+': frozen source relationship membership')
 insist_124i(1<=len(rels)<=5,path.name+': max 5 relationships')
 insist_124i(set(claims.relationship_id)<=set(rels.relationship_id) and set(claims.citation_key)<=set(refs.citation_key),path.name+': links internally valid')
 insist_124i(all(len(frame)==int(m[mk]) for frame,mk in [(refs,'n_primary_papers'),(claims,'n_evidence_claims'),(rels,'n_relationships')]),path.name+': declared counts match')
 for frame in (refs,claims,rels):
  insist_124i('review_batch_id' in frame and frame.review_batch_id.eq(b).all(),path.name+': row batch tags')
 if ver>=2:
  insist_124i(CLAIM_RECOMMENDED_V2_124i.issubset(claims.columns),path.name+': endpoint detail fields present')
  insist_124i({'n_endpoint_claim_rows','final_expert_adjudication_complete','selection_approved'}.issubset(rels.columns),path.name+': structured review guardrails')
  for _,row in rels.iterrows():
   observed=int(claims.relationship_id.eq(row.relationship_id).sum())
   insist_124i(int(row['n_endpoint_claim_rows'])==observed,row.relationship_id+': actual claim count matches summary')
   insist_124i(row['final_expert_adjudication_complete']=='NO' and row['selection_approved']=='NO',row.relationship_id+': not adjudicated and no selection')
 else:
  print('NOTICE | Legacy v1 screening packet accepted, without endpoint refinements:',b)
 return b,ver,refs,claims,rels,m

def run_124i():
 print('\n=== OmiPep M01 | 12.4i endpoint evidence importer v2 ===')
 insist_124i(SOURCE_124i.is_file() and BASE_124i.is_file() and PACKS_124i.is_dir(),'source tables and packets directory exist')
 orig_hash={p.name:digest_124i(p.read_bytes()) for p in (SOURCE_124i,BASE_124i)}
 dossier=read_124i(SOURCE_124i.read_bytes(),'dossier')
 base=read_124i(BASE_124i.read_bytes(),'29-column master')
 insist_124i(dossier.shape==(125,541) and base.shape==(125,29),'125 x 541 dossier and 125 x 29 baseline')
 for x in (base,dossier):
  insist_124i('relationship_id' in x and x.relationship_id.is_unique and x.relationship_id.ne('').all(),'source relationship IDs unique')
 insist_124i(set(base.relationship_id)==set(dossier.relationship_id),'exact same 125 source IDs')
 insist_124i(dossier.gene.nunique()==114,'114 mouse genes retained')
 for col,expect in [('docking_fragment_status','not_approved'),('final_selection_status','not_selected')]:
  insist_124i(col in dossier and dossier[col].eq(expect).all(),f'{col}: all 125 {expect}')
 groups=defaultdict(list)
 for p in PACKS_124i.iterdir():
  match=PATTERN_124i.fullmatch(p.name)
  if match and p.is_file():groups[match.group(1)].append((int(match.group(2)),p))
 insist_124i(bool(groups),'>=1 reviewed package found')
 active=[];superseded=[]
 for batch in sorted(groups):
  candidates=sorted(groups[batch],key=lambda x:x[0])
  insist_124i(len(set(v for v,_ in candidates))==len(candidates),'no duplicate revision for '+batch)
  chosen=candidates[-1];active.append(chosen[1]);superseded.extend(str(p.name) for _,p in candidates[:-1])
  if len(candidates)>1:
   with zipfile.ZipFile(chosen[1]) as z:m=json.loads(z.read('review_manifest.json'))
   old_v=[p for v,p in candidates[:-1] if v==1]
   if old_v and m.get('superseded_reviewed_zip_sha256'):
    insist_124i(digest_124i(old_v[-1].read_bytes())==m['superseded_reviewed_zip_sha256'],batch+': original v1 cryptographic supersession verified')
  print('ACTIVE  |',chosen[1].name)
 print('ARCHIVED IGNORE |',', '.join(superseded) or 'none')
 parts={'references':[],'claims':[],'reviews':[]};packet_provenance=[]
 for p in active:
  batch,ver,r,c,rel,m=extract_packet_124i(p,set(base.relationship_id))
  # Legacy record fields are explicitly unknown, not silently synthesized as negative.
  for col in CLAIM_RECOMMENDED_V2_124i:
   if col not in c.columns:c[col]=''
  for col,fill in LEGACY_FALLBACK_124i.items():
   if col not in rel.columns:rel[col]=fill
  if 'n_endpoint_claim_rows' not in rel.columns:
   rel['n_endpoint_claim_rows']=rel.relationship_id.map(c.relationship_id.value_counts()).fillna(0).astype(int).astype(str)
  parts['references'].append(r);parts['claims'].append(c);parts['reviews'].append(rel)
  packet_provenance.append({'batch_id':batch,'selected_version':ver,'filename':p.name,
                            'sha256':digest_124i(p.read_bytes()),'relationships':len(rel),
                            'claims':len(c),'publications_in_packet':len(r)})
 refs=pd.concat(parts['references'],ignore_index=True).fillna('')
 claims=pd.concat(parts['claims'],ignore_index=True).fillna('')
 reviews=pd.concat(parts['reviews'],ignore_index=True).fillna('')
 insist_124i(claims.claim_id.is_unique and reviews.relationship_id.is_unique,'unique cumulative claim and relationship IDs')
 insist_124i(set(claims.relationship_id)<=set(base.relationship_id),'no invented receptor relationships')
 insist_124i(set(claims.citation_key)<=set(refs.citation_key),'claim keys present in bibliography')
 refs['normalized_doi']=refs.doi.map(normdoi_124i)
 refs['publication_id']=['DOI:'+d if d else 'PMID:'+str(p).strip() for d,p in zip(refs.normalized_doi,refs.pmid)]
 insist_124i(~refs.publication_id.eq('PMID:').any(),'each publication has DOI or PMID')
 for pid,g in refs.groupby('publication_id'):
  insist_124i(g.citation_key.nunique()==1,'unique citation key for '+pid)
 for k,g in refs.groupby('citation_key'):
  insist_124i(g.publication_id.nunique()==1,'stable publication identity for '+k)
 unique=refs.drop_duplicates('publication_id',keep='first').reset_index(drop=True)
 # Cross-check raw count by relationship and safely include review summary fields.
 for _,r in reviews.iterrows():
  n=int(claims.relationship_id.eq(r.relationship_id).sum())
  insist_124i(n==int(r['n_endpoint_claim_rows']),f'{r.relationship_id}: {n} claim rows')
  insist_124i(r.get('selection_approved','NO')=='NO' and r.get('final_expert_adjudication_complete','NO')=='NO',r.relationship_id+': no approved target')
  all_keys=set(x.strip() for x in str(r['reviewed_primary_sources']).split(';') if x.strip())
  mapped=set(claims.loc[claims.relationship_id.eq(r.relationship_id),'citation_key'])
  insist_124i(all_keys==mapped,r.relationship_id+': exact claim/reference linkage')
 target=reviews[['relationship_id']].copy()
 for source,derived in EXTRA_COLUMNS_124i.items():
  target[derived]=reviews[source] if source in reviews else ''
 master=base.merge(target,on='relationship_id',how='left',sort=False,validate='one_to_one')
 for c in EXTRA_COLUMNS_124i.values():master[c]=master[c].fillna('')
 insist_124i(master.shape==(125,29+len(EXTRA_COLUMNS_124i)),'complete 125-row master with additive structured evidence')
 insist_124i(master[list(base.columns)].equals(base),'all 29 baseline scientific fields untouched')
 insist_124i(master.loc[~master.relationship_id.isin(reviews.relationship_id),'screened_literature_review_state'].eq('').all(),'unreviewed remain unknown, not negative')
 # Keep exactly the existing verified biology fields as they were in source baseline.
 for col in base.columns:
  if col.startswith('verified_') or col.startswith('n_verified_'):
   insist_124i(master[col].equals(base[col]),'source verification field unchanged: '+col)
 outputs={
  'EC_receptor_M01_reference_index_v2.csv':csvbytes_124i(unique),
  'EC_receptor_M01_endpoint_claim_ledger_v2.csv':csvbytes_124i(claims),
  'EC_receptor_M01_relationship_evidence_summaries_v2.csv':csvbytes_124i(reviews),
  'EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv':csvbytes_124i(master),
 }
 audit={'module':'OmiPep v0.2 M01','cell':'12.4i','scientific_status':'screened_not_final_expert_fulltext_adjudication',
        'source_relationships':125,'source_genes':114,'relationship_rows_with_screened_claims':len(reviews),
        'source_columns_preserved':29,'master_columns':len(master.columns),
        'n_publications':len(unique),'n_endpoint_claim_rows':len(claims),
        'n_multi_claim_publications':int((claims.groupby('citation_key').size()>1).sum()),
        'active_packets':packet_provenance,'archived_packets_ignored':superseded,
        'original_source_hashes_sha256':orig_hash,
        'output_sha256':{k:digest_124i(v) for k,v in outputs.items()},
        'policies':['no receptor selection','not tested is not negative','no derived scientific verified_* values',
                    'the reviewed ZIP with highest revision per batch is active',
                    'no original source or prior v1 outputs changed','Google Sheet not edited by importer']}
 outputs['EC_receptor_M01_endpoint_import_manifest_v2.json']=(json.dumps(audit,indent=2,ensure_ascii=False)+'\n').encode('utf8')
 for name,data in outputs.items():atomic_124i(OUTPUT_124i/name,data)
 for p in (SOURCE_124i,BASE_124i):
  insist_124i(digest_124i(p.read_bytes())==orig_hash[p.name],p.name+': upstream SHA-256 unchanged')
 print(f'\nPASS | active {len(active)} batch(es); publications {len(unique)}; endpoint claims {len(claims)}; screened relationships {len(reviews)}/125')
 print('PASS | all original data/selection decisions preserved; Cell 12.5 remains blocked')
 print('MASTER CSV |',OUTPUT_124i/'EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv')
 try:
  from IPython.display import display
  with pd.option_context('display.max_rows',125,'display.max_columns',60,'display.max_colwidth',70):
   display(master)
  cols=['mouse_gene','human_gene','screened_literature_n_publications_NOT_adjudicated','screened_literature_n_endpoint_claims_NOT_validated_transport','screened_native_luminal_access','screened_directional_BBB_transport','screened_conflicts_and_specificity']
  print('\nFIVE-RELATIONSHIP SCIENTIFIC SCREENING VIEW (NOT MERIT RANK)')
  with pd.option_context('display.max_columns',len(cols),'display.max_colwidth',90):
   display(master.loc[master.screened_literature_review_state.ne(''),cols])
 except ImportError:
  print(master.loc[master.screened_literature_review_state.ne(''),['mouse_gene','human_gene']].to_string(index=False))
 return master,claims,unique,reviews

OMIPEP_M01_MASTER_124i, OMIPEP_M01_ENDPOINT_LEDGER_124i, OMIPEP_M01_REFS_124i, OMIPEP_M01_REVIEWED_124i = run_124i()
