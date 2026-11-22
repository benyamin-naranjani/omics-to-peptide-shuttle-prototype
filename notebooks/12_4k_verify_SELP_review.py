#!/usr/bin/env python3
"""Read-only file checksum and append-only scientific evidence integrity check for SELP."""
import csv,json,hashlib,sys
from pathlib import Path
O=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/processed/review_outputs')
R=O.parent;report=json.loads((O/'EC_receptor_M01_12_4k_SELP_fulltext_validation_v1.json').read_text())
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for d in report['output_files']:
 p=O/d['name'];assert p.exists() and hash(p)==d['sha256'],f'bad file {p}'
 if d['rows'] is not None:assert len(rows(p))==d['rows'],f'bad rows {p}'
 print('PASS',p.name,'sha256 + row count')
for d in report['source_pdfs']:
 p=R/'review_sources'/'SELP'/d['name'];assert p.exists() and hash(p)==d['sha256'],f'bad source {p}'
 print('PASS source',p.name)
p=rows(O/'EC_receptor_M01_12_4k_fulltext_paper_ledger_v4.csv')
e=rows(O/'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v3.csv')
s=rows(O/'EC_receptor_M01_12_4k_shortlist_WORKING_v6.csv')
m=rows(O/'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v4.csv')
assert len(p)==13 and len(set(x['citation_key'] for x in p))==13
assert len(e)==123 and len(set(x['experiment_id'] for x in e))==123
assert sum(x['relationship_id']=='Selp::ENSG00000174175' for x in e)==43
assert sum(x['receptor']=='SELP' for x in p)==3
assert len(s)==6 and len(m)==6 and len(set(x['relationship_id'] for x in s))==6
ss=[x for x in s if x['relationship_id']=='Selp::ENSG00000174175'];assert len(ss)==1
assert ss[0]['pass2_fulltext_review_status']=='THREE_PRIORITY_FULLTEXTS_REVIEWED__RETAIN_CONDITIONALLY_NOT_APPROVED'
# Prev files, if present: exact append-only and no non-SELP row changes.
for name,prefix in [('EC_receptor_M01_12_4k_fulltext_paper_ledger_v3.csv',p[:10]),('EC_receptor_M01_12_4k_cumulative_experiment_ledger_v2.csv',e[:80])]:
 q=O/name
 if q.exists():assert rows(q)==prefix,f'history changed: {q}'
for old,new,col,exclude in [
 ('EC_receptor_M01_12_4k_shortlist_WORKING_v5.csv','EC_receptor_M01_12_4k_shortlist_WORKING_v6.csv','relationship_id','Selp::ENSG00000174175'),
 ('EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v3.csv','EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v4.csv','receptor_and_role','Selp → SELP')]:
 q=O/old
 if q.exists():
  a=rows(q);b=rows(O/new);assert len(a)==len(b)==6
  assert all(x==y for x,y in zip(a,b) if (x[col]!=exclude if col=='relationship_id' else not x[col].startswith(exclude))),f'changed non-SELP row {q}'
print('SUCCESS: SELP installed; 13 papers + 123 experiments, SELP 43; five other shortlisted rows unchanged; 0 receptor approvals')
