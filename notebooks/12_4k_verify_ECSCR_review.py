#!/usr/bin/env python3
"""OmiPep 12.4k ECSCR install sanity-check; read-only."""
import csv,hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/processed/review_outputs')
root=out.parent.parent.parent
report=json.loads((out/'EC_receptor_M01_12_4k_ECSCR_fulltext_validation_v1.json').read_text())
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for item in report['output_files']:
 p=out/item['name']; assert p.exists() and hash(p)==item['sha256'],f'bad output hash {p}'
 print('PASS',p.name,'sha256')
for item in report['sources']:
 p=out.parent/'review_sources'/'ECSCR'/item['name']; assert p.exists() and hash(p)==item['sha256'],f'bad PDF hash {p}'
 print('PASS',p.name,'source hash')
papers=rows(out/'EC_receptor_M01_12_4k_fulltext_paper_ledger_v3.csv')
exps=rows(out/'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v2.csv')
short=rows(out/'EC_receptor_M01_12_4k_shortlist_WORKING_v5.csv')
comp=rows(out/'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v3.csv')
assert len(papers)==10 and len(set(r['citation_key'] for r in papers))==10
assert sum(r['receptor']=='ECSCR' for r in papers)==3 and sum(r['receptor']=='TM4SF1' for r in papers)==4 and sum(r['receptor'].startswith('PROCR') for r in papers)==3
assert len(exps)==80 and len(set(r['experiment_id'] for r in exps))==80
assert sum(r['relationship_id'].startswith('Ecscr::') for r in exps)==33
assert len(short)==6 and len(set(r['relationship_id'] for r in short))==6 and len(comp)==6
r=[r for r in short if r['relationship_id']=='Ecscr::ENSG00000249751'];assert len(r)==1
assert r[0]['pass2_fulltext_review_status']=='THREE_PRIORITY_FULLTEXTS_REVIEWED__RETAIN_PROVISIONAL_NOT_APPROVED'
# verify historical prefix and unchanged non-ECSCR rows if previous input files remain installed
prior=[('EC_receptor_M01_12_4k_fulltext_paper_ledger_v2.csv',papers[:7]),('EC_receptor_M01_12_4k_cumulative_experiment_ledger_v1.csv',exps[:47])]
for n,prefix in prior:
 p=out/n
 if p.exists():assert rows(p)==prefix,f'append-only prefix changed {n}'
for prev,new,name,field,identifier in [
 ('EC_receptor_M01_12_4k_shortlist_WORKING_v4.csv','EC_receptor_M01_12_4k_shortlist_WORKING_v5.csv','shortlist','relationship_id','Ecscr::ENSG00000249751'),
 ('EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v2.csv','EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v3.csv','comparison','receptor_and_role','Ecscr → ECSCR')]:
 old=out/prev
 if old.exists():
  a=rows(old); b=rows(out/new)
  assert len(a)==len(b)==6
  assert all(x==y for x,y in zip(a,b) if (not x[field].startswith(identifier) if name=='comparison' else x[field]!=identifier)),f'non-ECSCR row changed in {name}'
print('SUCCESS: ECSCR installed; 10 papers + 80 experiments; 6-row shortlist stable; zero target approvals')
