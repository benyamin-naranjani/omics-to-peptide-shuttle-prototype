#!/usr/bin/env python3
"""Validate installed TM4SF1 incremental exports with source-file sha256 and invariant membership."""
import csv,hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/processed/review_outputs')
report=json.loads((out/'EC_receptor_M01_12_4k_TM4SF1_fulltext_validation_v1.json').read_text())
def read_csv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for item in report['files']:
 f=(out.parents[2]/'notebooks'/item['name']) if item['name']=='12_4k_verify_TM4SF1_review.py' else out/item['name']
 assert f.exists(),f'Missing {f}'
 assert sha(f)==item['sha256'],f'Changed checksum {f}'
 print('PASS',f.name,'sha256')
papers=read_csv(out/'EC_receptor_M01_12_4k_fulltext_paper_ledger_v2.csv')
assert len(papers)==7 and len({r['citation_key'] for r in papers})==7
assert sum(r['receptor']=='TM4SF1' for r in papers)==4
exp=read_csv(out/'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v1.csv')
assert len(exp)==47 and sum(r['relationship_id'].startswith('Tm4sf1::') for r in exp)==31
short=read_csv(out/'EC_receptor_M01_12_4k_shortlist_WORKING_v4.csv')
assert len(short)==6 and len({r['relationship_id'] for r in short})==6
p=[r for r in short if r['relationship_id']=='Tm4sf1::ENSG00000169908']
assert len(p)==1 and p[0]['pass2_fulltext_review_status'].startswith('FOUR_PRIORITY_FULLTEXTS_REVIEWED')
comparison=read_csv(out/'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v2.csv')
assert len(comparison)==6
for pdf in report['source_pdfs']:
 f=out.parent/'review_sources'/'TM4SF1'/pdf['name']
 assert f.exists() and sha(f)==pdf['sha256'],f'Source PDF missing/changed: {f}'
 print('PASS',f.name,'source PDF')
print('SUCCESS: TM4SF1 v1 installation, cumulative ledger and six-row table verified. NO receptor approval.')
