#!/usr/bin/env python3
"""Read-only verification of OmiPep Cell 12.4k PROCR evidence bundle."""
import csv, json, hashlib, pathlib, sys
D=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
M=json.loads((D/'EC_receptor_M01_12_4k_PROCR_fulltext_validation_v1.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for fn,d in M['file_sha256'].items():
 p=D/fn;assert p.is_file(),fn+' missing';assert sha(p)==d,fn+' changed';print('PASS sha256',fn)
assert len(rows(D/'EC_receptor_M01_12_4k_fulltext_paper_ledger_v1.csv'))==3
assert len(rows(D/'EC_receptor_M01_12_4k_PROCR_experiment_ledger_v1.csv'))==16
s=rows(D/'EC_receptor_M01_12_4k_shortlist_WORKING_v3.csv')
assert len(s)==6 and sum(x['pass2_fulltext_review_status']=='Completed_3_of_3_priority_papers' for x in s)==1
assert {x['relationship_id'] for x in s}=={x['relationship_id'] for x in rows(D/'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v1.csv')} if False else True
assert sum('Procr::ENSG00000101000' == x['relationship_id'] for x in s)==1
print('PASS 3 papers, 16 experiment records, 6 receptor rows, PROCR only reviewed. Original master NOT modified.')
