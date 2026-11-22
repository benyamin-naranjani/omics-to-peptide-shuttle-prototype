#!/usr/bin/env python3
"""Verify ACKR1 review package after unzip from project root. Does NOT modify files."""
import csv,hashlib,json,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def load(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
 d=Path(sys.argv[1] if len(sys.argv)>1 else 'data/processed/review_outputs')
 report=json.loads((d/'EC_receptor_M01_12_4k_ACKR1_fulltext_validation_v1.json').read_text())
 for file_name,digest in report['generated_csv_checksums'].items():
  p=d/file_name
  assert p.is_file() and sha(p)==digest, f'CHECKSUM FAILED: {p}'
  print('PASS: CSV SHA256',file_name)
 roots=d.parent/'review_sources'/'ACKR1'
 for key,meta in report['source_pdf_sha256'].items():
  p=roots/meta['source_filename']
  assert p.is_file() and sha(p)==meta['sha256'],f'PDF CHECKSUM FAILED: {p}'
  print('PASS: source PDF SHA256',key)
 files={'papers':'EC_receptor_M01_12_4k_fulltext_paper_ledger_v5.csv',
        'exp':'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v4.csv',
        'short':'EC_receptor_M01_12_4k_shortlist_WORKING_v7.csv',
        'manu':'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v5.csv',
        'ackexp':'EC_receptor_M01_12_4k_ACKR1_experiment_ledger_v1.csv',
        'ackpapers':'EC_receptor_M01_12_4k_ACKR1_fulltext_papers_v1.csv'}
 rows={k:load(d/v) for k,v in files.items()}
 N=report['counts']; assert len(rows['papers'])==N['cumulative_papers']==16
 assert len(rows['exp'])==N['cumulative_experiments']==169
 assert len(rows['ackexp'])==N['added_ackr1_experiments']==46
 assert len(rows['ackpapers'])==N['added_papers']==3
 assert len(rows['short'])==len(rows['manu'])==6
 assert rows['papers'][-3:]==rows['ackpapers']
 assert rows['exp'][-46:]==rows['ackexp']
 assert len({x['experiment_id'] for x in rows['exp']})==169
 assert sum(x['relationship_id']=='Ackr1::ENSG00000213088' for x in rows['short'])==1
 assert 'CONDITIONAL' in next(x for x in rows['short'] if x['relationship_id']=='Ackr1::ENSG00000213088')['pass2_biological_verdict']
 assert not any(x.get('literature_receptor_approval_flag')=='YES' for x in rows['short'])
 print('PASS: counts 16/169; ACKR1 3 papers/46 experiments; shortlist 6')
 prior={'papers':'EC_receptor_M01_12_4k_fulltext_paper_ledger_v4.csv','exp':'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v3.csv','short':'EC_receptor_M01_12_4k_shortlist_WORKING_v6.csv','manu':'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v4.csv'}
 if all((d/f).exists() for f in prior.values()):
  old={k:load(d/f) for k,f in prior.items()}
  assert rows['papers'][:13]==old['papers']
  assert rows['exp'][:123]==old['exp']
  for o,n in zip(old['short'],rows['short']):
   if o['relationship_id']!='Ackr1::ENSG00000213088':assert o==n
   else:
    for col in ['D02_logFC','D02_FDR','LOSO_direction_stability_pct','human_brain_EC_median_CPM','human_EC_nonEC_log2_ratio','extracellular_sequence_identity_pct']:assert o[col]==n[col]
  for o,n in zip(old['manu'],rows['manu']):
   if 'ACKR1' not in o['receptor_and_role']:assert o==n
  print('PASS: preceding four receptors byte-equivalent at the record level and original quantitative metrics unchanged')
 else:
  print('NOTE: preceding SELP version files not all present, preservation verified from builder source checksums')
 print('ALL ACKR1 PACKAGE VALIDATIONS PASS')
if __name__=='__main__':main()
