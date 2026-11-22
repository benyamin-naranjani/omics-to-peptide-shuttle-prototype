#!/usr/bin/env python3
"""Non-mutating validation of OmiPep M01 LRP1 installation package."""
import sys,csv,json,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def rows(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 d=Path(sys.argv[1] if len(sys.argv)>1 else 'data/processed/review_outputs')
 rep=json.loads((d/'EC_receptor_M01_12_4k_LRP1_fulltext_validation_v1.json').read_text())
 for name,hash_expected in rep['generated_csv_sha256'].items():
  p=d/name;assert p.exists() and sha(p)==hash_expected,('CSV integrity fail',p)
  print('PASS CSV',name)
 src=d.parent/'review_sources'/'LRP1'
 for k,v in rep['original_source_pdf_sha256'].items():
  p=src/v['source_filename'];assert p.exists() and sha(p)==v['sha256'],('PDF integrity fail',p)
  print('PASS PDF',k)
 p=rows(d/'EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv')
 e=rows(d/'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v5.csv')
 s=rows(d/'EC_receptor_M01_12_4k_shortlist_WORKING_v8.csv')
 m=rows(d/'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v6.csv')
 b=rows(d/'EC_receptor_M01_12_4k_LRP1_benchmark_peptides_v1.csv')
 x=rows(d/'EC_receptor_M01_12_4k_LRP1_experiment_ledger_v1.csv')
 y=rows(d/'EC_receptor_M01_12_4k_LRP1_fulltext_papers_v1.csv')
 assert (len(p),len(e),len(s),len(m),len(b),len(x),len(y))==(19,213,6,6,3,44,3)
 assert p[-3:]==y and e[-44:]==x
 assert len(set(i['experiment_id'] for i in e))==213
 assert all(len(i['sequence_unmodified_L_aa'])==int(i['length_aa']) for i in b)
 assert {i['benchmark_peptide'] for i in b}=={'Angiopep-2','Angiopep-7','L57'}
 assert sum(i['relationship_id']=='Lrp1::ENSG00000123384' for i in s)==1
 assert next(i for i in s if i['relationship_id']=='Lrp1::ENSG00000123384')['candidate_role']=='Benchmark only'
 prev={'papers':'EC_receptor_M01_12_4k_fulltext_paper_ledger_v5.csv','experiments':'EC_receptor_M01_12_4k_cumulative_experiment_ledger_v4.csv','shortlist':'EC_receptor_M01_12_4k_shortlist_WORKING_v7.csv','manuscript':'EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v5.csv'}
 if all((d/n).exists() for n in prev.values()):
  p0,e0,s0,m0=[rows(d/prev[k]) for k in ['papers','experiments','shortlist','manuscript']]
  assert p[:16]==p0 and e[:169]==e0
  for a,c in zip(s0,s):
   if a['relationship_id']!='Lrp1::ENSG00000123384':assert a==c
   else:
    for n in ['candidate_role','relationship_id','mouse_human_gene_pair','D02_logFC','D02_FDR','LOSO_direction_stability_pct','human_brain_EC_median_CPM','human_EC_nonEC_log2_ratio','extracellular_sequence_identity_pct']:
     assert a[n]==c[n]
  for a,c in zip(m0,m):
   if 'LRP1' not in a['receptor_and_role']:assert a==c
  print('PASS preservation previous five reviews and all original metrics')
 print('ALL LRP1 PACKAGE VALIDATION PASSED (19 paper rows / 213 experimental records / 6 receptors)')
if __name__=='__main__':main()
