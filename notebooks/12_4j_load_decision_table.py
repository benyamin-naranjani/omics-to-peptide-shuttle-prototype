"""Read-only notebook loader for the OmiPep M01 Pass-1 summary derivative.
Run from the repository's notebooks directory or root: %run -i 12_4j_load_decision_table.py
No source file is changed and no receptor or docking approval is made.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from IPython.display import display

root = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p/'data/processed/review_outputs').is_dir()), None)
assert root is not None, 'Run from within the OmiPep repository.'
directory = root/'data/processed/review_outputs'
stem = 'EC_receptor_M01_decision_table_12_4j'
report_124j = json.loads((directory/f'{stem}_validation_v1.json').read_text())
for filename, expected in [(report_124j['source_file'], report_124j['source_sha256_before']), (report_124j['decision_file'], report_124j['decision_sha256']), (report_124j['schema_mapping_file'], report_124j['schema_mapping_sha256'])]:
    path = directory/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Hash mismatch: {path}'
source_124j = pd.read_csv(directory/report_124j['source_file'], dtype=str, keep_default_na=False)
decision_124j = pd.read_csv(directory/report_124j['decision_file'], dtype=str, keep_default_na=False)
mapping_124j = pd.read_csv(directory/report_124j['schema_mapping_file'], dtype=str, keep_default_na=False)
assert source_124j.shape == (125,47) and decision_124j.shape == (125,15)
assert source_124j.relationship_id.is_unique and decision_124j.relationship_id.is_unique
assert set(source_124j.relationship_id)==set(decision_124j.relationship_id)
assert source_124j.mouse_gene.nunique()==114
assert list(mapping_124j.source_column)==list(source_124j.columns)
source_by_id = source_124j.set_index('relationship_id').loc[decision_124j.relationship_id]
for target, origin in zip(decision_124j.columns[2:8], source_124j.columns[5:11]):
    assert decision_124j[target].tolist()==source_by_id[origin].tolist(), target
assert source_124j.literature_receptor_approval_flag.eq('NO').all()
assert source_124j.literature_final_expert_adjudication_complete.eq('NO').all()
# Recompute presentation order from source, not from descriptive labels.
fc = pd.to_numeric(source_124j.mouse_D02_logFC,errors='coerce')
fdr = pd.to_numeric(source_124j.mouse_D02_FDR,errors='coerce')
order = source_124j.assign(group=2,cpm=pd.to_numeric(source_124j.human_brain_EC_median_CPM_atlas,errors='coerce'),old=pd.to_numeric(source_124j.DE_review_order_NOT_merit_rank))
order.loc[(fdr<.05)&(fc>0),'group']=0
order.loc[(fdr<.05)&(fc<0),'group']=1
order.loc[fc.isna()|fdr.isna(),'group']=3
order=order.sort_values(['group','cpm','old','relationship_id'],ascending=[True,False,True,True],na_position='last')
assert order.relationship_id.tolist()==decision_124j.relationship_id.tolist()
print('PASS | 125 relationships × 15 columns; 114 mouse genes; exact hashes, quantitative values and sorting verified.')
print('PASS | No receptor approval; claim-level audit and expert selection remain pending. Cell 12.5 remains blocked.')
display(decision_124j)
