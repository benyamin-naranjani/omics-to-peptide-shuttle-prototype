"""OmiPep v0.2 | Module 01 | Cell 12.4f — export ONE source-linked review packet.

Read-only on all 12.4 / 12.4d / 12.4e inputs; does not adjudicate literature,
select receptors, write to Google Sheets, or run 12.5.

Jupyter:
    OMIPEP_REVIEW_SESSION_ID = 'LIT-P01-01'
    %run -i 12_4f_export_review_microbatch_packet.py

Creates a small zip under data/processed/review_packets/ for review in ChatGPT.
Each new batch must use a different ID; existing packet outputs are never overwritten.
"""
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import hashlib
import json
import re
import pandas as pd

SESSION_ID = str(globals().get('OMIPEP_REVIEW_SESSION_ID', 'LIT-P01-01')).strip()
if not re.fullmatch(r'LIT-[PR]\d{2}-\d{2}', SESSION_ID):
    raise ValueError('Review session ID must look like LIT-P01-01 or LIT-R01-01')

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / 'data' / 'processed'
SOURCES = {
    'dossier': PROCESSED / 'EC_receptor_12_4_full_relationship_docket_v1.csv',
    'expert': PROCESSED / 'EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv',
    'worklist': PROCESSED / 'EC_receptor_12_4d_literature_review_worklist_v2.csv',
    'sessions': PROCESSED / 'EC_receptor_12_4e_literature_microbatch_plan_v1.csv',
    'provenance': PROCESSED / 'EC_receptor_12_4d_schema_and_provenance_v2.json',
}


def require(condition, message):
    if not condition:
        raise AssertionError('FAIL | ' + message)
    print('PASS | ' + message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_checked(path):
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream), [])
    require(header and len(header) == len(set(header)), f'{path.name}: unique header')
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig', low_memory=False)


def matching_col(frame, options, table):
    hits = [x for x in options if x in frame.columns]
    if not hits:
        raise AssertionError(f'{table}: missing one of {options}')
    return hits[0]


print(f'\n=== OmiPep M01 | 12.4f: {SESSION_ID} review packet ===')
require((REPO / 'notebooks').is_dir(), 'OmiPep notebook directory exists')
for label, path in SOURCES.items():
    require(path.is_file() and path.stat().st_size > 0, f'{label} source exists')

provenance = json.loads(SOURCES['provenance'].read_text(encoding='utf-8'))
require(isinstance(provenance, dict), '12.4d provenance parses')
frames = {name: read_checked(path) for name, path in SOURCES.items() if name != 'provenance'}
require([frames[name].shape for name in ('dossier', 'expert', 'worklist')] ==
        [(125, 541), (125, 29), (125, 38)], 'source tables have expected 125-row schemas')

for label, frame in frames.items():
    require('relationship_id' in frame and frame.relationship_id.ne('').all() and
            frame.relationship_id.is_unique and len(frame) == 125,
            f'{label}: 125 unique relationship IDs')

source_ids = set(frames['dossier'].relationship_id)
for label, frame in frames.items():
    require(set(frame.relationship_id) == source_ids, f'{label} preserves all source IDs')

session = frames['sessions']
require('review_microbatch_id' in session, 'microbatch ID field exists')
selected = session.loc[session.review_microbatch_id.eq(SESSION_ID)]
require(1 <= len(selected) <= 5, f'{SESSION_ID} contains 1–5 relationship records')
require(selected.relationship_id.is_unique, 'selected relationship IDs are unique')

selected_ids = selected.relationship_id.tolist()
selected_set = set(selected_ids)
packets = {}
for name, frame in frames.items():
    if name == 'sessions':
        continue
    subset = frame.loc[frame.relationship_id.isin(selected_set)].copy()
    require(len(subset) == len(selected_ids), f'{name} has each selected relationship')
    packets[name] = subset.set_index('relationship_id').loc[selected_ids].reset_index()

ref = packets['dossier']
source_gene = matching_col(ref, ['gene', 'mouse_gene'], 'dossier')
for name, frame in packets.items():
    other_gene = matching_col(frame, ['mouse_gene', 'gene'], name)
    require(frame[other_gene].tolist() == ref[source_gene].tolist(),
            f'{name} mouse gene agrees with immutable source')

# Check orthologue identity when both tables have a comparable identifier.
for name in ('expert', 'worklist'):
    for options in (('human_ensembl_gene_id', 'human_ensembl_ID'),):
        base_key = next((x for x in options if x in ref.columns), None)
        other_key = next((x for x in options if x in packets[name].columns), None)
        if base_key and other_key:
            require(ref[base_key].tolist() == packets[name][other_key].tolist(),
                    f'{name} preserves human Ensembl IDs')

if 'docking_fragment_status' in ref:
    require(ref.docking_fragment_status.eq('not_approved').all(),
            'selected receptor fragments are not approved')
if 'final_selection_status' in ref:
    require(ref.final_selection_status.eq('not_selected').all(),
            'selected receptor relationships remain unselected')

# Show upstream potential literature identifiers without interpreting their scientific validity.
search_tokens = ('doi', 'pmid', 'pubmed', 'literature', 'citation', 'reference',
                 'publication', 'evidence', 'claim', 'transcytosis', 'paper')
literature_cols = [name for name in ref.columns
                   if any(token in name.lower() for token in search_tokens)]
key_cols = list(dict.fromkeys(['relationship_id', source_gene] +
                             [c for c in ('human_gene_symbol', 'human_ensembl_gene_id') if c in ref]))
evidence_cols = list(dict.fromkeys(key_cols + literature_cols))
packets['source_claims'] = ref[evidence_cols].copy()

output_root = PROCESSED / 'review_packets'
folder = output_root / SESSION_ID
archive_path = output_root / f'{SESSION_ID}_source_packet.zip'
require(not folder.exists() and not archive_path.exists(),
        'no existing packet will be overwritten')
output_root.mkdir(parents=True, exist_ok=True)
folder.mkdir()

file_names = {
    'dossier': f'{SESSION_ID}_full_provenance_541_columns.csv',
    'expert': f'{SESSION_ID}_expert_master_29_columns.csv',
    'worklist': f'{SESSION_ID}_literature_worklist_38_columns.csv',
    'source_claims': f'{SESSION_ID}_source_literature_fields.csv',
}
for label, filename in file_names.items():
    packets[label].to_csv(folder / filename, index=False, encoding='utf-8-sig')

manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'module': 'OmiPep v0.2 Module 01', 'cell': '12.4f',
    'microbatch_id': SESSION_ID,
    'relationship_count': len(selected_ids),
    'relationship_ids_ordered': selected_ids,
    'mouse_genes_ordered': ref[source_gene].tolist(),
    'upstream_literature_column_names': literature_cols,
    'source_input_hashes_sha256': {name: digest(path) for name, path in SOURCES.items()},
    'generated_file_hashes_sha256': {label: digest(folder / filename)
                                     for label, filename in file_names.items()},
    'evidence_status': 'UNVERIFIED — source annotations only',
    'notes': [
        'No DOI or paper independently checked; inherited claims are not evidence validation.',
        'Original 541-column dossier remains unchanged.',
        'Never interpret binding or uptake as intact BBB transcytosis.',
        'No Google Sheet edits or receptor approvals performed.',
    ],
}
manifest_path = folder / f'{SESSION_ID}_packet_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
with ZipFile(archive_path, mode='w', compression=ZIP_DEFLATED) as archive:
    for file in sorted(folder.iterdir()):
        archive.write(file, arcname=file.name)

print('\nREVIEW RELATIONSHIPS (review order, NOT scientific rank):')
for _, row in ref.iterrows():
    print('  ', row['relationship_id'], '|', row[source_gene])
print('UPSTREAM LITERATURE-LIKE COLUMNS:', len(literature_cols))
print('CREATED |', archive_path.relative_to(REPO))
print('PASS | Packet contains immutable-source rows, expert preview, worklist and manifest')
print('NOTE | No papers reviewed or Sheet rows written. Upload the ZIP for literature adjudication.')
print('NOTE | 12.5 / ColabFold handoff remain BLOCKED.')
