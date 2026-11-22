"""OmiPep v0.2 | Module 01 | Cell 12.4h
Read-only scientific coverage audit and complete notebook display.

Run in the existing notebook:
    %run -i 12_4h_audit_and_display_evidence_master.py

Does not write files or change the upstream scientific tables, review packages,
Google Sheet, or scientific selection status.
"""

def _run_124h():
    from pathlib import Path
    import os
    import pandas as pd

    ROOT = Path(os.environ.get('OMIPEP_REPO_ROOT', str(Path(__file__).resolve().parent.parent))).resolve()
    OUTPUTS = ROOT/'data'/'processed'/'review_outputs'
    BASE = ROOT/'data'/'processed'/'EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv'
    FILES = {
     'master':OUTPUTS/'EC_receptor_M01_expert_master_WITH_SCREENED_EVIDENCE_v1.csv',
     'ledger':OUTPUTS/'EC_receptor_M01_evidence_claim_ledger_v1.csv',
     'references':OUTPUTS/'EC_receptor_M01_reference_index_v1.csv',
     'reviews':OUTPUTS/'EC_receptor_M01_relationship_review_summaries_v1.csv',
    }
    for name,path in {'baseline':BASE,**FILES}.items():
        if not path.is_file():
            raise FileNotFoundError(f'12.4h needs {name}: {path}')
    read=lambda p:pd.read_csv(p,dtype=str,keep_default_na=False,encoding='utf-8-sig',low_memory=False)
    baseline=read(BASE)
    master=read(FILES['master']); claims=read(FILES['ledger'])
    refs=read(FILES['references']); reviews=read(FILES['reviews'])
    assert baseline.shape==(125,29),baseline.shape
    assert master.shape==(125,40),master.shape
    assert len(claims)>=1 and len(refs)>=1 and len(reviews)>=1
    assert master.relationship_id.is_unique and baseline.relationship_id.is_unique
    assert list(master.columns[:29])==list(baseline.columns)
    assert master[baseline.columns.tolist()].equals(baseline)
    assert set(master.relationship_id)==set(baseline.relationship_id)
    assert claims.claim_id.is_unique and refs.citation_key.is_unique and reviews.relationship_id.is_unique
    assert set(claims.relationship_id).issubset(master.relationship_id)
    assert set(claims.citation_key).issubset(refs.citation_key)
    assert set(reviews.relationship_id).issubset(master.relationship_id)
    assert set(master.literature_screened_primary_source_keys.str.len().loc[
     master.literature_screening_review_state.eq('')])=={0}

    expected_n=master.literature_screening_review_state.ne('').sum()
    assert expected_n==len(reviews)

    # Summarize status without silently turning untested endpoints into negative evidence.
    fulltext_count=int(refs.source_read_depth.str.contains('full text inspected',case=False,na=False).sum()) if 'source_read_depth' in refs else 0
    only_part_count=len(refs)-fulltext_count
    screens_not_final=int(claims.review_status.str.contains('screened_not_final',case=False,na=False).sum())
    untested_transport=int(claims.intact_BBB_transcytosis_test_status.str.startswith('not_tested').sum())
    claimed_verified=int(refs.verification_status.str.contains('full_text_adjudicated',case=False,na=False).sum()) if 'verification_status' in refs else 0
    multi_claim_papers=claims.citation_key.value_counts().loc[lambda x:x>1].shape[0]

    print('\n=== OmiPep M01 | 12.4h: literature evidence coverage audit ===')
    for name,obj in [('Master',master),('References',refs),('Paper-relationship claims',claims),('Review summaries',reviews)]:
     print(f'{name:>26}: {obj.shape[0]} rows x {obj.shape[1]} columns')
    print(f'Original 29 master columns: unchanged | derived columns: {master.shape[1]-baseline.shape[1]}')
    print(f'Relationships screened: {expected_n}/125 | unscreened: {125-expected_n}/125')
    print(f'Publications tagged full text inspected: {fulltext_count}/{len(refs)}')
    print(f'Publications abstract/excerpt/partial: {only_part_count}/{len(refs)}')
    print(f'Claims marked screened but not finalized: {screens_not_final}/{len(claims)}')
    print(f'Claims with intact BBB test not performed/applicable: {untested_transport}/{len(claims)}')
    print(f'Publications explicitly tagged full-text adjudicated: {claimed_verified}')
    print(f'Papers with >1 claim record: {multi_claim_papers} (one claim per paper may oversummarize endpoints)')

    # Show original transcriptomics and context jointly with the newly screened evidence.
    view_cols=['DE_review_order_NOT_merit_rank','mouse_gene','human_gene',
    'mouse_D02_logFC','mouse_D02_FDR','human_brain_EC_median_CPM_atlas',
    'human_EC_nonEC_log2_ratio_atlas','mouse_human_extracellular_identity_pct',
    'mouse_protein_topology_context','HPA_peripheral_bulk_RNA_max_nTPM',
    'HPA_peripheral_bulk_peak_tissue','HPA_peripheral_EC_RNA_max_nTPM',
    'HPA_peripheral_EC_peak_organ_cluster','HPA_immune_RNA_max_nTPM',
    'literature_n_primary_sources_screened_NOT_verified',
    'literature_screened_native_luminal_access_summary',
    'literature_screened_binding_internalization_summary',
    'literature_screened_BBB_transport_summary',
    'literature_screened_abluminal_cargo_summary',
    'literature_screened_uncertainties_and_next_tests']
    assert set(view_cols).issubset(master.columns)
    reviewed_view=master.loc[master.literature_screening_review_state.ne(''),view_cols].copy()
    reviewed_view=reviewed_view.sort_values('DE_review_order_NOT_merit_rank',key=lambda x:pd.to_numeric(x,errors='raise'))

    # Demonstrate exactly where unknowns arise.
    completeness=master[['relationship_id','mouse_gene','human_gene',
     'literature_screening_review_state','literature_n_primary_sources_screened_NOT_verified',
     'literature_screened_native_luminal_access_summary',
     'literature_screened_BBB_transport_summary']].copy()
    completeness['data_state']=completeness.literature_screening_review_state.map(
     lambda s:'screened_not_adjudicated' if s else 'literature_not_yet_screened')

    try:
     from IPython.display import display, Markdown
     display(Markdown('### Entire 125-row / 40-column master (all rows, no filtering)'))
     with pd.option_context('display.max_rows', 130,'display.max_columns', 45, 'display.max_colwidth',65):
      display(master)
     display(Markdown('### Five screened relationships: omics + peripheral + literature side by side'))
     with pd.option_context('display.max_columns', len(view_cols), 'display.max_colwidth',100):
      display(reviewed_view)
     display(Markdown('### All paper–relationship claim records (not yet final adjudication)'))
     display_cols=['relationship_id','citation_key','endpoint','evidence_class',
                   'species_model','ligand_or_agent','experimental_observation',
                   'applicability_limitation','intact_BBB_transcytosis_test_status','review_status']
     assert set(display_cols).issubset(claims.columns)
     with pd.option_context('display.max_rows', max(30,len(claims)),'display.max_colwidth',100):
      display(claims[display_cols])
     display(Markdown('### Literature review coverage by relationship'))
     display(completeness.loc[completeness.data_state.eq('screened_not_adjudicated')])
    except ImportError:
     print('\nFirst five evidence review rows:\n',reviewed_view.to_string(index=False))

    # Reusable notebook variables, no emitted / changed source outputs.
    print('\nAUDIT INTERPRETATION: screening data integrated. Full-text/figure-level and expert review pending. No final ranking or docking approval.')
    return master, claims, refs, reviewed_view

OMIPEP_M01_MASTER_ALL_124h, OMIPEP_M01_CLAIM_LEDGER_124h, OMIPEP_M01_REFERENCES_124h, OMIPEP_M01_SCREENED_FIVE_124h = _run_124h()
