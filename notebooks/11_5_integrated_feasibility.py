# 11.5 — Integrated biological and structural feasibility (OmiPep v0.2)
# Run via %run -i after 11.4. No ranking, docking, or fragment approval.
import hashlib, json, re, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

FEASIBILITY_INTEGRATION_VERSION = '1.5'
EPMC_ENDPOINT = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
EPMC_CACHE = Path(human_metadata_dir) / 'literature_metadata_cache'
EPMC_CACHE.mkdir(parents=True, exist_ok=True)

# A. Bibliographic identity audit: metadata match is not endpoint verification.
def fi_source_query(source_id):
    if re.fullmatch(r'PMID:\d+', source_id):
        return 'pmid', source_id[5:], f'EXT_ID:{source_id[5:]} AND SRC:MED'
    if source_id.startswith('DOI:'):
        doi = source_id[4:].strip().lower()
        return 'doi', doi, f'DOI:"{doi}"'
    match = re.search(r'/articles/(PMC\d+)(?:/|$)', source_id, re.I)
    if match:
        pmcid = match.group(1).upper()
        return 'pmcid', pmcid, f'PMCID:{pmcid}'
    return None, None, None


def fi_exact_match(record, kind, value):
    if kind == 'pmid':
        return (str(record.get('pmid', '')) == value or
                (record.get('source') == 'MED' and str(record.get('id')) == value))
    if kind == 'doi':
        return str(record.get('doi', '')).lower() == value
    if kind == 'pmcid':
        return str(record.get('pmcid', '')).upper() == value or (
            record.get('source') == 'PMC' and str(record.get('id', '')).upper() == value)
    return False


def fi_fetch(query):
    key = hashlib.sha256(query.encode()).hexdigest()
    path = EPMC_CACHE / f'{key}.json'
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        assert cached['query'] == query
        assert hashlib.sha256(json.dumps(cached['payload'], sort_keys=True).encode()).hexdigest() == cached['sha256']
        return cached
    for attempt in range(3):
        try:
            response = requests.get(EPMC_ENDPOINT, params={'query': query, 'format': 'json', 'resultType': 'core', 'pageSize': 100}, timeout=(10, 35))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get('resultList', {}).get('result'), list):
                raise ValueError('Unexpected Europe PMC response')
            record = {'query': query, 'url': response.url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'payload': payload,
                      'sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8'); tmp.replace(path)
            time.sleep(0.25)
            return record
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
            if attempt < 2: time.sleep(2 ** attempt)
    return {'error': error, 'query': query}


def fi_resolve_metadata(source_id, payload):
    kind, value, _ = fi_source_query(source_id)
    if kind is None:
        return {'bibliographic_status': 'non_publication_source_manual_review'}
    records = payload['resultList']['result']
    matches = [r for r in records if fi_exact_match(r, kind, value)]
    if int(payload.get('hitCount', len(records))) > len(records):
        return {'bibliographic_status': 'incomplete_search_response'}
    if not matches:
        return {'bibliographic_status': 'no_exact_identifier_match'}
    # PubMed and PMC may both represent the same article. Do not merge conflicts.
    pmids = {str(r['pmid']) for r in matches if r.get('pmid')}
    dois = {str(r['doi']).lower() for r in matches if r.get('doi')}
    if len(pmids) > 1 or len(dois) > 1:
        return {'bibliographic_status': 'conflicting_exact_records'}
    best = max(matches, key=lambda r: (bool(r.get('abstractText')), r.get('source') == 'MED', bool(r.get('title'))))
    return {'bibliographic_status': 'identifier_matched', 'resolved_pmid': next(iter(pmids), value if kind == 'pmid' else None),
            'resolved_doi': next(iter(dois), value if kind == 'doi' else None), 'resolved_pmcid': best.get('pmcid', value if kind == 'pmcid' else None),
            'resolved_title': best.get('title'), 'resolved_year': best.get('pubYear'), 'resolved_journal': best.get('journalTitle'),
            'abstract_available': bool(best.get('abstractText')), 'publication_types_json': json.dumps(best.get('pubTypeList', {})),
            'retraction_metadata_json': json.dumps({k: best[k] for k in ('hasRetraction', 'isRetracted', 'pubTypeList') if k in best}),
            'n_exact_records': len(matches)}

assert fi_source_query('PMID:123')[2] == 'EXT_ID:123 AND SRC:MED'
assert fi_source_query('URL:https://pmc.ncbi.nlm.nih.gov/articles/PMC123/')[1] == 'PMC123'
assert fi_resolve_metadata('PMID:123', {'resultList': {'result': [{'source':'MED','id':'123','title':'Test','pubYear':'2024'}]}, 'hitCount':1})['bibliographic_status'] == 'identifier_matched'
assert fi_resolve_metadata('PMID:123', {'resultList': {'result': [{'id':'123','source':'PMC'}]}, 'hitCount':1})['bibliographic_status'] == 'no_exact_identifier_match'

fi_literature = receptor_literature_evidence_v12.copy()
assert not fi_literature.duplicated(['gene','source_id','evidence_endpoint']).any()
fi_sources = []
for i, source in enumerate(sorted(fi_literature.source_id.unique()), 1):
    kind, value, query = fi_source_query(source)
    row = {'source_id': source, 'query': query}
    if query is None:
        row['bibliographic_status'] = 'non_publication_source_manual_review'
    else:
        result = fi_fetch(query)
        if 'error' in result:
            row.update(bibliographic_status='retrieval_failed', error=result['error'])
        else:
            row.update(fi_resolve_metadata(source, result['payload']))
            row.update(retrieved_utc=result['retrieved_utc'], payload_sha256=result['sha256'], source_query_url=result['url'])
    fi_sources.append(row)
    if i % 10 == 0: print(f'Bibliographic metadata: {i} sources', flush=True)
receptor_bibliographic_audit = pd.DataFrame(fi_sources)
assert receptor_bibliographic_audit.source_id.is_unique
for c in ['resolved_pmid','resolved_doi','resolved_title','resolved_year','resolved_pmcid','publication_types_json']:
    if c not in receptor_bibliographic_audit: receptor_bibliographic_audit[c] = pd.NA
receptor_literature_endpoint_review = fi_literature.merge(receptor_bibliographic_audit,on='source_id',validate='many_to_one')
expected_year = pd.to_numeric(receptor_literature_endpoint_review.publication_year,errors='coerce')
resolved_year = pd.to_numeric(receptor_literature_endpoint_review.resolved_year,errors='coerce')
receptor_literature_endpoint_review['publication_year_discrepancy'] = (expected_year.ne(resolved_year) & expected_year.notna() & resolved_year.notna())
receptor_literature_endpoint_review['endpoint_verification_status'] = 'manual_source_endpoint_reconciliation_required'
# Explicitly preserve previous study claims as provisional; a matching identifier proves nothing about the claim.
# No abstract is copied into the manuscript evidence table; cached metadata contain the source abstract for review.

# B. Reconfirm exact integrated universe and structural tables.
fi_base = receptor_feasibility_evidence_v14.copy()
assert not fi_base.duplicated(feasibility_pair_keys).any()
fi_scope = fi_base.loc[fi_base.structural_assessment_scope].copy()
assert set(fi_scope.gene) == set(structural_scope_genes)
assert receptor_structural_region_worklist.region_id.is_unique
assert not receptor_structural_protein_audit.duplicated(['species','identity']).any()
assert set(receptor_structural_region_worklist.region_status) == {'topology_interval_not_approved_model_fragment'}
assert set(fi_scope.human_ensembl_gene_id.dropna()).issubset(set(hpa_systemic_gene_evidence.human_ensembl_gene_id))

# C. Summarize topology review flags per protein (never assume a flagged interval is a valid domain).
def fi_list(value):
    if isinstance(value, list): return value
    if pd.isna(value): return []
    value = json.loads(value)
    assert isinstance(value, list)
    return value

fi_region = receptor_structural_region_worklist.copy()
fi_region['cross_boundary_disulfide'] = fi_region.paired_disulfides_crossing_region_json.map(lambda x: bool(fi_list(x)))
fi_region['single_endpoint_disulfide'] = fi_region.single_endpoint_disulfides_json.map(lambda x: bool(fi_list(x)))
fi_region['full_PDBe_span'] = fi_region.max_single_PDBe_span_coverage_pct.eq(100)
fi_region['full_exact_AFDB_span'] = fi_region.max_single_exact_AFDB_span_coverage_pct.eq(100)
fi_region['short_interval'] = fi_region.review_flags_json.map(lambda x: 'short_region_context_review' in fi_list(x))
fi_region['membrane_adjacent'] = fi_region.review_flags_json.map(lambda x: 'sequence_adjacent_to_membrane_feature' in fi_list(x))
fi_region['glycan_review'] = fi_region.review_flags_json.map(lambda x: 'glycan_context_review' in fi_list(x))
fi_region['contains_inference'] = fi_region.n_inferred_residues.gt(0)
fi_region_summary = fi_region.groupby(['species','identity'],as_index=False).agg(
    n_intervals=('region_id','size'), n_short_intervals=('short_interval','sum'),
    n_membrane_adjacent_intervals=('membrane_adjacent','sum'), n_inferred_intervals=('contains_inference','sum'),
    n_intervals_crossing_disulfides=('cross_boundary_disulfide','sum'), n_intervals_single_endpoint_disulfides=('single_endpoint_disulfide','sum'),
    n_intervals_glycan_review=('glycan_review','sum'), n_intervals_full_PDBe_span=('full_PDBe_span','sum'),
    n_intervals_full_exact_AFDB_span=('full_exact_AFDB_span','sum'))
# Span sums count observed spans, not proof of missing structural templates.
for species, key in [('mouse','gene'),('human','human_ensembl_gene_id')]:
    sr = fi_region_summary.loc[fi_region_summary.species.eq(species)].drop(columns='species').rename(columns={'identity':key})
    sr = sr.rename(columns={c:f'{species}_review_{c}' for c in sr.columns if c != key})
    fi_scope = fi_scope.merge(sr,on=key,how='left',validate='many_to_one')

# D. Keep distinct evidence dimensions and unresolved states.
# These annotations are navigation aids, not a reward function or a viability filter.
fi_surface_classes = {'surface_accessible','surface_mature_chain_supported'}
fi_context_classes = {'surface_isoform_dependent','surface_topology_unresolved','membrane_external_topology','membrane_unknown_location','mixed_or_context_dependent','mixed_localization'}
fi_negative_context = {'organelle_context_requires_surface_evidence','predominantly_intracellular_in_reviewed_human_model'}
fi_scope['accessibility_review_context'] = np.select([
    fi_scope.biological_context_flag.isin(fi_negative_context),
    fi_scope.biological_context_flag.eq('context_dependent_surface_pool_ER_dominant'),
    fi_scope.mouse_surface_class.isin(fi_surface_classes),
    fi_scope.mouse_surface_class.isin(fi_context_classes),
    fi_scope.mouse_surface_class.eq('intracellular_or_organelle')],
    ['organelle_or_intracellular_context','context_dependent_surface_pool','database_surface_evidence','surface_or_membrane_context_unresolved','intracellular_annotation'],
    default='other_or_unresolved_localisation')
fi_scope['both_species_topology_available'] = (fi_scope.mouse_structure_structural_review_status.eq('topology_intervals_available') & fi_scope.human_structure_structural_review_status.eq('topology_intervals_available'))
fi_scope['human_template_context'] = np.select([
    fi_scope.human_structure_n_regions_with_full_single_PDBe_span.gt(0),
    fi_scope.human_structure_n_regions_with_full_single_exact_AFDB_span.gt(0),
    fi_scope.human_structure_n_PDBe_valid_span_records.gt(0) | fi_scope.human_structure_n_AFDB_exact_sequence_models.gt(0)],
    ['at_least_one_full_experimental_mapping_span','at_least_one_full_exact_predicted_span','partial_or_other_template_inventory'],default='no_suitable_span_established')
fi_scope['human_orthology_review_context'] = np.select([
    fi_scope.orthology_type.eq('ortholog_one2one') & pd.to_numeric(fi_scope.orthology_confidence,errors='coerce').eq(1),
    fi_scope.supporting_sources.fillna('').str.contains('MGI',regex=False) & fi_scope.supporting_sources.fillna('').str.contains('NCBI',regex=False),
    fi_scope.supporting_sources.fillna('').str.contains(r'\+')],
    ['Ensembl_high_confidence_one_to_one','MGI_and_NCBI_reported_review_relationship','multiple_reporting_sources_review_relationship'],default='single_source_or_unresolved_review_relationship')
fi_scope['literature_primary_profile_available'] = fi_scope.literature_review_status.isin(['targeted_primary_evidence_reviewed','extended_targeted_primary_evidence'])
fi_scope['stroke_upregulation_supported'] = ((fi_scope.D02_vs_Sham_logFC.gt(0) & fi_scope.D02_vs_Sham_FDR.lt(CANDIDATE_FDR)) | (fi_scope.D14_vs_Sham_logFC.gt(0) & fi_scope.D14_vs_Sham_FDR.lt(CANDIDATE_FDR)))
# Gene-level study claims remain linked to actual species / ligand in the evidence table.
fi_scope['delivery_review_context'] = fi_scope.literature_delivery_category.fillna('not_yet_reviewed')
fi_scope['systemic_context_review_required'] = True
fi_scope['luminal_accessibility_validated_for_planned_ligand'] = pd.NA
fi_scope['fragment_approved'] = False
fi_scope['peptide_transport_validated'] = False

# E. Explicitly expose remaining review tasks. Include ALL structural-scope relationships.
# Use the same relationship for all evidence; do not combine maxima across orthologues.
fi_scope['review_lane'] = np.select([
    fi_scope.is_data_driven_candidate & fi_scope.stroke_upregulation_supported & fi_scope.accessibility_review_context.isin(['database_surface_evidence','surface_or_membrane_context_unresolved','context_dependent_surface_pool']),
    fi_scope.is_data_driven_candidate & fi_scope.stroke_upregulation_supported,
    fi_scope.is_curated_reference],
    ['stroke_candidate_surface_or_context_review','stroke_candidate_localisation_review','reference_review'],default='other_context_review')
# This is a worklist, not an exclusion gate. Missing evidence is displayed independently.
fi_scope['additional_primary_review_required'] = ~fi_scope.literature_primary_profile_available
fi_scope['source_endpoint_reconciliation_required'] = True
fi_scope['manual_coordinate_review_required'] = True
fi_scope['systemic_blood_cell_and_soluble_target_review_required'] = True

# F. HPA brain EC context if represented; no cross-resource ratios or risk scores.
for c in ['HPA_brain_EC_cluster_n_observed','HPA_brain_EC_cluster_max_nTPM','HPA_peripheral_EC_cluster_n_observed','HPA_peripheral_EC_cluster_max_nTPM']:
    assert c in fi_scope
fi_scope['HPA_brain_and_peripheral_EC_context_available'] = fi_scope.HPA_brain_EC_cluster_n_observed.gt(0) & fi_scope.HPA_peripheral_EC_cluster_n_observed.gt(0)

# G. Bibliographic coverage at receptor level, not a measure of claim truth.
fi_bib = receptor_literature_endpoint_review.assign(_matched=lambda x:x.bibliographic_status.eq('identifier_matched'),_publication=lambda x:x.study_type.isin(['primary','review']))
fi_bib_sources = fi_bib[['gene','source_id','_matched','_publication']].drop_duplicates()
fi_bib_counts = fi_bib_sources.groupby('gene',as_index=False).agg(n_curated_source_identifiers=('source_id','size'), n_publication_identifiers=('_publication','sum'),n_matched_publication_identifiers=('_matched','sum'))
fi_scope = fi_scope.merge(fi_bib_counts,on='gene',how='left',validate='many_to_one')
for c in ['n_curated_source_identifiers','n_publication_identifiers','n_matched_publication_identifiers']:
    fi_scope[c] = fi_scope[c].fillna(0).astype(int)
assert (fi_scope.n_matched_publication_identifiers <= fi_scope.n_publication_identifiers).all()

# H. Create full 586-row integration plus complete decision packet.
fi_extra_cols = [c for c in fi_scope.columns if c not in fi_base.columns]
receptor_feasibility_evidence_v15 = fi_base.merge(fi_scope[feasibility_pair_keys + fi_extra_cols],on=feasibility_pair_keys,how='left',validate='one_to_one')
assert len(receptor_feasibility_evidence_v15) == len(fi_base)
pd.testing.assert_frame_equal(fi_base.reset_index(drop=True),receptor_feasibility_evidence_v15[fi_base.columns].reset_index(drop=True),check_dtype=False)
receptor_integrated_feasibility_review = receptor_feasibility_evidence_v15.loc[receptor_feasibility_evidence_v15.structural_assessment_scope].copy()
assert len(receptor_integrated_feasibility_review) == len(fi_scope)
assert set(receptor_integrated_feasibility_review.gene) == set(structural_scope_genes)
assert not receptor_integrated_feasibility_review.fragment_approved.any()

# Review packet views retain all relationships, not a top-N.
fi_compact_cols = ['gene','human_gene_symbol','human_ensembl_gene_id','is_data_driven_candidate','is_curated_reference','stroke_upregulation_supported','D02_vs_Sham_logFC','D02_vs_Sham_FDR','D14_vs_Sham_logFC','D14_vs_Sham_FDR',
'mouse_surface_class','accessibility_review_context','human_orthology_review_context','supporting_sources','external_identity_pct','external_mouse_joint_coverage_pct','external_human_joint_coverage_pct','unsorted_median_CPM','EC_vs_non_EC_median_paired_log2_ratio','n_non_EC_comparators_robust_EC_higher',
'HPA_peripheral_EC_cluster_max_nTPM','HPA_peripheral_EC_cluster_peak_context','HPA_brain_EC_cluster_max_nTPM','HPA_brain_EC_cluster_n_observed','HPA_immune_cell_max_nTPM','HPA_IHC_max_level','human_template_context','human_structure_n_exact_TM_intervals','human_structure_n_exact_GPI_annotations','human_review_n_intervals_crossing_disulfides','human_review_n_membrane_adjacent_intervals',
'literature_review_status','delivery_review_context','BBB_transcytosis_evidence','physiological_function_risk','n_matched_publication_identifiers','n_publication_identifiers','review_lane','additional_primary_review_required','fragment_approved']
fi_compact = receptor_integrated_feasibility_review[fi_compact_cols].sort_values(['review_lane','gene','human_ensembl_gene_id'])
fi_pending = fi_compact.loc[fi_compact.additional_primary_review_required].copy()
fi_pending_surface = fi_pending.loc[fi_pending.review_lane.eq('stroke_candidate_surface_or_context_review')].copy()
fi_gene_lanes = fi_compact[['gene','review_lane']].drop_duplicates()
assert fi_gene_lanes.gene.is_unique

# I. Exports (independent of prior cell state; reruns start from immutable v1.4).
fi_tables = {'EC_receptor_bibliographic_audit':receptor_bibliographic_audit,'EC_receptor_literature_endpoint_review':receptor_literature_endpoint_review,'EC_structural_interval_review_summary':fi_region_summary,'EC_integrated_receptor_feasibility_evidence_v1_5':receptor_feasibility_evidence_v15,'EC_integrated_feasibility_review':receptor_integrated_feasibility_review,'EC_integrated_feasibility_compact':fi_compact,'EC_remaining_primary_literature_review':fi_pending,'EC_remaining_surface_candidate_review':fi_pending_surface}
for name,table in fi_tables.items(): table.to_csv(Path(out_dir)/f'{name}.csv',index=False)
fi_manifest = {'version':FEASIBILITY_INTEGRATION_VERSION,'generated_utc':datetime.now(timezone.utc).isoformat(),'bibliographic_api':EPMC_ENDPOINT,'n_source_identifiers':len(receptor_bibliographic_audit),'n_mouse_genes_in_scope':fi_compact.gene.nunique(),'n_relationships_in_scope':len(fi_compact),'n_pending_surface_candidate_genes':fi_pending_surface.gene.nunique(),'review_lanes':'Navigation only; no exclusion, final ranking or numerical aggregation.','limitations':['Bibliographic identity matching is not claim-level verification or retraction clearance.','Primary literature profiles are targeted and incomplete; negative results and unresolved endpoints remain provisional.','Predicted structure matches and experimental mapping spans are not resolved coordinate coverage or approved fragments.','No cross-resource RNA ratios; HPA cluster maxima do not establish systemic targeting specificity.','Cell-type RNA context does not cover all circulating or soluble target pools.','No delivery validation for generated peptides.']}
(Path(out_dir)/'EC_integrated_feasibility_manifest_v1_5.json').write_text(json.dumps(fi_manifest,indent=2),encoding='utf-8')
receptor_feasibility_evidence = receptor_feasibility_evidence_v15
feasibility = receptor_feasibility_evidence
receptor_biological_review_worklist = receptor_integrated_feasibility_review

print('\nBibliographic source status:'); display(receptor_bibliographic_audit.bibliographic_status.value_counts())
print('\nReference identity/year exceptions (not endpoint validation):')
display(receptor_literature_endpoint_review.loc[receptor_literature_endpoint_review.publication_year_discrepancy | (~receptor_literature_endpoint_review.bibliographic_status.eq('identifier_matched') & receptor_literature_endpoint_review.study_type.isin(['primary','review'])), ['gene','source_id','publication_year','resolved_year','resolved_title','bibliographic_status']].drop_duplicates())
print('\nReview lanes (unique mouse genes):'); display(fi_gene_lanes.review_lane.value_counts())
print('\nHPA brain and peripheral EC context available (relationship rows):'); display(fi_scope.HPA_brain_and_peripheral_EC_context_available.value_counts())
print('\nRemaining surface/context stroke candidate review (all relationships):')
display(fi_pending_surface[['gene','human_gene_symbol','mouse_surface_class','D02_vs_Sham_logFC','D02_vs_Sham_FDR','D14_vs_Sham_logFC','D14_vs_Sham_FDR','external_identity_pct','unsorted_median_CPM','EC_vs_non_EC_median_paired_log2_ratio','human_template_context','literature_review_status']].round(4))
print('\nSelected feasibility context (illustrative, not ranking):')
fi_selected = fi_compact.loc[fi_compact.gene.isin(['Procr','Tm4sf1','Ecscr','Ace','Ackr1','Tnfrsf1a','Cd93','Itm2a','Ly6e','Tfrc','Lrp1','Lrp8','Insr','Igf1r','Itgb1','Cd200','Slc3a2'])]
display(fi_selected[['gene','human_gene_symbol','stroke_upregulation_supported','accessibility_review_context','external_identity_pct','HPA_peripheral_EC_cluster_max_nTPM','human_template_context','delivery_review_context','n_matched_publication_identifiers','n_publication_identifiers']].round(3))
print('\nIntegrated genes:',fi_base.gene.nunique(),'| relationship/context rows:',len(fi_base),'| structural scope:',len(fi_scope))
print('Pending primary review genes:',fi_pending.gene.nunique(),'| surface/context stroke candidate genes:',fi_pending_surface.gene.nunique())
print('Integrated feasibility packet exported. No fragment approved and no final ranking performed.')
