"""OmiPep v0.2 | Notebook cell 12.1: receptor selection evidence matrix.

Use after 11.5b with: %run -i 12_1_receptor_prioritization_framework.py
This is PRE-prioritisation, not selection. No receptor scores, peptide results,
transport claims, approved fragments, or automatic exclusions are produced.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SELECTION_FRAMEWORK_VERSION = "1.0"
_selection_out = Path(out_dir)
_selection_out.mkdir(parents=True, exist_ok=True)

# 1. Frozen, relationship-specific base. No mutation of upstream evidence.
_base = receptor_feasibility_evidence_v16.copy(deep=True)
_pair = ["gene", "human_ensembl_gene_id"]
_required = {
    *_pair, "structural_assessment_scope", "review_lane",
    "is_data_driven_candidate", "is_curated_reference",
    "D02_vs_Sham_logFC", "D02_vs_Sham_FDR",
    "D14_vs_Sham_logFC", "D14_vs_Sham_FDR",
    "mouse_surface_class", "accessibility_review_context",
    "human_orthology_review_context", "human_template_context",
    "human_structure_structural_review_status",
    "mouse_structure_structural_review_status",
    "literature_review_status", "delivery_review_context",
    "BBB_transcytosis_evidence", "human_ec_detection_class",
    "unsorted_n_assessed_donors", "unsorted_median_CPM",
    "EC_vs_non_EC_comparison_status", "EC_vs_non_EC_median_paired_log2_ratio",
    "HPA_peripheral_EC_cluster_n_observed", "HPA_peripheral_EC_cluster_max_nTPM",
    "HPA_immune_cell_max_nTPM", "HPA_audited_brain_EC_cluster_n_observed",
    "additional_primary_review_required", "additional_targeted_primary_source_review_available",
    "n_directly_reviewed_PMIDs", "fragment_approved", "endpoint_reconciliation_complete",
}
_missing = sorted(_required - set(_base.columns))
if _missing:
    raise RuntimeError(f"11.5b prerequisite is missing columns: {_missing}")
assert not _base.duplicated(_pair).any(), "Gene-human relationship must remain unique"
assert _base.gene.nunique() == len(set(_base.gene))
assert set(_base.loc[_base.structural_assessment_scope, "gene"]) == set(structural_scope_genes)
assert not _base.loc[_base.structural_assessment_scope, "fragment_approved"].fillna(False).any()

_scope = _base.loc[_base.structural_assessment_scope].copy()
assert _scope.gene.nunique() == len(set(structural_scope_genes))
assert not _scope.duplicated(_pair).any()

# Explicitly distinguish an unresolved orthology relationship from a measured zero.
def _positive(value):
    return bool(pd.notna(value) and float(value) > 0)

def _true(value):
    return bool(pd.notna(value) and value == True)

def _txt(value, fallback="unassessed"):
    return fallback if pd.isna(value) or str(value).strip() == "" else str(value)

# Source-specific evidence types. These are FORMS OF EVIDENCE, not predictions of peptide transport.
_DELIVERY_FAMILIES = {
    "peptide_delivery_reference": "published_peptide_delivery_reference",
    "established_delivery_reference": "published_antibody_delivery_reference",
    "experimental_delivery_reference": "experimental_antibody_model_reference",
    "native_ligand_delivery_reference": "published_native_ligand_model",
    "native_ligand_transport_candidate": "published_native_ligand_model",
    "viral_vector_delivery_reference": "viral_vector_only_reference",
    "secondary_literature_delivery_reference": "secondary_literature_only",
    "reverse_transport_candidate": "reverse_direction_transport_context",
    "exploratory_with_negative_transport_evidence": "negative_or_inconclusive_ligand_test",
    "endothelial_uptake_candidate": "internalisation_without_BBB_delivery",
    "endothelial_recycling_candidate": "recycling_without_BBB_delivery",
    "endothelial_surface_signalling_candidate": "signalling_without_BBB_delivery",
    "barrier_function_candidate": "barrier_integrity_perturbation_context",
}
_INTRACELLULAR_CONTEXT = {
    "organelle_context_candidate", "organelle_with_context_surface_evidence",
    "predominantly_intracellular_candidate",
}

# 2. Full audit fields are retained. No fixed nTPM/CPM identity threshold is imposed.
DISCOVERY_FDR_THRESHOLD = float(globals().get("CANDIDATE_FDR", 0.05))
assert 0 < DISCOVERY_FDR_THRESHOLD < 1
_evidence = _scope.copy()
_evidence["relationship_id"] = (
    _evidence["gene"].astype(str) + "::" +
    _evidence["human_ensembl_gene_id"].fillna("HUMAN_UNRESOLVED").astype(str)
)
assert _evidence.relationship_id.is_unique

_evidence["mouse_D02_FDR_significant_up"] = (
    _evidence.D02_vs_Sham_logFC.gt(0) & _evidence.D02_vs_Sham_FDR.lt(DISCOVERY_FDR_THRESHOLD)
).where(_evidence.D02_vs_Sham_FDR.notna(), pd.NA).astype("boolean")
_evidence["mouse_D14_FDR_significant_up"] = (
    _evidence.D14_vs_Sham_logFC.gt(0) & _evidence.D14_vs_Sham_FDR.lt(DISCOVERY_FDR_THRESHOLD)
).where(_evidence.D14_vs_Sham_FDR.notna(), pd.NA).astype("boolean")
_evidence["stroke_association_context"] = np.select(
    [
        _evidence.mouse_D02_FDR_significant_up.fillna(False).astype(bool),
        _evidence.mouse_D14_FDR_significant_up.fillna(False).astype(bool),
    ],
    ["D02_positive_DE", "D14_positive_DE"],
    default="no_significant_positive_DE_or_untested",
)
_evidence["stroke_acute_and_late_up"] = (
    _evidence.mouse_D02_FDR_significant_up.fillna(False).astype(bool) &
    _evidence.mouse_D14_FDR_significant_up.fillna(False).astype(bool)
)

_evidence["human_orthologue_mapped"] = _evidence.human_ensembl_gene_id.notna()
_evidence["human_EC_transcript_assessable"] = _evidence.unsorted_n_assessed_donors.gt(0)
_evidence["human_EC_transcript_detected"] = _evidence.human_EC_transcript_assessable & _evidence.unsorted_median_CPM.gt(0)
_evidence["human_EC_specificity_assessable"] = _evidence.EC_vs_non_EC_comparison_status.eq("descriptively_assessed")
_evidence["human_EC_specificity_direction"] = np.select(
    [
        _evidence.human_EC_specificity_assessable & _evidence.EC_vs_non_EC_median_paired_log2_ratio.gt(0),
        _evidence.human_EC_specificity_assessable & _evidence.EC_vs_non_EC_median_paired_log2_ratio.lt(0),
        _evidence.human_EC_specificity_assessable & _evidence.EC_vs_non_EC_median_paired_log2_ratio.eq(0),
    ],
    ["endothelial_higher", "comparator_higher", "equal"],
    default="not_assessable",
)
_evidence["human_orthology_evidence_lane"] = _evidence.human_orthology_review_context.fillna("unresolved")

_evidence["topology_both_species_available"] = (
    _evidence.mouse_structure_structural_review_status.eq("topology_intervals_available") &
    _evidence.human_structure_structural_review_status.eq("topology_intervals_available")
)
_evidence["template_metadata_context"] = _evidence.human_template_context.fillna("unavailable")
_evidence["docking_fragment_status"] = "none_approved_pending_manual_structure_review"

_evidence["peripheral_EC_RNA_context"] = np.where(
    _evidence.HPA_peripheral_EC_cluster_n_observed.gt(0),
    "peripheral_endothelial_transcript_detected_or_assessed",
    "peripheral_endothelial_source_unassessable",
)
_evidence["peripheral_EC_expression_is_binding_risk"] = "not_established_without_protein_and_ligand_assays"
_evidence["HPA_brain_to_peripheral_RNA_ratio_status"] = np.where(
    _evidence.HPA_audited_brain_EC_cluster_n_observed.fillna(0).gt(0),
    "HPA_brain_context_present_but_ratio_not_computed",
    "unavailable_no_exact_supported_HPA_brain_EC_cluster",
)
_evidence["delivery_evidence_family"] = _evidence.delivery_review_context.map(_DELIVERY_FAMILIES).fillna(
    "other_or_not_yet_reviewed"
)
_evidence["ligand_transport_extrapolation"] = "no_generated_peptide_transport_validation"
_evidence["primary_claim_reconciliation"] = np.where(
    _evidence.endpoint_reconciliation_complete.eq(True),
    "endpoint_verified", "endpoint_review_pending"
)
_evidence["targeted_literature_available"] = _evidence.n_directly_reviewed_PMIDs.gt(0) | (
    ~_evidence.additional_primary_review_required.fillna(True)
)
_evidence["primary_literature_review_pending"] = (
    _evidence.additional_primary_review_required.fillna(True).astype(bool) &
    ~_evidence.additional_targeted_primary_source_review_available.fillna(False).astype(bool)
)

# 3. A list of reviewer questions, not an automatic exclusion/filter or unproven risk score.
def _outstanding(row):
    q = []
    if not _true(row["human_orthologue_mapped"]):
        q.append("resolve_human_orthology")
    elif _txt(row["human_orthology_evidence_lane"]) != "Ensembl_high_confidence_one_to_one":
        q.append("assess_orthology_isoform_or_paralog_uncertainty")
    if not _true(row["human_EC_transcript_assessable"]):
        q.append("human_brain_EC_transcript_assay_unassessable")
    if not _true(row["human_EC_specificity_assessable"]):
        q.append("human_brain_cell_specificity_unassessable")
    if not _true(row["topology_both_species_available"]):
        q.append("resolve_mouse_human_extracellular_topology")
    else:
        q.append("validate_luminal_protein_epitope_and_isoform")
    category = _txt(row.get("literature_delivery_category"))
    if category in _INTRACELLULAR_CONTEXT or _txt(row.get("accessibility_review_context")) in {
        "organelle_or_intracellular_context", "intracellular_annotation"
    }:
        q.append("organelle_or_intracellular_not_automatically_blood_accessible")
    elif _txt(row.get("accessibility_review_context")) not in {"database_surface_evidence"}:
        q.append("surface_localisation_and_membrane_context_review")
    if category == "reverse_transport_candidate":
        q.append("transport_direction_opposes_intended_delivery")
    if category == "exploratory_with_negative_transport_evidence":
        q.append("retain_published_negative_transport_result")
    if category == "barrier_function_candidate":
        q.append("assess_barrier_perturbation_risk")
    if _true(row["primary_literature_review_pending"]):
        q.append("targeted_primary_biology_review_pending")
    q.extend([
        "reconcile_ligand_specific_delivery_claims_with_primary_sources",
        "confirm_systemic_blood_cell_soluble_receptor_and_peripheral_binding_sink",
        "inspect_glycans_disulfides_oligomers_and_membrane_environment",
        "approve_actual_receptor_model_coordinates_before_docking",
        "evaluate_native_uptake_vs_abluminal_intact_peptide_cargo_release",
    ])
    # Order-preserving unique entries.
    return list(dict.fromkeys(q))

_evidence["open_reviewer_questions_json"] = _evidence.apply(
    lambda row: json.dumps(_outstanding(row), ensure_ascii=False), axis=1
)
_evidence["n_open_reviewer_questions"] = _evidence.open_reviewer_questions_json.map(lambda x: len(json.loads(x)))

# 4. Review lanes prioritize documentation effort, NEVER biological merit.
def _review_lane(row):
    category = _txt(row.get("literature_delivery_category"))
    if category in _INTRACELLULAR_CONTEXT or row.accessibility_review_context in {
        "organelle_or_intracellular_context", "intracellular_annotation"
    }:
        return "intracellular_context_review"
    if row.is_curated_reference and category in {
        "peptide_delivery_reference", "established_delivery_reference",
        "experimental_delivery_reference", "viral_vector_delivery_reference",
        "native_ligand_delivery_reference", "secondary_literature_delivery_reference",
    }:
        return "delivery_reference_claim_audit"
    if row.stroke_association_context in {"D02_positive_DE", "D14_positive_DE"} and row.accessibility_review_context in {
        "database_surface_evidence", "surface_or_membrane_context_unresolved", "context_dependent_surface_pool"
    }:
        return "stroke_surface_candidate_source_and_epitope_review"
    if row.is_curated_reference:
        return "reference_context_review"
    return "additional_candidate_or_localisation_review"

_evidence["documentation_review_lane"] = _evidence.apply(_review_lane, axis=1)
_evidence["scientific_selection_status"] = "not_selected_manual_adjudication_pending"
_evidence["candidate_score"] = pd.NA
_evidence["selection_claim"] = "candidate_or_reference_hypothesis_not_validated_peptide_shuttle"

# 5. Gene-level index: preserve IDs and relationship counts. Do NOT select the best
#    human orthologue by MAX(CPM), MAX(identity), or MAX(systemic nTPM).
_gene_index = (
    _evidence.groupby("gene", sort=True)
    .agg(
        n_relationship_rows=("relationship_id", "size"),
        n_human_mapped_genes=("human_ensembl_gene_id", "nunique"),
        human_gene_ids_json=("human_ensembl_gene_id", lambda x: json.dumps(sorted(set(x.dropna().astype(str))))),
        n_relations_with_both_topologies=("topology_both_species_available", "sum"),
        n_relations_human_EC_assessable=("human_EC_transcript_assessable", "sum"),
        n_relations_peripheral_EC_assessed=("HPA_peripheral_EC_cluster_n_observed", lambda x: int((x.fillna(0) > 0).sum())),
        is_data_driven_candidate=("is_data_driven_candidate", "first"),
        is_curated_reference=("is_curated_reference", "first"),
        literature_review_status=("literature_review_status", "first"),
    )
    .reset_index()
)
assert _gene_index.gene.is_unique
assert _gene_index.n_relationship_rows.sum() == len(_evidence)

# 6. Manual decision sheet is intentionally NEVER overwritten on rerun.
#    The user can review it separately, with evidence IDs and sources.
_template_path = _selection_out / "EC_receptor_manual_adjudication_template_v1.csv"
_template_cols = [
    "relationship_id", "gene", "human_ensembl_gene_id", "manual_review_status",
    "intended_role", "selected_for_structural_modeling", "luminal_epitope_verdict",
    "intact_cargo_transcytosis_verdict", "peripheral_binding_risk_verdict",
    "human_functional_equivalence_verdict", "receptor_complex_and_glycan_verdict",
    "manual_model_fragment_coordinates", "primary_publication_PMIDs_checked",
    "experimental_controls_needed", "reviewer_notes", "reviewer", "review_date_utc",
]
_template_new = _evidence[["relationship_id", "gene", "human_ensembl_gene_id"]].copy()
_template_new["manual_review_status"] = "pending"
_template_new["intended_role"] = "undecided"
_template_new["selected_for_structural_modeling"] = False
for c in _template_cols[6:]:
    _template_new[c] = ""
_template_new = _template_new[_template_cols]
if _template_path.exists():
    _existing = pd.read_csv(_template_path, dtype=str, keep_default_na=False)
    if set(_template_cols) - set(_existing.columns):
        raise RuntimeError("Existing manual template schema differs; resolve manually, no overwrite.")
    if _existing.relationship_id.duplicated().any():
        raise RuntimeError("Duplicate relationship_id in existing manual template")
    old_ids, new_ids = set(_existing.relationship_id), set(_template_new.relationship_id)
    if old_ids != new_ids:
        raise RuntimeError("Relationship IDs changed; inspect manually before updating adjudication template")
    print(f"Preserved existing manual adjudication sheet: {_template_path}")
else:
    _template_new.to_csv(_template_path, index=False)
    print(f"Created manual adjudication sheet: {_template_path}")

# 7. Manuscript-ready evidence audit with exact denominators and explicit caveats.
_audit = pd.DataFrame([
    ("original_mouse_gene_universe", _base.gene.nunique()),
    ("original_mouse_human_relationship_or_unmapped_rows", len(_base)),
    ("structural_scope_mouse_genes", _evidence.gene.nunique()),
    ("structural_scope_relationship_rows", len(_evidence)),
    ("relationships_human_EC_transcript_assessable", int(_evidence.human_EC_transcript_assessable.sum())),
    ("relationships_human_EC_specificity_assessable", int(_evidence.human_EC_specificity_assessable.sum())),
    ("relationships_both_species_topology_available", int(_evidence.topology_both_species_available.sum())),
    ("relationships_HPA_peripheral_EC_clusters_assessed", int((_evidence.HPA_peripheral_EC_cluster_n_observed.fillna(0) > 0).sum())),
    ("relationships_HPA_exact_brain_EC_clusters_assessed", int((_evidence.HPA_audited_brain_EC_cluster_n_observed.fillna(0) > 0).sum())),
    ("relationships_primary_literature_review_pending", int(_evidence.primary_literature_review_pending.sum())),
    ("mouse_genes_primary_literature_review_pending", _evidence.loc[_evidence.primary_literature_review_pending, "gene"].nunique()),
    ("relationships_any_scoped_direct_PMID_review", int((_evidence.n_directly_reviewed_PMIDs.fillna(0) > 0).sum())),
    ("approved_receptor_fragments", int(_evidence.fragment_approved.fillna(False).sum())),
    ("scientifically_finalized_primary_receptors", 0),
], columns=["metric", "value"])

# Independent input integrity: never silently transform source evidence.
pd.testing.assert_frame_equal(
    _base.reset_index(drop=True),
    receptor_feasibility_evidence_v16.reset_index(drop=True),
    check_dtype=False,
)
assert _evidence.relationship_id.is_unique
assert _evidence.scientific_selection_status.eq("not_selected_manual_adjudication_pending").all()
assert _evidence.candidate_score.isna().all()
assert not _evidence.fragment_approved.fillna(False).any()
assert not _evidence.peptide_transport_validated.fillna(False).any()
assert _evidence.HPA_brain_to_peripheral_RNA_ratio_status.notna().all()

# 8. Export, with full lineage and no implicit exclusions.
_exports = {
    "EC_receptor_multiaxis_selection_evidence_v1": _evidence,
    "EC_receptor_gene_orthologue_index_v1": _gene_index,
    "EC_receptor_review_task_matrix_v1": _evidence[[
        "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id",
        "documentation_review_lane", "stroke_association_context",
        "human_orthology_evidence_lane", "human_EC_transcript_assessable",
        "human_EC_specificity_direction", "accessibility_review_context",
        "delivery_evidence_family", "peripheral_EC_RNA_context",
        "primary_claim_reconciliation", "open_reviewer_questions_json",
        "n_open_reviewer_questions", "scientific_selection_status",
    ]],
    "EC_receptor_selection_framework_summary_v1": _audit,
}
for name, frame in _exports.items():
    frame.to_csv(_selection_out / f"{name}.csv", index=False)

_manifest = {
    "version": SELECTION_FRAMEWORK_VERSION,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "upstream": "11.5b receptor_feasibility_evidence_v16; no input mutations",
    "unit": "mouse_gene_and_specific_human_Ensembl_gene_relationship",
    "stage": "manual_receptor_adjudication_preparation",
    "scalar_receptor_score": False,
    "auto_selection": False,
    "approved_fragments": 0,
    "no_human_orthologue_merging": True,
    "manual_adjudication_file": str(_template_path),
    "manual_adjudication_overwrite_policy": "never overwrite an existing sheet",
    "discovery_FDR_threshold": DISCOVERY_FDR_THRESHOLD,
    "n_pending_primary_review_genes": int(_evidence.loc[_evidence.primary_literature_review_pending, "gene"].nunique()),
    "known_major_uncertainties": [
        "Pending targeted primary literature review is tracked per gene and relationship",
        "Direct PubMed identifier lookup does not verify every endpoint claim",
        "HPA v24.1 has no supported exact brain endothelial cluster for matching against peripheral EC clusters",
        "Peripheral RNA expression is not a ligand-binding risk estimate",
        "Human normal temporal-lobe tissue is not a human stroke transport model",
        "Peptide uptake and intact abluminal cargo delivery have not been shown",
        "Predicted model metadata and topology intervals are not approved docking fragments",
        "Mouse-human orthology, especially LY6 family, can be one-to-many or uncertain",
    ],
    "separate_optimization_and_independent_evaluation": True,
    "next_step": "12.2 prespecified comparison and manual candidate/reference adjudication",
}
(_selection_out / "EC_receptor_selection_framework_manifest_v1.json").write_text(
    json.dumps(_manifest, indent=2), encoding="utf-8"
)

print("\nSection 12.1 selection evidence framework:")
if "display" in globals():
    display(_audit)
    display(_evidence.documentation_review_lane.value_counts().rename("relationship_rows").to_frame())
    _show = _evidence.loc[_evidence.gene.isin([
        "Tm4sf1", "Ecscr", "Procr", "Ace", "Selp", "Itm2a", "Tfrc", "Lrp1",
        "Ly6e", "Ackr1", "Tnfrsf1a", "Cd93", "Lrp8"
    ])]
    display(_show[[
        "gene", "human_gene_symbol", "human_ensembl_gene_id",
        "documentation_review_lane", "stroke_association_context",
        "human_EC_specificity_direction", "delivery_evidence_family",
        "HPA_peripheral_EC_cluster_max_nTPM", "human_template_context",
        "scientific_selection_status",
    ]].sort_values(["gene", "human_ensembl_gene_id"]))
else:
    print(_audit.to_string(index=False))
    print(_evidence.documentation_review_lane.value_counts().to_string())

print("\n12.1 exports complete. No receptor ranked, excluded, selected, or approved for docking.")
