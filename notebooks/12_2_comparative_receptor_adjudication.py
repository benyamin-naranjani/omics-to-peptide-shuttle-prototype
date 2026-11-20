"""OmiPep v0.2 | Notebook section 12.2: comparative receptor adjudication.

Run in the existing notebook with:
    %run -i 12_2_comparative_receptor_adjudication.py

Produces a versioned, relationship-specific comparison and review agenda.
Does NOT select/rank receptors, overwrite manual adjudications, verify transport
claims, approve fragments, or convert missing evidence to a negative result.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COMPARATIVE_ADJUDICATION_VERSION = "1.0"
_selection_dir = Path(out_dir)
_selection_dir.mkdir(parents=True, exist_ok=True)
_source_path = _selection_dir / "EC_receptor_multiaxis_selection_evidence_v1.csv"
_manual_path = _selection_dir / "EC_receptor_manual_adjudication_template_v1.csv"
if not _source_path.is_file() or not _manual_path.is_file():
    raise FileNotFoundError("Run Section 12.1 before Section 12.2; expected source and manual sheet are missing.")

# Focal cases represent a transparent, revisable discussion agenda, NOT a rank.
FOCAL_EXPLORATORY_HYPOTHESES = ("Ecscr", "Tm4sf1", "Procr")
FOCAL_ALTERNATIVE_HYPOTHESES = (
    "Ace", "Selp", "Ackr1", "Cd93", "Bst2", "Itgb1",
    "Il4ra", "Cd200", "Tnfrsf1a", "Mcam",
)
FOCAL_REFERENCE_CONTEXT = {
    "Lrp1": "published_peptide_delivery_reference; verify ligand, epitope and quantitative endpoints",
    "Tfrc": "antibody_brain_delivery_reference; separate peptide T7 benchmark claims",
    "Insr": "antibody_brain_delivery_reference; species and insulin-signalling context",
    "Igf1r": "antibody_brain_delivery_reference; construct and species context",
    "Slc3a2": "antibody_delivery_reference; transporter-complex context",
    "Ly6a": "mouse_AAV_vector_reference_not_human_peptide_evidence",
    "Ly6e": "AAV_vector_reference_not_standalone_peptide_evidence",
    "Car4": "AAV_vector_reference_not_standalone_peptide_evidence",
    "Itm2a": "exploratory_reference_with_negative_or_inconclusive_transport_result",
}

# Preserve original metadata and all human paralog/ortholog relationships.
_e12 = pd.read_csv(_source_path, low_memory=False, dtype={
    "gene": "string", "human_ensembl_gene_id": "string", "relationship_id": "string"
})
_manual12 = pd.read_csv(_manual_path, dtype=str, keep_default_na=False)
_required12 = {
    "gene", "human_ensembl_gene_id", "human_gene_symbol", "relationship_id",
    "is_data_driven_candidate", "is_curated_reference", "stroke_association_context",
    "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "D14_vs_Sham_logFC", "D14_vs_Sham_FDR",
    "documentation_review_lane", "mouse_surface_class", "accessibility_review_context",
    "human_orthology_evidence_lane", "human_EC_transcript_assessable",
    "human_EC_specificity_assessable", "human_EC_specificity_direction",
    "topology_both_species_available", "human_template_context", "delivery_evidence_family",
    "BBB_transcytosis_evidence", "literature_review_status", "primary_claim_reconciliation",
    "primary_literature_review_pending", "open_reviewer_questions_json",
    "HPA_peripheral_EC_cluster_n_observed", "HPA_peripheral_EC_cluster_max_nTPM",
    "HPA_audited_brain_EC_cluster_n_observed",
    "HPA_brain_to_peripheral_RNA_ratio_status", "fragment_approved",
    "scientific_selection_status", "candidate_score", "unsorted_median_CPM",
    "EC_vs_non_EC_median_paired_log2_ratio", "external_identity_pct",
}
missing = sorted(_required12 - set(_e12.columns))
if missing:
    raise RuntimeError(f"12.1 evidence table has unexpected schema, missing: {missing}")
manual_required = {"relationship_id", "gene", "human_ensembl_gene_id", "manual_review_status",
                   "intended_role", "selected_for_structural_modeling", "manual_model_fragment_coordinates"}
if manual_required - set(_manual12.columns):
    raise RuntimeError(f"Manual adjudication sheet missing: {sorted(manual_required-set(_manual12.columns))}")
if _e12.relationship_id.isna().any() or _e12.relationship_id.duplicated().any():
    raise RuntimeError("12.1 relationship IDs are missing or duplicated")
if _manual12.relationship_id.duplicated().any() or set(_manual12.relationship_id) != set(_e12.relationship_id):
    raise RuntimeError("Manual adjudication relationship IDs are missing, duplicated or changed")
if _e12.gene.nunique() != 114 or len(_e12) != 125:
    raise RuntimeError("Expected 114 structural-scope mouse genes and 125 relationship rows; inspect upstream first")
if set(_e12.loc[_e12.human_ensembl_gene_id.notna(), "human_ensembl_gene_id"].dropna().astype(str)).difference(
    set(_e12.human_ensembl_gene_id.dropna().astype(str))
):
    raise RuntimeError("Inconsistent human IDs")

# Parsing is strict: absent entries stay unknown rather than being cast to False.
def _as_bool(series):
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = series.astype("string").str.lower().str.strip().map(mapping).astype("boolean")
    bad = series.notna() & values.isna()
    if bad.any():
        raise ValueError(f"Unrecognized boolean values: {series.loc[bad].unique().tolist()}")
    return values

for col in ("fragment_approved", "human_EC_transcript_assessable", "human_EC_specificity_assessable",
            "topology_both_species_available", "primary_literature_review_pending"):
    _e12[col] = _as_bool(_e12[col])
if _e12.fragment_approved.fillna(False).any():
    raise RuntimeError("Upstream fragment approval unexpectedly changed; audit before continuing")
if _e12.candidate_score.notna().any():
    raise RuntimeError("Upstream candidate scores unexpectedly populated; audit first")
if not _e12.scientific_selection_status.eq("not_selected_manual_adjudication_pending").all():
    raise RuntimeError("Upstream final selection status changed; do not overwrite a manual decision")

# A frozen-on-first-run, disclosed comparison agenda. Not retrospective pre-registration.
_protocol_file = _selection_dir / "EC_receptor_12_2_comparison_protocol_v1.json"
_protocol = {
    "version": COMPARATIVE_ADJUDICATION_VERSION,
    "comparison_is_a_documentation_agenda_not_a_final_ranking": True,
    "provisional_exploratory_review_examples": list(FOCAL_EXPLORATORY_HYPOTHESES),
    "alternative_review_examples": list(FOCAL_ALTERNATIVE_HYPOTHESES),
    "reference_comparison_examples": FOCAL_REFERENCE_CONTEXT,
    "comparison_axes": [
        "mouse_D02_D14_DE_and_formal_effect_robustness",
        "human_brain_EC_expression_and_donor_paired_specificity",
        "relationship_specific_mouse_human_orthology_and_extracellular_conservation",
        "blood_facing_accessibility_not_inferred_from_topology",
        "published_ligand_specific_uptake_direction_and_intact_cargo_BBB_transport",
        "systemic_tissue_peripheral_endothelial_and_immune_RNA_context",
        "native_receptor_complex_glycans_disulfides_and_membrane_structure",
        "primary_claim_verification_and_manual_review",
    ],
    "no_composite_score": True,
    "no_automatic_selection": True,
    "no_automatic_exclusion": True,
    "no_fragment_approval": True,
    "no_orthologue_merging": True,
    "missing_data_remain_unassessed": True,
}
if _protocol_file.exists():
    existing = json.loads(_protocol_file.read_text(encoding="utf-8"))
    if existing != _protocol:
        raise RuntimeError("12.2 comparison protocol changed; version it explicitly rather than silently overwriting")
else:
    _protocol_file.write_text(json.dumps(_protocol, indent=2), encoding="utf-8")

all_focal_genes = set(FOCAL_EXPLORATORY_HYPOTHESES) | set(FOCAL_ALTERNATIVE_HYPOTHESES) | set(FOCAL_REFERENCE_CONTEXT)
missing_focal = sorted(all_focal_genes - set(_e12.gene))
if missing_focal:
    raise RuntimeError(f"Focal comparison gene(s) absent from structural review: {missing_focal}")

_manual_small = _manual12[["relationship_id", "manual_review_status", "intended_role",
                            "selected_for_structural_modeling", "manual_model_fragment_coordinates"]].rename(
    columns={"manual_review_status":"manual_review_status_recorded",
             "intended_role":"manual_intended_role_recorded",
             "selected_for_structural_modeling":"manual_modeling_choice_recorded",
             "manual_model_fragment_coordinates":"manual_fragment_coordinates_recorded"}
)
_comparison = _e12.merge(_manual_small, on="relationship_id", how="left", validate="one_to_one")
assert len(_comparison) == len(_e12)

def _discussion_lane(gene):
    if gene in FOCAL_EXPLORATORY_HYPOTHESES:
        return "provisional_exploratory_deep_review_example"
    if gene in FOCAL_ALTERNATIVE_HYPOTHESES:
        return "alternative_exploratory_review"
    if gene in FOCAL_REFERENCE_CONTEXT:
        return "reference_or_negative_control_review"
    return "full_universe_retained_not_a_focal_example"

_comparison["comparison_discussion_lane"] = _comparison.gene.map(_discussion_lane)
_comparison["reference_ligand_format_context"] = _comparison.gene.map(FOCAL_REFERENCE_CONTEXT).fillna("")
_comparison["focal_example_not_final_selection"] = _comparison.gene.isin(all_focal_genes)
_comparison["comparison_priority_is_not_biological_merit"] = True
_comparison["human_orthology_relationship_retained_separately"] = True
_comparison["generated_peptide_BBB_transport_validated"] = False
_comparison["docking_fragment_approved_by_this_cell"] = False
_comparison["final_receptor_selected_by_this_cell"] = False

# Critical gaps are factual review requirements, not automatic failures.
def _txt12(x):
    return "" if pd.isna(x) else str(x).strip()

def _review_flags12(row):
    flags = list(json.loads(row.open_reviewer_questions_json))
    if row.primary_claim_reconciliation != "endpoint_verified":
        flags.append("primary_claim_and_endpoint_reconciliation_pending")
    if pd.isna(row.primary_literature_review_pending) or bool(row.primary_literature_review_pending):
        flags.append("targeted_primary_literature_pending")
    if pd.isna(row.human_EC_transcript_assessable) or not bool(row.human_EC_transcript_assessable):
        flags.append("human_brain_EC_expression_unassessable")
    if pd.isna(row.human_EC_specificity_assessable) or not bool(row.human_EC_specificity_assessable):
        flags.append("human_brain_EC_specificity_unassessable")
    if pd.isna(row.topology_both_species_available) or not bool(row.topology_both_species_available):
        flags.append("mouse_human_topology_not_both_resolved")
    if row.delivery_evidence_family == "reverse_direction_transport_context":
        flags.append("published_transport_direction_not_blood_to_brain")
    if row.delivery_evidence_family == "negative_or_inconclusive_ligand_test":
        flags.append("retain_negative_or_inconclusive_ligand_study")
    if _txt12(row.accessibility_review_context) in {"organelle_or_intracellular_context", "intracellular_annotation"}:
        flags.append("intracellular_context_requires_blood_facing_protein_evidence")
    flags.append("HPA_peripheral_RNA_not_a_quantitative_peripheral_binding_risk")
    flags.append("no_HPA_exact_supported_brain_EC_reference_for_matched_ratio")
    return list(dict.fromkeys(flags))

_comparison["comparative_review_gaps_json"] = _comparison.apply(
    lambda r: json.dumps(_review_flags12(r), ensure_ascii=False), axis=1)
_comparison["n_comparative_review_gaps"] = _comparison.comparative_review_gaps_json.map(
    lambda x: len(json.loads(x)))

# A final selection cannot be claimed by this stage. Manual entries are read-only.
_comparison["manual_review_needs_completion"] = _comparison.manual_review_status_recorded.str.lower().ne("completed")
_comparison["peptide_design_readiness"] = "not_approved_manual_biology_epitope_and_structure_review_pending"

# Preserve all original axes; generate a compact comparison view without using MAX()
# across alternative human gene relationships.
_compact_cols = [
    "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id",
    "comparison_discussion_lane", "is_data_driven_candidate", "is_curated_reference",
    "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "D14_vs_Sham_logFC", "D14_vs_Sham_FDR",
    "stroke_association_context", "mouse_surface_class", "accessibility_review_context",
    "human_orthology_evidence_lane", "external_identity_pct", "unsorted_median_CPM",
    "EC_vs_non_EC_median_paired_log2_ratio", "human_EC_specificity_direction",
    "HPA_peripheral_EC_cluster_max_nTPM", "HPA_brain_to_peripheral_RNA_ratio_status",
    "delivery_evidence_family", "BBB_transcytosis_evidence", "primary_claim_reconciliation",
    "literature_review_status", "human_template_context", "topology_both_species_available",
    "manual_review_status_recorded", "manual_intended_role_recorded", "manual_modeling_choice_recorded",
    "peptide_design_readiness", "comparative_review_gaps_json",
]
for optional in (
    "D02_vs_Sham_candidate_robust_minimum_effect",
    "D02_vs_Sham_candidate_fully_robust",
    "HPA_immune_cell_max_nTPM",
    "HPA_peripheral_tissue_max_nTPM",
    "mouse_n_total_external_residues",
    "human_annotation_n_total_external_residues",
    "multiple_optimal_alignments",
    "external_mouse_joint_coverage_pct",
    "external_human_joint_coverage_pct",
):
    if optional in _comparison.columns:
        _compact_cols.append(optional)

_compact = _comparison[_compact_cols].copy()
_focus = _compact.loc[_compact.gene.isin(all_focal_genes)].copy()
_focus["discussion_order"] = _focus.gene.map({
    **{g:i for i,g in enumerate(FOCAL_EXPLORATORY_HYPOTHESES)},
    **{g:i+len(FOCAL_EXPLORATORY_HYPOTHESES) for i,g in enumerate(FOCAL_ALTERNATIVE_HYPOTHESES)},
    **{g:i+len(FOCAL_EXPLORATORY_HYPOTHESES)+len(FOCAL_ALTERNATIVE_HYPOTHESES)
       for i,g in enumerate(FOCAL_REFERENCE_CONTEXT)},
})
_focus = _focus.sort_values(["discussion_order", "gene", "human_ensembl_gene_id"], kind="stable")

# This audit shows whether the manual worksheet has any entries, but does not apply them
# automatically as receptor selection, and NEVER mutates or rewrites the worksheet.
_manual_audit = _manual12[["relationship_id", "gene", "human_ensembl_gene_id",
                           "manual_review_status", "intended_role", "selected_for_structural_modeling",
                           "manual_model_fragment_coordinates"]].copy()
_manual_audit["manual_review_recorded"] = _manual_audit.manual_review_status.str.strip().str.lower().ne("pending")
_manual_audit["manual_role_recorded"] = ~_manual_audit.intended_role.str.strip().str.lower().isin(["", "undecided"])
_manual_audit["modeling_choice_recorded"] = _manual_audit.selected_for_structural_modeling.str.strip().str.lower().eq("true")
_manual_audit["coordinates_text_entered"] = _manual_audit.manual_model_fragment_coordinates.str.strip().ne("")
_manual_audit["manual_fragment_text_requires_validation"] = _manual_audit.coordinates_text_entered
_manual_audit["manual_entry_is_not_finalized_by_this_cell"] = True

# Prespecified design-decision questions, formulated as requested evidence rather than claims.
_decision_questions = [
    ("1", "stroke_signal", "Is D02/D14 differential expression robust at sample level and beyond a meaningful effect size?", "edgeR QL, LOSO, annotation sensitivity and glmTreat", "required_for_stroke_discovery_claim"),
    ("2", "human_translatability", "Is this specific human orthologue confidently mapped, EC-expressed and accessible?", "orthology, human donor RNA and canonical protein topology", "required_for_human_relevance_claim"),
    ("3", "luminal_accessibility", "Is the native adult brain EC epitope exposed on the blood-facing membrane?", "polarized endothelial protein/antibody or accessibility assay", "requires_new_evidence"),
    ("4", "receptor_traffic", "Does the selected ligand mediate productive internalization rather than degradation/recycling?", "matched ligand format, receptor dependence and trafficking", "requires_target_specific_evidence"),
    ("5", "brain_delivery", "Is intact peptide/cargo shown beyond brain vasculature, with preserved barrier integrity?", "quantitative abluminal or capillary-depleted parenchymal transport", "not_yet_established_for_generated_peptides"),
    ("6", "peripheral_sink", "Could peripheral vessels, blood cells or soluble receptor bind the candidate?", "HPA transcript context plus native protein and ligand binding", "RNA_context_only"),
    ("7", "model_selection", "Are receptor construct, membrane context, glycan/SS bridges and PDB/model quality suitable?", "target-specific structural inspection and explicitly approved coordinates", "fragments_not_approved"),
    ("8", "controls", "Which positive and negative ligand/cargo controls share the same transport assay and endpoint?", "defined ligand-format controls, intact-peptide detection and barrier readout", "assay_controls_not_yet_finalized"),
]
_question_df = pd.DataFrame(_decision_questions, columns=[
    "question_id", "axis", "adjudication_question", "required_evidence", "present_status"])

# No silent omission, no source mutations, no effects inferred from missing values.
assert len(_comparison) == 125 and _comparison.gene.nunique() == 114
assert _comparison.relationship_id.is_unique
assert set(_comparison.relationship_id) == set(_manual12.relationship_id)
assert not _comparison.final_receptor_selected_by_this_cell.any()
assert not _comparison.docking_fragment_approved_by_this_cell.any()
assert not _comparison.generated_peptide_BBB_transport_validated.any()
assert _comparison.candidate_score.isna().all()
assert _comparison.fragment_approved.fillna(False).eq(False).all()
assert set(_focus.gene) == all_focal_genes
assert len(_focus) >= len(all_focal_genes)  # More than one relationship per mouse gene is expected.

_exports12 = {
    "EC_receptor_comparative_adjudication_full_v1": _comparison,
    "EC_receptor_comparative_adjudication_compact_v1": _compact,
    "EC_receptor_comparative_focus_cases_v1": _focus,
    "EC_receptor_manual_adjudication_readonly_audit_v1": _manual_audit,
    "EC_receptor_adjudication_decision_questions_v1": _question_df,
}
for stem, table in _exports12.items():
    table.to_csv(_selection_dir / (stem + ".csv"), index=False)

_summary12 = pd.DataFrame([
    ("structural_scope_mouse_genes", _comparison.gene.nunique()),
    ("structural_scope_relationship_rows", len(_comparison)),
    ("provisional_exploratory_review_examples", len(FOCAL_EXPLORATORY_HYPOTHESES)),
    ("alternative_exploratory_review_examples", len(FOCAL_ALTERNATIVE_HYPOTHESES)),
    ("reference_or_negative_control_review_examples", len(FOCAL_REFERENCE_CONTEXT)),
    ("focal_relationship_rows", len(_focus)),
    ("relationship_rows_pending_manual_review", int(_comparison.manual_review_needs_completion.sum())),
    ("relationship_rows_pending_primary_literature", int(_comparison.primary_literature_review_pending.fillna(True).sum())),
    ("relationship_rows_with_both_species_topology", int(_comparison.topology_both_species_available.fillna(False).sum())),
    ("relationship_rows_with_human_brain_EC_RNA", int(_comparison.human_EC_transcript_assessable.fillna(False).sum())),
    ("relationship_rows_with_HPA_exact_brain_EC_context", int(pd.to_numeric(
        _comparison["HPA_audited_brain_EC_cluster_n_observed"], errors="coerce").fillna(0).gt(0).sum())),
    ("approved_fragments", 0),
    ("scientifically_finalized_receptors", 0),
], columns=["metric", "value"])
_summary12.to_csv(_selection_dir / "EC_receptor_12_2_comparison_summary_v1.csv", index=False)

_manifest12 = {
    "version": COMPARATIVE_ADJUDICATION_VERSION,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "input_csv": _source_path.name,
    "input_sha256": hashlib.sha256(_source_path.read_bytes()).hexdigest(),
    "manual_sheet_csv": _manual_path.name,
    "manual_sheet_sha256_at_analysis": hashlib.sha256(_manual_path.read_bytes()).hexdigest(),
    "manual_sheet_untouched": True,
    "comparison_protocol": _protocol_file.name,
    "n_relationships": len(_comparison),
    "n_mouse_genes": _comparison.gene.nunique(),
    "provisional_deep_review_examples_not_selected": list(FOCAL_EXPLORATORY_HYPOTHESES),
    "no_human_orthologue_merging": True,
    "no_composite_score": True,
    "no_auto_ranking_or_exclusion": True,
    "no_approved_fragment": True,
    "human_brain_EC_vs_HPA_peripheral_ratio_unavailable": True,
    "interpretation_constraints": [
        "Literature identity and endpoint reconciliation is incomplete",
        "Human control brain RNA is not human stroke transport evidence",
        "Peripheral EC transcript abundance is not a ligand-specific off-target risk measurement",
        "Alternative orthologue relationships are not merged or optimized independently",
        "Published antibody, native ligand or AAV delivery is not generated peptide delivery",
        "Structural mapping availability is not docking fragment approval",
        "A provisional focus list is documentation triage, never a final receptor choice",
    ],
}
(_selection_dir / "EC_receptor_12_2_comparison_manifest_v1.json").write_text(
    json.dumps(_manifest12, indent=2), encoding="utf-8")

print("\nSection 12.2 comparative adjudication:")
if "display" in globals():
    display(_summary12)
    display(_focus[[
        "gene", "human_gene_symbol", "human_ensembl_gene_id", "comparison_discussion_lane",
        "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "unsorted_median_CPM",
        "EC_vs_non_EC_median_paired_log2_ratio", "external_identity_pct",
        "delivery_evidence_family", "primary_claim_reconciliation", "manual_review_status_recorded",
    ]].round(3))
    display(_comparison.comparison_discussion_lane.value_counts().rename("relationship_rows").to_frame())
else:
    print(_summary12.to_string(index=False))
    print(_focus[["gene", "human_gene_symbol", "comparison_discussion_lane"]].to_string(index=False))
print("\nAll 125 relationships retained. Manual adjudication sheet preserved.")
print("Three genes are provisional deep-review examples, NOT selected receptors.")
print("No composite scoring, exclusion, fragment approval, or final target selection performed.")
