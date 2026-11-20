"""OmiPep v0.2 — Notebook cell 12.3, receptor decision gates and control plan.

Usage inside notebooks/:
    %run -i 12_3_receptor_decision_gates.py

Inputs: Section 12.2 full comparison and the UNEDITED-BY-THIS-SCRIPT
relationship-specific manual adjudication template created in 12.1.

This script does NOT select receptors, score them, approve fragments, assume
blood-facing protein expression, validate published evidence or claim delivery
of generated peptides. It generates explicit evidence requirements and a
read-only manual-adjudication audit, keeping alternative human genes distinct.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

DECISION_GATE_VERSION = "1.0"
_dir123 = Path(out_dir)
_dir123.mkdir(parents=True, exist_ok=True)
_comparison_path123 = _dir123 / "EC_receptor_comparative_adjudication_full_v1.csv"
_manual_path123 = _dir123 / "EC_receptor_manual_adjudication_template_v1.csv"
for _path123 in (_comparison_path123, _manual_path123):
    if not _path123.is_file():
        raise FileNotFoundError(f"Required Section 12.2 input absent: {_path123}")

_hash123 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
_manual_hash_before123 = _hash123(_manual_path123)
_comparison123 = pd.read_csv(
    _comparison_path123,
    low_memory=False,
    dtype={"gene": "string", "human_ensembl_gene_id": "string", "relationship_id": "string"},
)
_manual123 = pd.read_csv(_manual_path123, dtype=str, keep_default_na=False)

_required_comparison123 = {
    "relationship_id", "gene", "human_ensembl_gene_id", "human_gene_symbol",
    "comparison_discussion_lane", "is_data_driven_candidate", "is_curated_reference",
    "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "D14_vs_Sham_logFC",
    "D14_vs_Sham_FDR", "human_EC_transcript_assessable",
    "human_EC_specificity_assessable", "human_EC_specificity_direction",
    "unsorted_median_CPM", "EC_vs_non_EC_median_paired_log2_ratio",
    "topology_both_species_available", "human_template_context", "mouse_surface_class",
    "delivery_evidence_family", "BBB_transcytosis_evidence",
    "primary_claim_reconciliation", "primary_literature_review_pending",
    "HPA_peripheral_EC_cluster_n_observed", "HPA_peripheral_EC_cluster_max_nTPM",
    "HPA_audited_brain_EC_cluster_n_observed", "fragment_approved",
    "scientific_selection_status", "candidate_score", "comparative_review_gaps_json",
}
_required_manual123 = {
    "relationship_id", "gene", "human_ensembl_gene_id", "manual_review_status",
    "intended_role", "selected_for_structural_modeling", "luminal_epitope_verdict",
    "intact_cargo_transcytosis_verdict", "peripheral_binding_risk_verdict",
    "human_functional_equivalence_verdict", "receptor_complex_and_glycan_verdict",
    "manual_model_fragment_coordinates", "primary_publication_PMIDs_checked",
    "experimental_controls_needed", "reviewer_notes", "reviewer", "review_date_utc",
}
for _table123, _needed123, _name123 in (
    (_comparison123, _required_comparison123, "12.2 comparison"),
    (_manual123, _required_manual123, "12.1 manual template"),
):
    _missing123 = sorted(_needed123 - set(_table123.columns))
    if _missing123:
        raise RuntimeError(f"{_name123} columns missing: {_missing123}")

if len(_comparison123) != 125 or _comparison123.gene.nunique() != 114:
    raise RuntimeError("Unexpected relationship universe: expected 125 rows/114 mouse genes")
if _comparison123.relationship_id.isna().any() or _comparison123.relationship_id.duplicated().any():
    raise RuntimeError("Relationship IDs must be unique and non-missing")
if _manual123.relationship_id.duplicated().any() or set(_manual123.relationship_id) != set(_comparison123.relationship_id):
    raise RuntimeError("Manual sheet relationship IDs do not match the comparison universe")

# Validate exact gene and human identifier associations, not only row counts.
_expected123 = _comparison123[["relationship_id", "gene", "human_ensembl_gene_id"]].fillna("").astype(str)
_actual123 = _manual123[["relationship_id", "gene", "human_ensembl_gene_id"]].fillna("").astype(str)
_check123 = _expected123.merge(
    _actual123, on="relationship_id", how="outer", suffixes=("_comparison", "_manual"),
    validate="one_to_one", indicator=True,
)
if (
    not _check123._merge.eq("both").all()
    or not _check123.gene_comparison.eq(_check123.gene_manual).all()
    or not _check123.human_ensembl_gene_id_comparison.eq(_check123.human_ensembl_gene_id_manual).all()
):
    raise RuntimeError("Manual sheet gene/orthologue assignments differ from 12.2")

_bool_values123 = {"true": True, "false": False, "1": True, "0": False}
def _parse_bool123(series):
    original = series.astype("string").str.strip().str.lower()
    result = original.map(_bool_values123).astype("boolean")
    invalid = original.notna() & result.isna()
    if invalid.any():
        raise ValueError(f"Unexpected boolean entries: {sorted(original.loc[invalid].unique().tolist())}")
    return result

for _name123 in (
    "human_EC_transcript_assessable", "human_EC_specificity_assessable",
    "topology_both_species_available", "primary_literature_review_pending", "fragment_approved",
):
    _comparison123[_name123] = _parse_bool123(_comparison123[_name123])

if _comparison123.fragment_approved.fillna(False).any():
    raise RuntimeError("An upstream fragment has been approved; re-audit before using v1 protocol")
if _comparison123.candidate_score.notna().any():
    raise RuntimeError("Unexpected candidate scoring in 12.2 input")
if not _comparison123.scientific_selection_status.eq("not_selected_manual_adjudication_pending").all():
    raise RuntimeError("An upstream selection state has changed; version the decision protocol")

# 1. Decision policy. These are distinctly different kinds of evidence:
# discovery claim, modeling feasibility, and experimental efficacy claim.
# They must NOT be collapsed into a scalar score or treated as interchangeable.
_GATE_COLUMNS123 = [
    "gate_id", "claim_or_workflow", "acceptance_question", "minimum_evidence_for_claim",
    "current_evidence_role", "required_next_validation", "automatic_exclusion", "scope",
]
_GATE_ROWS123 = [
    ("D01", "stroke_responsive_discovery", "Is the specific mouse gene positively DE after stroke?",
     "Correct sample-level DE direction, FDR, and documented LOSO/formal-effect sensitivity",
     "Existing mouse edgeR and robustness data; not evidence of BBB transport",
     "Review sample/replicate effects and mouse endothelial subtype contributions", False, "stroke-responsive candidate claim only"),
    ("D02", "human_orthology", "Which specific human gene and protein are the orthologous counterparts?",
     "Stable ID relationship, orthology confidence/paralog/isoform review, and aligned extracellular sequence",
     "Relationship-specific computational evidence, not functional equivalence",
     "Manually reconcile uncertain relationships and native receptor isoforms", False, "human translatability claim"),
    ("D03", "human_brain_ec_rna", "Is the candidate detected in human brain EC donors?",
     "Measured donor-level gene expression with correct distinction between zero and unassessable",
     "Wälchli adult temporal-lobe RNA; no claim of protein abundance or luminal expression",
     "Check disease-context and protein-level validation", False, "human brain endothelial context"),
    ("D04", "blood_facing_accessibility", "Does the mature native receptor present the target epitope luminally?",
     "Native polarized brain EC protein localization and isoform/epitope confirmation",
     "UniProt topology provides only a candidate accessible region",
     "Native luminal antibody/ligand binding and relevant membrane/receptor context", False, "biological suitability / fragment choice"),
    ("D05", "traffic_and_direction", "Does receptor-bound cargo travel blood-to-brain instead of recycling/degrading?",
     "Ligand-specific uptake, receptor dependence and productive intracellular transport",
     "Published antibody/native-ligand/AAV evidence is format-specific; reverse direction must be labeled",
     "Receptor dependence, endosomal routing, matched controls", False, "delivery mechanism claim"),
    ("D06", "intact_cargo_transcytosis", "Does an intact generated peptide/cargo reach the brain-facing side?",
     "Quantitative intact transport, preserved barrier integrity, and appropriate extravascular readouts",
     "Not established for the generated candidates",
     "Standardized BBB translocation assay and conditional in vivo capillary depletion", False, "experimental efficacy claim only"),
    ("D07", "peripheral_sink_and_safety", "Do peripheral tissues, immune cells or soluble receptors bind the ligand?",
     "Specific protein accessibility and ligand interaction, informed by systemic RNA and physiology",
     "HPA peripheral context cannot be ratioed to Wälchli RNA and cannot predict binding risk",
     "Check peripheral expression, blood cells, ligand sinks and signalling risks", False, "off-target context"),
    ("D08", "structure_and_fragment", "Is a specific receptor structure/fragment suitable for reproducible docking?",
     "Explicitly reviewed protein accession/version, residue coordinates, glycans, disulfides, TM/oligomer context, structure quality",
     "Structural-template availability and sequence spans are NOT fragment approval",
     "Manual coordinate approval with structural provenance; membrane/complex model if needed", False, "receptor-model preparation"),
    ("D09", "assay_controls", "Are established positive and mechanistic negative controls defined?",
     "Reference ligands on matched endpoints, scrambled/wrong-receptor controls, barrier and intact-peptide readouts",
     "Benchmark families identified, exact assay and sequence verification pending",
     "Freeze assay-specific controls before reporting performance", False, "independent evaluation"),
    ("D10", "publication_claim_integrity", "Does the proposed Results wording match the strongest verified evidence?",
     "Reconciled primary sources, explicit negative studies, no extrapolation from RNA to transcytosis",
     "Claim-level literature checks and manual adjudications remain incomplete",
     "Targeted literature completion and manuscript/source audit", False, "manuscript readiness"),
]
_gate_table123 = pd.DataFrame(_GATE_ROWS123, columns=_GATE_COLUMNS123)

# 2. Freeze protocol; do not pretend it was preregistered.
_protocol_path123 = _dir123 / "EC_receptor_12_3_decision_protocol_v1.json"
_protocol123 = {
    "version": DECISION_GATE_VERSION,
    "purpose": "pre-decision evidence gates, manual review audit and benchmark planning",
    "not_preregistered": True,
    "decision_units": "individual_mouse_gene_and_specific_human_gene_relationship",
    "decision_rule_ids": _gate_table123.gate_id.tolist(),
    "role_examples_not_selection": {
        "exploratory_deep_review": ["Ecscr", "Tm4sf1", "Procr"],
        "references": ["Lrp1", "Tfrc", "Insr", "Igf1r", "Slc3a2"],
    },
    "manual_sheet_is_read_only": True,
    "no_automatic_selection_or_exclusion": True,
    "no_composite_score": True,
    "no_fragment_approval": True,
    "no_generated_peptide_transport_claim": True,
    "evidence_claim_levels_separate": [
        "mouse_stroke_transcriptomic_association", "human_EC_transcriptomic_context",
        "native_receptor_structure_feasibility", "published_ligand_transport_context",
        "experimental_intact_generated_peptide_delivery_not_demonstrated",
    ],
    "limitations": [
        "73 of 125 structural relationships and 72 of 114 genes had pending targeted primary literature review in 12.1",
        "HPA contains no supported exact brain EC comparator; peripheral nTPM is not quantitative ligand binding",
        "One-to-many orthologues remain separate; never use a maximum across relationships for choosing a human target",
        "An entered manual modeling choice or coordinate string is not a structurally approved fragment",
        "Human control brain transcriptomics does not verify human stroke receptor expression or cargo transport",
    ],
}
if _protocol_path123.exists():
    if json.loads(_protocol_path123.read_text(encoding="utf-8")) != _protocol123:
        raise RuntimeError("12.3 decision protocol changed; create a new version, do not silently overwrite")
else:
    _protocol_path123.write_text(json.dumps(_protocol123, indent=2), encoding="utf-8")

# 3. Manual adjudication: read-only, with explicit completeness checks. Any
# text in a verdict is NOT parsed as a scientific pass/fail or evidence of approval.
_merged123 = _comparison123[[
    "relationship_id", "gene", "human_ensembl_gene_id", "human_gene_symbol",
    "comparison_discussion_lane", "is_data_driven_candidate", "is_curated_reference",
    "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "D14_vs_Sham_logFC", "D14_vs_Sham_FDR",
    "unsorted_median_CPM", "EC_vs_non_EC_median_paired_log2_ratio",
    "human_EC_transcript_assessable", "human_EC_specificity_assessable",
    "human_EC_specificity_direction", "topology_both_species_available",
    "human_template_context", "mouse_surface_class", "delivery_evidence_family",
    "BBB_transcytosis_evidence", "primary_claim_reconciliation",
    "primary_literature_review_pending", "HPA_peripheral_EC_cluster_n_observed",
    "HPA_peripheral_EC_cluster_max_nTPM", "HPA_audited_brain_EC_cluster_n_observed",
    "comparative_review_gaps_json",
]].merge(_manual123, on=["relationship_id", "gene", "human_ensembl_gene_id"],
          how="left", validate="one_to_one")
if len(_merged123) != 125 or _merged123.relationship_id.duplicated().any():
    raise RuntimeError("Relationship universe changed in manual merge")

_str123 = lambda s: s.fillna("").astype("string").str.strip()
_status123 = _str123(_merged123.manual_review_status).str.lower()
_allowed_status123 = {"pending", "in_progress", "completed", "needs_revision"}
_merged123["manual_status_valid"] = _status123.isin(_allowed_status123)
_merged123["manual_review_completed"] = _status123.eq("completed")
_role123 = _str123(_merged123.intended_role).str.lower()
_allowed_roles123 = {
    "undecided", "provisional_primary", "provisional_secondary", "reference_control",
    "negative_control", "deferred", "not_applicable",
}
_merged123["manual_role_value_valid"] = _role123.isin(_allowed_roles123)
_merged123["manual_role_is_assignment"] = _role123.ne("undecided") & _merged123.manual_role_value_valid
_model_raw123 = _str123(_merged123.selected_for_structural_modeling).str.lower()
_merged123["manual_modeling_field_valid"] = _model_raw123.isin({"true", "false", "1", "0"})
_merged123["manual_modeling_requested"] = _model_raw123.isin({"true", "1"})

_material_fields123 = [
    "luminal_epitope_verdict", "peripheral_binding_risk_verdict",
    "human_functional_equivalence_verdict", "receptor_complex_and_glycan_verdict",
    "primary_publication_PMIDs_checked", "reviewer_notes", "reviewer", "review_date_utc",
]
for _field123 in _material_fields123 + ["intact_cargo_transcytosis_verdict", "manual_model_fragment_coordinates"]:
    _merged123[f"{_field123}_recorded"] = _str123(_merged123[_field123]).ne("")
_merged123["all_review_documentation_recorded"] = _merged123[
    [f"{f}_recorded" for f in _material_fields123]
].all(axis=1)
_merged123["complete_record_documented"] = (
    _merged123.manual_review_completed
    & _merged123.manual_status_valid
    & _merged123.manual_role_is_assignment
    & _merged123.manual_modeling_field_valid
    & _merged123.all_review_documentation_recorded
)
_merged123["modeling_request_has_coordinates"] = (
    ~_merged123.manual_modeling_requested
    | _merged123.manual_model_fragment_coordinates_recorded
)
_merged123["needs_manual_correction"] = (
    ~_merged123.manual_status_valid
    | ~_merged123.manual_role_value_valid
    | ~_merged123.manual_modeling_field_valid
    | (_merged123.manual_review_completed & ~_merged123.complete_record_documented)
    | (_merged123.manual_modeling_requested & ~_merged123.modeling_request_has_coordinates)
)

# Explicit descriptive evidence lanes: no compensatory optimization score.
_d02_tested123 = pd.to_numeric(_merged123.D02_vs_Sham_FDR, errors="coerce").notna()
_d02_assoc123 = (
    pd.to_numeric(_merged123.D02_vs_Sham_FDR, errors="coerce").lt(0.05)
    & pd.to_numeric(_merged123.D02_vs_Sham_logFC, errors="coerce").gt(0)
)
_merged123["mouse_D02_positive_DE_evidence"] = np.select(
    [~_d02_tested123, _d02_assoc123],
    ["not_tested", "positive_FDR_under_0.05"],
    default="no_positive_FDR_significant_association",
)
_merged123["human_brain_EC_RNA_lane"] = np.select(
    [~_merged123.human_EC_transcript_assessable.fillna(False).astype(bool),
     pd.to_numeric(_merged123.unsorted_median_CPM, errors="coerce").gt(0)],
    ["unassessable", "donor_expression_detected"],
    default="assessed_zero_or_missing",
)
_merged123["specificity_lane"] = np.where(
    _merged123.human_EC_specificity_assessable.fillna(False).astype(bool),
    _merged123.human_EC_specificity_direction.fillna("assessed_unresolved"),
    "unassessable",
)
_merged123["structural_lane"] = np.where(
    _merged123.topology_both_species_available.fillna(False).astype(bool),
    "canonical_topology_coordinates_available_not_fragment_approved",
    "topology_unresolved_not_fragment_approved",
)
_merged123["literature_lane"] = np.where(
    _merged123.primary_literature_review_pending.fillna(True).astype(bool),
    "targeted_primary_review_pending", "targeted_review_or_reference_present_claims_still_need_check",
)
_merged123["peripheral_evidence_lane"] = np.where(
    pd.to_numeric(_merged123.HPA_peripheral_EC_cluster_n_observed, errors="coerce").fillna(0).gt(0),
    "peripheral_EC_RNA_assessed_not_binding_risk", "peripheral_EC_RNA_unassessable",
)
_merged123["design_readiness"] = "not_approved_native_luminal_epitope_and_fragment_review_pending"
_merged123["generated_peptide_BBB_transport_validated"] = False
_merged123["final_receptor_selected"] = False
_merged123["fragment_approved_by_section_12_3"] = False
_merged123["hypothesis_comparison_only"] = True

# 4. Manual review needs: capture blockers without assuming a negative biological result.
def _gaps123(row):
    outstanding = list(json.loads(row["comparative_review_gaps_json"]))
    if not row["manual_status_valid"]:
        outstanding.append("correct_manual_review_status")
    if not row["manual_role_value_valid"]:
        outstanding.append("correct_manual_intended_role")
    if not row["manual_modeling_field_valid"]:
        outstanding.append("correct_manual_modeling_field")
    if row["manual_review_completed"] and not row["complete_record_documented"]:
        outstanding.append("completed_review_requires_evidence_reviewer_and_date")
    if row["manual_modeling_requested"]:
        outstanding.append("modeling_requested_does_not_approve_docking_fragment")
        if not row["modeling_request_has_coordinates"]:
            outstanding.append("provide_manual_model_fragment_coordinates")
    return json.dumps(list(dict.fromkeys(outstanding)), ensure_ascii=False)

_merged123["remaining_decision_questions_json"] = _merged123.apply(_gaps123, axis=1)
_merged123["n_remaining_decision_questions"] = _merged123.remaining_decision_questions_json.map(
    lambda x: len(json.loads(x)))

# 5. Compact relationship-specific comparison register, plus full manual state.
_register_cols123 = [
    "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id",
    "comparison_discussion_lane", "is_data_driven_candidate", "is_curated_reference",
    "mouse_D02_positive_DE_evidence", "human_brain_EC_RNA_lane", "specificity_lane",
    "structural_lane", "literature_lane", "peripheral_evidence_lane",
    "delivery_evidence_family", "BBB_transcytosis_evidence", "primary_claim_reconciliation",
    "manual_review_status", "manual_status_valid", "intended_role", "manual_role_value_valid",
    "selected_for_structural_modeling", "complete_record_documented", "needs_manual_correction",
    "design_readiness", "generated_peptide_BBB_transport_validated", "final_receptor_selected",
    "remaining_decision_questions_json",
]
_register123 = _merged123[_register_cols123].copy()
_focus123 = _register123.loc[
    _register123.comparison_discussion_lane.ne("full_universe_retained_not_a_focal_example")
].copy()

# 6. Proposed assay/control specification (planning, not execution/results).
_control_rows123 = [
    ("P01", "published_peptide_reference", "Angiopep-2", "LRP1", "Published ligand-specific peptide reference; confirm receptor and assay context", "Intact translocation + barrier integrity", "Reference peptide transport not yet measured in this workflow"),
    ("P02", "published_peptide_reference_conditional", "L57", "LRP1", "Confirm sequence, ligand mechanism and published intact-delivery endpoint", "Same quantitative assay as generated peptide", "Verify direct source before use"),
    ("P03", "published_peptide_reference_conditional", "T7", "TFRC", "Verify sequence, human receptor affinity and transport endpoint", "Same quantitative assay as generated peptide", "Binding alone is not BBB crossing"),
    ("P04", "delivery_reference_different_format", "Receptor-targeting antibody", "TFRC / INSR / IGF1R / CD98hc", "Literature context only unless antibody and peptide formats compared carefully", "Receptor dependence and brain delivery", "Do not infer peptide performance from antibody delivery"),
    ("N01", "sequence_negative_control", "Composition-matched scrambled peptide", "Matched exploratory receptor", "Match length, residue composition, charge and assay matrix", "Intact transport and endothelial retention", "Scrambling does not guarantee loss of receptor affinity"),
    ("N02", "sequence_negative_control", "Length/charge-matched unrelated peptide", "Matched exploratory receptor", "Specify sequence and physicochemical properties", "Matched transport assay", "Negative binding must be checked"),
    ("N03", "receptor_specificity_control", "Wrong receptor / competing ligand", "Exploratory receptor", "Predefine the noncognate receptor or competition dose", "Binding, uptake, intact transport", "Receptor expression and perturbation effects measured"),
    ("N04", "receptor_dependency_control", "Genetic KD/KO or receptor blocking", "Exploratory receptor", "Independent perturbation with viability and barrier readouts", "Receptor-dependent uptake and transport", "Barrier perturbation is not evidence of transcytosis"),
    ("Q01", "barrier_quality_control", "TEER + paracellular tracer", "BBB model", "Predefine qualification/acceptance windows", "Barrier integrity during peptide assay", "Leakage cannot be counted as receptor-mediated shuttle transport"),
    ("Q02", "intact_cargo_analytical_control", "Intact peptide quantification", "BBB model", "LC-MS or qualified molecular assay, with recovery and degradation controls", "Intact luminal, cellular and abluminal amounts", "Reporter-only signal may not reflect intact peptide"),
    ("Q03", "later_in_vivo_validation", "Vascular washout and capillary depletion", "Brain parenchyma", "Conditional follow-up after quantitative in vitro evidence", "Extravascular intact cargo and tissue localization", "Total brain signal can reflect vascular binding"),
]
_control123 = pd.DataFrame(_control_rows123, columns=[
    "control_id", "role", "suggested_control", "target_or_context", "qualification_needed",
    "matched_endpoint", "interpretation_limit",
])
_control123["planned_only"] = True

# 7. Summarize all relationships, without selecting a 'best' human orthologue.
_gene_queue123 = _register123.groupby("gene", sort=True).agg(
    n_human_relationship_rows=("relationship_id", "size"),
    n_manual_review_completed=("complete_record_documented", "sum"),
    n_manual_corrections_needed=("needs_manual_correction", "sum"),
    n_manual_modeling_requests=("selected_for_structural_modeling", lambda s: int(s.astype(str).str.lower().isin({"true", "1"}).sum())),
).reset_index()
_gene_queue123["review_queue_status"] = np.where(
    _gene_queue123.n_manual_review_completed.eq(_gene_queue123.n_human_relationship_rows),
    "all_relationships_documented_not_final_selection",
    "manual_adjudication_pending",
)

_assert_ids123 = set(_comparison123.relationship_id)
assert len(_register123) == len(_comparison123) == 125
assert _register123.gene.nunique() == 114 and _register123.relationship_id.is_unique
assert set(_register123.relationship_id) == _assert_ids123
assert _gene_queue123.n_human_relationship_rows.sum() == 125
assert not _register123.final_receptor_selected.any()
assert not _register123.generated_peptide_BBB_transport_validated.any()
assert not _merged123.fragment_approved_by_section_12_3.any()
assert _manual_hash_before123 == _hash123(_manual_path123), "Manual adjudication file unexpectedly changed"

_outputs123 = {
    "EC_receptor_12_3_decision_acceptance_rules_v1": _gate_table123,
    "EC_receptor_12_3_relationship_decision_register_v1": _register123,
    "EC_receptor_12_3_full_manual_evidence_audit_v1": _merged123,
    "EC_receptor_12_3_focal_review_packet_v1": _focus123,
    "EC_receptor_12_3_gene_review_queue_v1": _gene_queue123,
    "EC_receptor_12_3_reference_and_negative_control_plan_v1": _control123,
}
for _stem123, _table123 in _outputs123.items():
    _table123.to_csv(_dir123 / (_stem123 + ".csv"), index=False)

_summary123 = pd.DataFrame([
    ("structural_scope_mouse_genes", 114),
    ("structural_scope_relationships", len(_register123)),
    ("focal_review_relationships", len(_focus123)),
    ("manual_rows_marked_completed_with_nonempty_fields_not_claim_verified", int(_merged123.complete_record_documented.sum())),
    ("manual_entries_needing_correction", int(_merged123.needs_manual_correction.sum())),
    ("relationship_rows_pending_primary_literature", int(_comparison123.primary_literature_review_pending.fillna(True).sum())),
    ("human_EC_transcript_assessable", int(_comparison123.human_EC_transcript_assessable.fillna(False).sum())),
    ("mouse_human_topology_both_available", int(_comparison123.topology_both_species_available.fillna(False).sum())),
    ("experimental_control_plan_rows", len(_control123)),
    ("approved_docking_fragments", 0),
    ("scientifically_finalized_primary_receptors", 0),
], columns=["metric", "value"])
_summary123.to_csv(_dir123 / "EC_receptor_12_3_selection_gate_summary_v1.csv", index=False)

_manifest123 = {
    "version": DECISION_GATE_VERSION,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "comparison_input": _comparison_path123.name,
    "comparison_input_sha256": _hash123(_comparison_path123),
    "manual_adjudication_input": _manual_path123.name,
    "manual_adjudication_sha256": _manual_hash_before123,
    "manual_adjudication_modified": False,
    "n_decision_questions": len(_gate_table123),
    "n_relationships": len(_register123),
    "n_mouse_genes": _register123.gene.nunique(),
    "output_stage": "selection_criteria_and_review_register_not_scientific_selection",
    "no_composite_scoring": True,
    "no_automatic_approval_or_exclusion": True,
    "no_unreviewed_literature_claim_promoted": True,
    "manual_field_completeness_is_not_scientific_claim_verification": True,
    "separate_optimization_from_independent_evaluation": True,
}
(_dir123 / "EC_receptor_12_3_decision_manifest_v1.json").write_text(
    json.dumps(_manifest123, indent=2), encoding="utf-8")

print("\nSection 12.3 receptor decision framework:")
if "display" in globals():
    display(_summary123)
    display(_focus123[[
        "gene", "human_gene_symbol", "comparison_discussion_lane",
        "mouse_D02_positive_DE_evidence", "human_brain_EC_RNA_lane",
        "specificity_lane", "structural_lane", "manual_review_status",
    ]])
else:
    print(_summary123.to_string(index=False))
    print(_focus123[["gene", "human_gene_symbol", "comparison_discussion_lane"]].to_string(index=False))
print("\nManual adjudication template preserved unchanged.")
print("No receptor scored, excluded, selected or approved for docking.")
