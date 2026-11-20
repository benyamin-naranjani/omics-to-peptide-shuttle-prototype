"""OmiPep v0.2 — Notebook cell 12.4: conditional receptor shortlist and manuscript decision packet.

Run from notebooks/ after 12.3:
    %run -i 12_4_provisional_shortlist_and_manuscript_table.py

Creates a source-linked, human-orthologue-specific comparison for all 125 structural-
scope relationships, with an explicitly PROVISIONAL three-receptor discussion agenda,
10 alternative review examples and 9 distinct reference/control contexts.

Does NOT automatically select, reject, score, approve docking fragments or claim
transport of generated peptides.  Never rewrites the 12.1 manual adjudication sheet.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

SHORTLIST_PACKET_VERSION_124 = "1.0"
output_124 = Path(out_dir)
output_124.mkdir(parents=True, exist_ok=True)

files_124 = {
    "comparison": output_124 / "EC_receptor_comparative_adjudication_full_v1.csv",
    "decision": output_124 / "EC_receptor_12_3_relationship_decision_register_v1.csv",
    "manual": output_124 / "EC_receptor_manual_adjudication_template_v1.csv",
    "rules": output_124 / "EC_receptor_12_3_decision_acceptance_rules_v1.csv",
    "controls": output_124 / "EC_receptor_12_3_reference_and_negative_control_plan_v1.csv",
}
for name_124, path_124 in files_124.items():
    if not path_124.is_file():
        raise FileNotFoundError(f"Required 12.3 input missing ({name_124}): {path_124}")
sha_124 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manual_hash_124 = sha_124(files_124["manual"])

read_types_124 = {"relationship_id":"string", "gene":"string", "human_ensembl_gene_id":"string"}
comparison_124 = pd.read_csv(files_124["comparison"], dtype=read_types_124, low_memory=False)
decision_124 = pd.read_csv(files_124["decision"], dtype=read_types_124, low_memory=False)
manual_124 = pd.read_csv(files_124["manual"], dtype=str, keep_default_na=False)
rules_124 = pd.read_csv(files_124["rules"], dtype=str, keep_default_na=False)
controls_124 = pd.read_csv(files_124["controls"], dtype=str, keep_default_na=False)

required_comparison_124 = {
    "relationship_id", "gene", "human_ensembl_gene_id", "human_gene_symbol", "comparison_discussion_lane",
    "is_data_driven_candidate", "is_curated_reference", "D02_vs_Sham_logFC", "D02_vs_Sham_FDR",
    "D14_vs_Sham_logFC", "D14_vs_Sham_FDR", "mouse_surface_class", "external_identity_pct",
    "human_orthology_evidence_lane", "human_EC_transcript_assessable", "human_EC_specificity_assessable",
    "unsorted_median_CPM", "EC_vs_non_EC_median_paired_log2_ratio", "topology_both_species_available",
    "delivery_evidence_family", "BBB_transcytosis_evidence", "primary_literature_review_pending",
    "primary_claim_reconciliation", "human_template_context", "HPA_peripheral_EC_cluster_max_nTPM",
    "candidate_score", "scientific_selection_status", "fragment_approved",
}
required_decision_124 = {
    "relationship_id", "gene", "human_ensembl_gene_id", "manual_review_status", "intended_role",
    "complete_record_documented", "needs_manual_correction", "remaining_decision_questions_json",
    "generated_peptide_BBB_transport_validated", "final_receptor_selected", "design_readiness",
}
for obj_124, required_124, label_124 in (
    (comparison_124, required_comparison_124, "12.2 comparative evidence"),
    (decision_124, required_decision_124, "12.3 decision register"),
):
    missing_124 = sorted(required_124 - set(obj_124.columns))
    if missing_124:
        raise RuntimeError(f"{label_124} has missing columns: {missing_124}")
# IHC/immune annotations can be absent in a scoped export; unknown is not zero.
if "HPA_immune_cell_max_nTPM" not in comparison_124.columns:
    comparison_124["HPA_immune_cell_max_nTPM"] = np.nan
if not {"relationship_id", "gene", "human_ensembl_gene_id"}.issubset(manual_124.columns):
    raise RuntimeError("12.1 manual adjudication schema changed")
if not {"gate_id", "acceptance_question"}.issubset(rules_124.columns):
    raise RuntimeError("12.3 gate schema changed")
if not {"control_id", "role", "suggested_control"}.issubset(controls_124.columns):
    raise RuntimeError("12.3 control schema changed")

for obj_124, label_124 in ((comparison_124, "comparison"), (decision_124, "decision")):
    if len(obj_124) != 125 or obj_124.gene.nunique() != 114:
        raise RuntimeError(f"Unexpected {label_124} universe (expected 125/114); investigate rather than trim")
    if obj_124.relationship_id.isna().any() or obj_124.relationship_id.duplicated().any():
        raise RuntimeError(f"Invalid {label_124} relationship identifiers")

# The stable relationship ID, mouse gene and human ENSG must all agree across stages.
relation_keys_124 = ["relationship_id", "gene", "human_ensembl_gene_id"]
base_ids_124 = comparison_124[relation_keys_124].fillna("").astype(str)
for label_124, obj_124 in (("12.3", decision_124), ("manual", manual_124)):
    actual_124 = obj_124[relation_keys_124].fillna("").astype(str)
    if actual_124.relationship_id.duplicated().any():
        raise RuntimeError(f"Duplicate relationship IDs in {label_124}")
    verification_124 = base_ids_124.merge(actual_124, on="relationship_id", how="outer",
                                          suffixes=("_base", "_other"), validate="one_to_one", indicator=True)
    if (not verification_124._merge.eq("both").all()
        or not verification_124.gene_base.eq(verification_124.gene_other).all()
        or not verification_124.human_ensembl_gene_id_base.eq(verification_124.human_ensembl_gene_id_other).all()):
        raise RuntimeError(f"Conflicting gene-to-human identity in {label_124}")

if comparison_124.candidate_score.notna().any():
    raise RuntimeError("Upstream score appeared unexpectedly: revise protocol before continuing")
if not comparison_124.scientific_selection_status.eq("not_selected_manual_adjudication_pending").all():
    raise RuntimeError("Upstream receptor status changed; do not overwrite a scientific decision")

def yes_124(col):
    mapped_124 = col.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}).astype("boolean")
    if (col.notna() & mapped_124.isna()).any():
        raise ValueError(f"Unrecognized boolean value(s) in {col.name}")
    return mapped_124

for col_124 in ("fragment_approved", "human_EC_transcript_assessable", "human_EC_specificity_assessable",
                "topology_both_species_available", "primary_literature_review_pending"):
    comparison_124[col_124] = yes_124(comparison_124[col_124])
for col_124 in ("complete_record_documented", "needs_manual_correction",
                "generated_peptide_BBB_transport_validated", "final_receptor_selected"):
    decision_124[col_124] = yes_124(decision_124[col_124])
if (comparison_124.fragment_approved.fillna(False).any()
    or decision_124.final_receptor_selected.fillna(False).any()
    or decision_124.generated_peptide_BBB_transport_validated.fillna(False).any()):
    raise RuntimeError("An approval or transport claim is present. Reconcile, do not silently continue")

# Reuse the exact, already established review families. They are DISCUSSION lanes, not ranks.
EXPLORATORY_HYPOTHESES_124 = ("Ecscr", "Tm4sf1", "Procr")
ALTERNATIVE_HYPOTHESES_124 = (
    "Ace", "Selp", "Ackr1", "Cd93", "Bst2", "Itgb1", "Il4ra", "Cd200", "Tnfrsf1a", "Mcam")
REFERENCE_CONTEXT_124 = ("Lrp1", "Tfrc", "Insr", "Igf1r", "Slc3a2", "Ly6a", "Ly6e", "Car4", "Itm2a")
focus_genes_124 = (*EXPLORATORY_HYPOTHESES_124, *ALTERNATIVE_HYPOTHESES_124, *REFERENCE_CONTEXT_124)
if len(set(focus_genes_124)) != len(focus_genes_124):
    raise RuntimeError("Overlapping focus classifications")
if set(focus_genes_124) - set(comparison_124.gene.dropna()):
    raise RuntimeError("One or more focal examples were lost from the comparison universe")

# Preserve all 125 relationships, including multiple possible human genes.
extra_decision_124 = ["relationship_id", "manual_review_status", "intended_role", "complete_record_documented",
                      "needs_manual_correction", "remaining_decision_questions_json", "design_readiness"]
dossier_124 = comparison_124.merge(decision_124[extra_decision_124], on="relationship_id",
                                   how="left", validate="one_to_one")
if len(dossier_124) != 125 or dossier_124["remaining_decision_questions_json"].isna().any():
    raise RuntimeError("Incomplete 12.3 decision-register linkage")

# Define simple, noncompensatory evidence descriptors. These are NOT a receptor score.
d02_fc_124 = pd.to_numeric(dossier_124.D02_vs_Sham_logFC, errors="coerce")
d02_fdr_124 = pd.to_numeric(dossier_124.D02_vs_Sham_FDR, errors="coerce")
d14_fc_124 = pd.to_numeric(dossier_124.D14_vs_Sham_logFC, errors="coerce")
d14_fdr_124 = pd.to_numeric(dossier_124.D14_vs_Sham_FDR, errors="coerce")

stage_fields_124 = pd.DataFrame(index=dossier_124.index)
stage_fields_124["D02_stroke_discovery_context"] = np.select(
    [d02_fdr_124.isna(), d02_fdr_124.lt(.05) & d02_fc_124.gt(0)],
    ["not_tested", "positive_FDR_significant"], default="no_positive_FDR_significant_association")
stage_fields_124["D14_stroke_discovery_context"] = np.select(
    [d14_fdr_124.isna(), d14_fdr_124.lt(.05) & d14_fc_124.gt(0)],
    ["not_tested", "positive_FDR_significant"], default="no_positive_FDR_significant_association")
stage_fields_124["adult_human_EC_RNA_context"] = np.select(
    [~dossier_124.human_EC_transcript_assessable.fillna(False).astype(bool),
     pd.to_numeric(dossier_124.unsorted_median_CPM, errors="coerce").gt(0)],
    ["unassessable", "detected_in_adult_control_EC"], default="assessed_zero_or_missing")
stage_fields_124["human_EC_specificity_context"] = np.select(
    [~dossier_124.human_EC_specificity_assessable.fillna(False).astype(bool),
     pd.to_numeric(dossier_124.EC_vs_non_EC_median_paired_log2_ratio, errors="coerce").gt(0),
     pd.to_numeric(dossier_124.EC_vs_non_EC_median_paired_log2_ratio, errors="coerce").lt(0)],
    ["unassessable", "EC_higher_descriptive", "non_EC_higher_descriptive"], default="comparable_or_neutral")
stage_fields_124["canonical_external_topology_context"] = np.where(
    dossier_124.topology_both_species_available.fillna(False).astype(bool),
    "topology_coordinates_both_species_not_luminal_approval", "topology_incomplete")
stage_fields_124["primary_literature_context"] = np.where(
    dossier_124.primary_literature_review_pending.fillna(True).astype(bool),
    "primary_review_pending", "targeted_review_recorded_not_claim_verified")
stage_fields_124["peripheral_RNA_caveat"] = "HPA_RNA_not_ligand_binding_or_safety"
stage_fields_124["transcytosis_caveat"] = "published_ligand_format_specific_no_generated_peptide_transport"
stage_fields_124["docking_fragment_status"] = "not_approved"
stage_fields_124["final_selection_status"] = "not_selected"
stage_fields_124["comparison_is_not_merit_ranking"] = True

# 12.1 already defines docking_fragment_status as an upstream review status.
# Preserve that provenance explicitly; 12.4 creates a distinct approval status.
if "docking_fragment_status" in dossier_124.columns:
    original_fragment_state_124 = dossier_124["docking_fragment_status"].astype("string")
    allowed_unapproved_states_124 = {
        "none_approved_pending_manual_structure_review", "not_approved"
    }
    if not original_fragment_state_124.isin(allowed_unapproved_states_124).all():
        raise RuntimeError(
            "Unexpected upstream fragment status; reconcile approvals before continuing"
        )
    dossier_124 = dossier_124.rename(columns={
        "docking_fragment_status": "upstream_docking_fragment_status"
    })

# Protect against every other accidental column collision, rather than silently
# overwriting existing provenance or creating duplicate column names.
shared_stage_columns_124 = dossier_124.columns.intersection(stage_fields_124.columns)
if len(shared_stage_columns_124):
    raise RuntimeError(
        f"Conflicting 12.4 stage column names: {list(shared_stage_columns_124)}"
    )
dossier_124 = pd.concat([dossier_124.copy(), stage_fields_124], axis=1).copy()
if not dossier_124.columns.is_unique:
    raise RuntimeError("Duplicate columns in 12.4 dossier")

role_map_124 = {
    "provisional_exploratory_deep_review_example": "provisional_exploratory_deep_review_not_selected",
    "alternative_exploratory_review": "alternative_exploratory_review_not_excluded",
    "reference_or_negative_control_review": "reference_or_comparator_context_not_design_approval",
    "full_universe_retained_not_a_focal_example": "full_universe_retained_not_excluded",
}
dossier_124["manuscript_discussion_role"] = dossier_124.comparison_discussion_lane.map(role_map_124)
if dossier_124.manuscript_discussion_role.isna().any():
    raise RuntimeError("Unknown 12.2 comparison lane; do not silently change roles")

# Additional manuscript language cannot exceed evidence (especially native-ligand vs peptide transport).
shortlist_narratives_124 = {
    "Ecscr": (
        "Mouse D02-associated endothelial transcript; human brain EC RNA and donor-paired enrichment; "
        "proposed extracellular epitope requires native luminal confirmation.",
        "Extracellular conservation is modest; peripheral endothelial RNA is observed; "
        "ECSCR-targeted cargo uptake and intact blood-to-brain transfer remain unestablished."),
    "Tm4sf1": (
        "Mouse D02-associated transcript; substantial human EC RNA and enrichment; "
        "reported peripheral vascular EC antibody uptake provides ligand-format-specific context.",
        "Four-pass receptor with short extracellular loops and substantial peripheral endothelial RNA; "
        "native membrane/glycan architecture and abluminal cargo release unverified."),
    "Procr": (
        "Mouse D02-associated transcript; adult human brain EC RNA; published native activated "
        "protein C transport context in a mouse BBB model.",
        "Native APC transport does not demonstrate peptide transport; peripheral vascular binding, "
        "protein C/coagulation signalling and native epitope availability require review."),
}
reference_notes_124 = {
    "Lrp1":"Angiopep-2/L57 peptide delivery references; epitope, receptor polarity and assay claims require source matching.",
    "Tfrc":"Receptor-mediated antibody delivery context; T7 peptide binding/transport claims need independent endpoint verification.",
    "Insr":"Species/ligand-specific antibody delivery reference; insulin pathway perturbation must be assessed.",
    "Igf1r":"Antibody-derived brain delivery reference; not automatically applicable to generated peptides.",
    "Slc3a2":"CD98hc antibody brain delivery context; transporter heterocomplex/ligand effects require review.",
    "Ly6a":"Mouse AAV capsid-delivery reference only; no established equivalent human peptide mechanism.",
    "Ly6e":"AAV crossing context; not an independently demonstrated standalone peptide shuttle.",
    "Car4":"Engineered AAV/receptor context; native ligand/cargo and enzyme function require review.",
    "Itm2a":"Human EC-enriched receptor reference with negative/inconclusive prior antibody transport evidence.",
}

focus_124 = dossier_124.loc[dossier_124.gene.isin(focus_genes_124)].copy()
focus_124["discussion_order"] = focus_124.gene.map({gene:i for i,gene in enumerate(focus_genes_124)})
focus_124 = focus_124.sort_values(["discussion_order", "gene", "human_ensembl_gene_id"], kind="stable")

provisional_124 = focus_124.loc[focus_124.gene.isin(EXPLORATORY_HYPOTHESES_124)].copy()
provisional_124["evidence_based_rationale"] = provisional_124.gene.map(lambda g: shortlist_narratives_124[str(g)][0])
provisional_124["key_scientific_uncertainty"] = provisional_124.gene.map(lambda g: shortlist_narratives_124[str(g)][1])
provisional_124["not_final_selection"] = True
alternatives_124 = focus_124.loc[focus_124.gene.isin(ALTERNATIVE_HYPOTHESES_124)].copy()
reference_124 = focus_124.loc[focus_124.gene.isin(REFERENCE_CONTEXT_124)].copy()
reference_124["benchmark_context_not_peptide_validation"] = reference_124.gene.map(reference_notes_124)

# A compact view for downstream scientific review and the manuscript draft.
publication_columns_124 = [
    "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id", "manuscript_discussion_role",
    "is_data_driven_candidate", "is_curated_reference", "D02_vs_Sham_logFC", "D02_vs_Sham_FDR",
    "D14_vs_Sham_logFC", "D14_vs_Sham_FDR", "external_identity_pct", "unsorted_median_CPM",
    "EC_vs_non_EC_median_paired_log2_ratio", "mouse_surface_class", "human_orthology_evidence_lane",
    "human_template_context", "delivery_evidence_family", "BBB_transcytosis_evidence",
    "HPA_peripheral_EC_cluster_max_nTPM", "HPA_immune_cell_max_nTPM", "primary_claim_reconciliation",
    "primary_literature_context", "manual_review_status", "intended_role", "docking_fragment_status",
    "final_selection_status", "remaining_decision_questions_json",
]
publication_124 = focus_124[publication_columns_124].copy()
publication_124["noncompensatory_evidence_policy"] = "No composite score; missing data do not mean biological absence"
publication_124["manuscript_claim_stage"] = "comparative_evidence_only_not_final_receptor_selection"

# Offer a draft paragraph that reflects the current evidence, not an unpublished transport result.
markdown_path_124 = output_124 / "EC_receptor_12_4_manuscript_decision_table_v1.md"
def fmt_124(value, digits=2):
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"

header_124 = [
    "# Provisional receptor-comparison packet — OmiPep v0.2",
    "",
    "> **NOT a final receptor selection or docking authorization.** All receptor roles are discussion-only,",
    "> primary-source endpoint checks remain incomplete, and no generated peptide has demonstrated BBB transport.",
    "",
    "## Three exploratory hypotheses retained for focused adjudication",
    "",
    "| Mouse gene | Human orthologue (stable ID) | D02 log2FC (FDR) | Human EC median CPM | EC/non-EC log2 ratio | Human extracellular identity | Core caveat |",
    "|---|---|---:|---:|---:|---:|---|",
]
for row_124 in provisional_124.to_dict("records"):
    gene_124 = str(row_124["gene"])
    caveat_124 = shortlist_narratives_124[gene_124][1].replace("|", ";")
    human_124 = str(row_124["human_gene_symbol"])
    if pd.isna(row_124["human_gene_symbol"]):
        human_124 = "NA"
    header_124.append(
        f"| {gene_124} | {human_124} (`{row_124['human_ensembl_gene_id']}`) | "
        f"{fmt_124(row_124['D02_vs_Sham_logFC'])} ({fmt_124(row_124['D02_vs_Sham_FDR'],4)}) | "
        f"{fmt_124(row_124['unsorted_median_CPM'])} | "
        f"{fmt_124(row_124['EC_vs_non_EC_median_paired_log2_ratio'])} | "
        f"{fmt_124(row_124['external_identity_pct'])}% | {caveat_124} |"
    )
header_124 += [
    "",
    "The comparison retained ECSCR, TM4SF1, and PROCR as **provisional review hypotheses** alongside ten",
    "alternative exploratory receptors and nine delivery or biological reference contexts. These groups",
    "were selected to organize evidence review rather than by a learned/composite biological-merit score.",
    "Human transcriptomic measurements were from adult temporal-lobe control donors; systemic Human Protein",
    "Atlas expression was assessed separately, without cross-platform brain-to-periphery expression ratios.",
    "Neither extracellular topology nor available structural templates establish a native luminal epitope.",
    "Antibody, native-ligand and viral-vector transport reports are not evidence of intact transport",
    "of de novo generated peptides. Manual adjudication and target-specific structural approval are pending.",
    "",
    "## Proposed independent-evaluation controls (planned, not yet tested)",
    "",
    "Positive references: Angiopep-2 and conditionally L57–LRP1 and T7–TFRC, following verification of",
    "published sequence, species, receptor dependence and transport endpoints. Negative controls:",
    "composition-matched scrambled peptides, unrelated length/charge-matched peptides, noncognate receptor",
    "or receptor blockade, with intact-peptide quantification and barrier-integrity controls on identical assays.",
    "",
    "## Claims not permitted at this stage",
    "",
    "No receptor has been scientifically finalized or approved for docking; no extracellular fragment",
    "has passed native glycan/disulfide/membrane/oligomeric structural review; no generated peptide",
    "has demonstrated receptor-dependent intact transcytosis or extravascular brain parenchymal delivery.",
    "",
]
markdown_content_124 = "\n".join(header_124)

# Explicit blocked downstream hand-off, preventing accidental use as an approved target list.
blocked_handoff_124 = {
    "version": SHORTLIST_PACKET_VERSION_124,
    "status": "blocked_pending_manual_target_and_fragment_approval",
    "approved_receptors": [],
    "approved_receptor_fragments": [],
    "provisional_review_genes_not_approved": list(EXPLORATORY_HYPOTHESES_124),
    "manual_approval_source": files_124["manual"].name,
    "ready_for_colabfold": False,
    "ready_for_peptide_generation_receptor_conditioning": False,
    "generation_and_evaluation_metrics_must_remain_separate": True,
}

# Validate before writing any outputs; a failed audit must not create
# partially updated or misleading manuscript files.
assert len(dossier_124) == 125 and dossier_124.gene.nunique() == 114
assert dossier_124.relationship_id.is_unique
assert set(dossier_124.relationship_id) == set(comparison_124.relationship_id)
assert set(provisional_124.gene) == set(EXPLORATORY_HYPOTHESES_124)
assert set(alternatives_124.gene) == set(ALTERNATIVE_HYPOTHESES_124)
assert set(reference_124.gene) == set(REFERENCE_CONTEXT_124)
assert set(focus_124.gene) == set(focus_genes_124)
assert dossier_124.docking_fragment_status.eq("not_approved").all()
assert dossier_124.final_selection_status.eq("not_selected").all()
assert dossier_124.manuscript_discussion_role.notna().all()
assert dossier_124.columns.is_unique
assert focus_124.columns.is_unique
assert provisional_124.columns.is_unique
assert publication_124.columns.is_unique
assert sha_124(files_124["manual"]) == manual_hash_124

# Export immutable-in-meaning source-linked outputs; user-owned manual sheet is READ ONLY.
exports_124 = {
    "EC_receptor_12_4_full_relationship_docket_v1.csv": dossier_124,
    "EC_receptor_12_4_focal_comparison_v1.csv": focus_124,
    "EC_receptor_12_4_provisional_exploratory_hypotheses_v1.csv": provisional_124,
    "EC_receptor_12_4_alternative_hypotheses_v1.csv": alternatives_124,
    "EC_receptor_12_4_reference_and_context_v1.csv": reference_124,
    "EC_receptor_12_4_manuscript_decision_table_v1.csv": publication_124,
}
for fname_124, table_124 in exports_124.items():
    table_124.to_csv(output_124 / fname_124, index=False)
markdown_path_124.write_text(markdown_content_124, encoding="utf-8")
(output_124 / "EC_receptor_12_4_downstream_handoff_BLOCKED_v1.json").write_text(
    json.dumps(blocked_handoff_124, indent=2), encoding="utf-8")


summary_124 = pd.DataFrame([
    ("full_relationship_rows_retained", len(dossier_124)),
    ("structural_scope_mouse_genes", dossier_124.gene.nunique()),
    ("provisional_exploratory_mouse_genes_not_selected", provisional_124.gene.nunique()),
    ("provisional_exploratory_relationship_rows", len(provisional_124)),
    ("alternative_exploratory_mouse_genes_not_excluded", alternatives_124.gene.nunique()),
    ("reference_or_context_mouse_genes", reference_124.gene.nunique()),
    ("focal_relationship_rows", len(focus_124)),
    ("completed_documented_manual_relationship_rows_not_claim_verified",
     int(decision_124.complete_record_documented.fillna(False).sum())),
    ("manual_entries_requiring_correction", int(decision_124.needs_manual_correction.fillna(False).sum())),
    ("primary_literature_review_pending_relationship_rows",
     int(comparison_124.primary_literature_review_pending.fillna(True).sum())),
    ("approved_receptors", 0),
    ("approved_docking_fragments", 0),
    ("validated_generated_peptide_transport", 0),
], columns=["metric", "value"])
summary_124.to_csv(output_124 / "EC_receptor_12_4_shortlist_packet_summary_v1.csv", index=False)
manifest_124 = {
    "version": SHORTLIST_PACKET_VERSION_124,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "inputs_sha256": {name:sha_124(path) for name,path in files_124.items()},
    "manual_adjudication_untouched": sha_124(files_124["manual"]) == manual_hash_124,
    "provisional_review_examples": list(EXPLORATORY_HYPOTHESES_124),
    "not_a_rank": True,
    "no_composite_score": True,
    "no_auto_exclusion": True,
    "no_final_receptor_selection": True,
    "no_docking_fragment_approval": True,
    "no_generated_peptide_transport_claim": True,
    "preserves_all_human_relationships": True,
    "HPA_brain_EC_matched_ratio_unavailable": True,
    "manuscript_table": markdown_path_124.name,
    "upstream_gate_ids": rules_124.gate_id.tolist(),
    "independent_control_plan_rows": len(controls_124),
    "limitations": [
        "Manual adjudication and primary-publication claim reconciliation remain incomplete",
        "Discussion roles reflect a versioned review agenda, not a final ranking",
        "Human control-brain RNA does not validate native luminal protein or human stroke delivery",
        "Peripheral HPA nTPM measures cannot be directly divided by Wälchli CPM",
        "Native ligand/antibody/AAV transport is not de novo peptide-cargo transport",
        "Structural template existence is not receptor fragment approval",
    ],
}
(output_124 / "EC_receptor_12_4_shortlist_packet_manifest_v1.json").write_text(
    json.dumps(manifest_124, indent=2), encoding="utf-8")

print("\nSection 12.4 — provisional receptor comparison and manuscript packet:")
if "display" in globals():
    display(summary_124)
    display(provisional_124[["gene", "human_gene_symbol", "human_ensembl_gene_id",
                             "D02_vs_Sham_logFC", "D02_vs_Sham_FDR", "unsorted_median_CPM",
                             "EC_vs_non_EC_median_paired_log2_ratio", "external_identity_pct",
                             "manual_review_status", "docking_fragment_status"]].round(3))
else:
    print(summary_124.to_string(index=False))
    print(provisional_124[["gene", "human_gene_symbol", "manual_review_status"]].to_string(index=False))
print("\nManual adjudication sheet preserved unchanged.")
print("All relationships retained; three review hypotheses, zero finalized receptors.")
print("Manuscript Markdown exported; downstream receptor-model handoff explicitly BLOCKED.")
