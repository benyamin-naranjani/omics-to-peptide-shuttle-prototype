"""OmiPep v0.2 | Module 01 | Cell 12.4d (revised 2026-11-20).

Evidence-preserving export only. A one-record-per-relationship expert preview and
separate 125-record deep-literature worklist are staged, not adjudicated.

Run in existing notebook kernel, after verified Cell 12.4c:
    %run -i 12_4d_relationship_inventory_and_literature_batch_plan.py

No existing source tables, 12.1 manual decision file, notebook variables, ColabFold
handoff, or manuscript are changed. No receptor is approved. No DOI is invented.
"""

import csv
import hashlib
import json
from pathlib import Path
import pandas as pd

print("\n=== OmiPep M01 | 12.4d REVISED: source-verified 125-row expert table ===")

ROOT_124d = Path(__file__).resolve().parent.parent
PROC_124d = ROOT_124d / "data" / "processed"
FILES_124d = {
    "docket": PROC_124d / "EC_receptor_12_4_full_relationship_docket_v1.csv",
    "summary": PROC_124d / "EC_receptor_12_4_shortlist_packet_summary_v1.csv",
    "manifest": PROC_124d / "EC_receptor_12_4_shortlist_packet_manifest_v1.json",
    "handoff": PROC_124d / "EC_receptor_12_4_downstream_handoff_BLOCKED_v1.json",
    "manual": PROC_124d / "EC_receptor_manual_adjudication_template_v1.csv",
}
for label_124d, path_124d in FILES_124d.items():
    assert path_124d.is_file() and path_124d.stat().st_size, f"Missing {label_124d}: {path_124d}"

def checksum_124d(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

HASHES_124d = {k: checksum_124d(p) for k, p in FILES_124d.items()}
manifest_124d = json.loads(FILES_124d["manifest"].read_text(encoding="utf-8"))
handoff_124d = json.loads(FILES_124d["handoff"].read_text(encoding="utf-8"))
assert manifest_124d["inputs_sha256"]["manual"] == HASHES_124d["manual"]
assert manifest_124d.get("no_final_receptor_selection") is True
assert manifest_124d.get("no_composite_score") is True
assert handoff_124d.get("status", "").startswith("blocked_")
assert handoff_124d.get("approved_receptors") == []
assert handoff_124d.get("approved_receptor_fragments") == []
assert handoff_124d.get("ready_for_colabfold") is False
assert handoff_124d.get("ready_for_peptide_generation_receptor_conditioning") is False

with FILES_124d["docket"].open(encoding="utf-8-sig", newline="") as handle_124d:
    raw_header_124d = next(csv.reader(handle_124d))
assert len(raw_header_124d) == len(set(raw_header_124d)), "Duplicate original CSV columns"
base_124d = pd.read_csv(FILES_124d["docket"], dtype=str, keep_default_na=False,
                         encoding="utf-8-sig", low_memory=False)
assert len(base_124d) == 125 and base_124d["gene"].nunique() == 114
assert base_124d.columns.is_unique
assert base_124d["relationship_id"].is_unique
assert base_124d["relationship_id"].astype(str).str.strip().ne("").all()
assert base_124d["manual_review_status"].eq("pending").all()
assert base_124d["final_selection_status"].eq("not_selected").all()
assert base_124d["docking_fragment_status"].eq("not_approved").all()

EXACT_124d = [
    "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id",
    "manuscript_discussion_role", "D02_vs_Sham_logFC", "D02_vs_Sham_FDR",
    "D02_vs_Sham_LOSO_direction_stability_pct", "unsorted_median_CPM",
    "EC_vs_non_EC_median_paired_log2_ratio", "external_identity_pct",
    "mouse_surface_class", "HPA_peripheral_tissue_max_nTPM",
    "HPA_peripheral_tissue_peak_context", "HPA_peripheral_EC_cluster_max_nTPM",
    "HPA_peripheral_EC_cluster_peak_context", "HPA_immune_cell_max_nTPM",
    "HPA_IHC_max_level", "BBB_transcytosis_evidence", "literature_source_ids",
    "literature_source_urls", "literature_review_status",
    "primary_literature_review_pending", "manual_review_status",
    "docking_fragment_status", "final_selection_status",
]
missing_124d = sorted(set(EXACT_124d) - set(base_124d.columns))
assert not missing_124d, f"Source changed; missing exact previously inspected columns: {missing_124d}"

summary_124d = pd.read_csv(FILES_124d["summary"], dtype=str,
                            keep_default_na=False)
metrics_124d = dict(zip(summary_124d["metric"], summary_124d["value"]))
assert int(metrics_124d["primary_literature_review_pending_relationship_rows"]) == 73
assert int(metrics_124d["completed_documented_manual_relationship_rows_not_claim_verified"]) == 0

pending_strings_124d = base_124d["primary_literature_review_pending"].str.strip().str.casefold()
assert set(pending_strings_124d.unique()) <= {"true", "false"}
base_124d["_pending_review_124d"] = pending_strings_124d.eq("true")
assert int(base_124d["_pending_review_124d"].sum()) == 73

fc_124d = pd.to_numeric(base_124d["D02_vs_Sham_logFC"], errors="coerce")
fdr_124d = pd.to_numeric(base_124d["D02_vs_Sham_FDR"], errors="coerce")
assert ((fdr_124d.dropna() >= 0) & (fdr_124d.dropna() <= 1)).all()
assert not (base_124d["D02_vs_Sham_FDR"].str.strip().ne("") & fdr_124d.isna()).any()
assert not (base_124d["D02_vs_Sham_logFC"].str.strip().ne("") & fc_124d.isna()).any()

# A review ordering, NOT a ranked biological merit score. Neither expression nor
# literature status is an exclusion criterion. Duplicate mouse-gene relationships
# share DE information; the replicate unit remains the biological sample.
base_124d["_fdr_124d"] = fdr_124d
base_124d["_absfc_124d"] = fc_124d.abs()
ordered_124d = base_124d.sort_values(
    ["_fdr_124d", "_absfc_124d", "relationship_id"],
    ascending=[True, False, True], na_position="last", kind="stable"
).copy().reset_index(drop=True)
ordered_124d.insert(0, "preliminary_DE_review_order", range(1, 126))
ordered_124d["review_cohort"] = ordered_124d["_pending_review_124d"].map({
    True: "pending_primary_literature_review_73",
    False: "existing_annotation_requires_primary_source_reconciliation_52",
})
ordered_124d["review_batch"] = ""
for cohort_124d, prefix_124d in [
    ("pending_primary_literature_review_73", "LIT-P"),
    ("existing_annotation_requires_primary_source_reconciliation_52", "LIT-R"),
]:
    matching_ids_124d = ordered_124d.index[ordered_124d["review_cohort"].eq(cohort_124d)]
    for j_124d, rowidx_124d in enumerate(matching_ids_124d):
        ordered_124d.loc[rowidx_124d, "review_batch"] = f"{prefix_124d}{j_124d // 10 + 1:02d}"
assert ordered_124d["review_batch"].str.len().gt(0).all()

# Expert-facing table: a compact overview built from exactly named, audited
# source columns. Every literature judgement below is BLANK until verified from
# the original paper. The prior unverified claim is separately labeled.
expert_map_124d = {
    "preliminary_DE_review_order": "DE_review_order_NOT_merit_rank",
    "relationship_id": "relationship_id",
    "gene": "mouse_gene",
    "human_gene_symbol": "human_gene",
    "human_ensembl_gene_id": "human_ensembl_ID",
    "D02_vs_Sham_logFC": "mouse_D02_logFC",
    "D02_vs_Sham_FDR": "mouse_D02_FDR",
    "D02_vs_Sham_LOSO_direction_stability_pct": "mouse_D02_LOSO_direction_stability_pct",
    "unsorted_median_CPM": "human_brain_EC_median_CPM_atlas",
    "EC_vs_non_EC_median_paired_log2_ratio": "human_EC_nonEC_log2_ratio_atlas",
    "external_identity_pct": "mouse_human_extracellular_identity_pct",
    "mouse_surface_class": "mouse_protein_topology_context",
    "HPA_peripheral_tissue_max_nTPM": "HPA_peripheral_bulk_RNA_max_nTPM",
    "HPA_peripheral_tissue_peak_context": "HPA_peripheral_bulk_peak_tissue",
    "HPA_peripheral_EC_cluster_max_nTPM": "HPA_peripheral_EC_RNA_max_nTPM",
    "HPA_peripheral_EC_cluster_peak_context": "HPA_peripheral_EC_peak_organ_cluster",
    "HPA_immune_cell_max_nTPM": "HPA_immune_RNA_max_nTPM",
    "HPA_IHC_max_level": "HPA_peripheral_IHC_max_level_context_only",
    "BBB_transcytosis_evidence": "UPSTREAM_transport_assertion_UNVERIFIED",
    "review_batch": "review_batch",
}
expert_124d = ordered_124d[list(expert_map_124d)].rename(columns=expert_map_124d).copy()
for name_124d in (
    "verified_native_luminal_protein_access",
    "verified_binding_or_internalization_endpoint",
    "verified_intact_BBB_transcytosis_endpoint",
    "verified_peptide_cargo_abluminal_delivery",
    "supporting_citation_keys",
    "contradictory_citation_keys",
    "n_verified_supporting_papers",
    "n_verified_contradictory_papers",
    "expert_feedback_and_uncertainties",
):
    expert_124d[name_124d] = ""  # NOT zero/not established: awaiting primary review

# Full record-level review worklist: distinct literature claims and source IDs
# can be adjudicated by batch; no evidence state is auto-inferred from abstracts.
worklist_cols_124d = [
    "preliminary_DE_review_order", "review_batch", "review_cohort",
    "relationship_id", "gene", "human_gene_symbol", "human_ensembl_gene_id",
    "literature_source_ids", "literature_source_urls", "literature_review_status",
    "primary_literature_review_pending", "BBB_transcytosis_evidence",
    "manuscript_discussion_role", "HPA_peripheral_tissue_peak_context",
    "HPA_peripheral_EC_cluster_peak_context",
]
review_124d = ordered_124d[worklist_cols_124d].rename(columns={
    "preliminary_DE_review_order":"DE_review_order_NOT_merit_rank",
    "gene":"mouse_gene", "human_gene_symbol":"human_gene",
    "human_ensembl_gene_id":"human_ensembl_ID",
    "literature_source_ids":"UPSTREAM_source_ids_UNVERIFIED",
    "literature_source_urls":"UPSTREAM_source_urls_UNVERIFIED",
    "literature_review_status":"UPSTREAM_literature_annotation_status_UNVERIFIED",
    "BBB_transcytosis_evidence":"UPSTREAM_transport_assertion_UNVERIFIED",
}).copy()
for name_124d in (
    "verified_DOI_list", "verified_PMID_list", "verified_citation_keys",
    "primary_species_and_BBB_model", "transported_material_and_cargo_intactness",
    "primary_endpoint_measured", "native_luminal_protein_access",
    "receptor_binding", "internalization", "receptor_specificity",
    "intact_BBB_transcytosis", "abluminal_parenchymal_release",
    "structural_and_glycan_constraints", "supporting_studies_keys",
    "contradictory_studies_keys", "context_only_studies_keys",
    "verified_positive_paper_count", "verified_negative_paper_count",
    "contradictions_and_limitations", "peripheral_binding_and_safety_context",
    "evidence_strength_and_uncertainty", "expert_comment", "manual_literature_review_complete",
):
    review_124d[name_124d] = "" if name_124d != "manual_literature_review_complete" else "NO"

assert len(expert_124d) == len(review_124d) == 125
assert expert_124d.relationship_id.is_unique and review_124d.relationship_id.is_unique
assert set(expert_124d.relationship_id) == set(base_124d.relationship_id)
assert expert_124d["n_verified_supporting_papers"].eq("").all()
assert expert_124d["verified_intact_BBB_transcytosis_endpoint"].eq("").all()
assert review_124d["manual_literature_review_complete"].eq("NO").all()
assert review_124d["review_batch"].str.startswith("LIT-").all()

# Freeze by version: refuse to overwrite prior edited review work or outputs.
export_paths_124d = {
    "expert": PROC_124d / "EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv",
    "review": PROC_124d / "EC_receptor_12_4d_literature_review_worklist_v2.csv",
    "schema": PROC_124d / "EC_receptor_12_4d_schema_and_provenance_v2.json",
}
from io import StringIO

def csv_bytes_124d(frame):
    buf = StringIO(newline="")
    frame.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")

schema_124d = {
    "version": "2.0", "status": "PRELIMINARY_not_manually_reviewed",
    "relationship_rows": 125, "mouse_genes_distinct": 114,
    "pending_primary_review_records": 73,
    "reconcile_existing_annotation_records": 52,
    "verified_primary_papers_in_this_cell": 0,
    "review_order_method": "D02_vs_Sham_FDR ascending, then absolute D02_vs_Sham_logFC descending; missing values last; relationship_id tie-break; NOT a biological rank",
    "same_gene_effect_not_independent_replication": True,
    "evidence_hierarchy": ["binding", "internalization", "intracellular trafficking", "receptor-mediated intact BBB transcytosis", "abluminal intact cargo delivery"],
    "reference_policy": "One unique paper per normalized DOI in connected Google Sheet; one paper may link multiple relationships, and one relationship may have many papers; claims must have exact endpoints and direction",
    "zero_means_unreviewed": False,
    "HPA_note": "HPA normal systemic nTPM contextual, not measured ligand-binding risk; no cross-platform HPA-vs-atlas ratios",
    "source_dataset_sha256": HASHES_124d,
    "source_columns_used": list(expert_map_124d.keys()),
    "expert_columns": list(expert_124d.columns),
    "review_columns": list(review_124d.columns),
    "batches": {k:int(v) for k,v in review_124d["review_batch"].value_counts().sort_index().items()},
    "receptor_approval": False, "fragment_approval": False,
    "downstream_handoff": "BLOCKED",
}
payloads_124d = {
    "expert": csv_bytes_124d(expert_124d),
    "review": csv_bytes_124d(review_124d),
    "schema": (json.dumps(schema_124d, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
}
for k_124d, p_124d in FILES_124d.items():
    assert checksum_124d(p_124d) == HASHES_124d[k_124d], f"Protected input changed: {k_124d}"
for k_124d, p_124d in export_paths_124d.items():
    if p_124d.exists():
        assert p_124d.read_bytes() == payloads_124d[k_124d], (
            f"Existing output has different content; REFUSE overwrite: {p_124d}"
        )
        print(f"UNCHANGED | {p_124d.relative_to(ROOT_124d)}")
    else:
        p_124d.write_bytes(payloads_124d[k_124d])
        print(f"CREATED | {p_124d.relative_to(ROOT_124d)}")

print(f"PASS | source {len(base_124d)} rows; {base_124d.gene.nunique()} mouse genes; {len(raw_header_124d)} original fields")
print(f"PASS | exact pending flag: 73 to deep-review, 52 to re-adjudicate")
print(f"PASS | expert master: {expert_124d.shape[0]} rows x {expert_124d.shape[1]} columns")
print(f"PASS | evidence worklist: {review_124d.shape[0]} rows x {review_124d.shape[1]} columns")
print("BATCH COUNTS (each <= 10):")
print(review_124d.review_batch.value_counts().sort_index().to_string())
print("\nFIRST 12 REVIEW-ORDER RECORDS (NOT receptor merit ranking):")
print(expert_124d[["DE_review_order_NOT_merit_rank", "mouse_gene", "human_gene", "mouse_D02_logFC", "mouse_D02_FDR", "review_batch"]].head(12).to_string(index=False))
ace2_124d = base_124d["gene"].str.casefold().eq("ace2") | base_124d["human_gene_symbol"].str.casefold().eq("ace2")
print("\nACE2 in frozen 125 relationship universe:", int(ace2_124d.sum()), "records")
print("NOTE | No paper reviewed here; upstream PMIDs and claims are unverified, not proof.")
print("NOTE | No low-expression exclusions; missing evidence is unknown, never zero.")
print("PASS | Manual sheet, verified 12.4 packet and ColabFold BLOCKED handoff preserved.")
print("NEXT | Inspect actual export outputs; then review LIT-P01 primary papers and link verified DOIs.")
