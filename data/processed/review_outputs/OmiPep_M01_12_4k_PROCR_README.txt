OmiPep v0.2 / Module 01 / Cell 12.4k / PROCR full-text pass 2 / 2026-11-22

CANONICAL IMMUTABLE ORIGINAL SOURCES:
- EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv (125 x 47)
- EC_receptor_M01_12_4k_provisional_shortlist_v2.csv (6 rows)

NEW DERIVED OUTPUTS (keep in data/processed/review_outputs):
1. EC_receptor_M01_12_4k_fulltext_paper_ledger_v1.csv (3 distinct full-text papers)
2. EC_receptor_M01_12_4k_PROCR_experiment_ledger_v1.csv (16 specific experimental endpoints)
3. EC_receptor_M01_12_4k_shortlist_WORKING_v3.csv (six receptor rows, PROCR reassessed)
4. EC_receptor_M01_12_4k_manuscript_comparison_PROVISIONAL_v1.csv (six rows, eight domains; NOT FINAL)
5. EC_receptor_M01_12_4k_PROCR_additional_context_TO_VERIFY_v1.csv (two externally found, NOT full-text-adjudicated leads)
6. EC_receptor_M01_12_4k_PROCR_fulltext_validation_v1.json (input and output hashes)
7. 12_4k_verify_PROCR_review.py (read-only validation)

WORKFLOW:
- Add each new paper as a unique DOI/citation key row in the global fulltext_paper_ledger.
- Add one record per experimentally distinct endpoint to experiment_ledger; DO NOT count them as independent biological replicates.
- After a receptor's pass-2 review, revise only that receptor row in the 6-receptor working table.
- Version the three derived datasets (v2/v3/etc). Never overwrite earlier source data or mark receptors 'approved'.
- Always attach the latest paper ledger and 6-row table to the next conversation to carry research forward without memory-dependent gaps.
- Keep the Google Sheet references deduplicated: all 3 source DOI records were already present.
- The two 2025/2026 external leads were NOT confirmed in the existing Google Sheet and require bibliographic review before insertion.

REVIEW LIMITATIONS:
- APC in Deane: whole-brain intactness biochemical assays and capillary-depletion data are distinct, not combined into direct proof of intact cargo in interstitial space.
- Nayak: non-BBB cell models and FVIIa rather than engineered peptide; basal intactness asserted without shown corresponding gel.
- Domotor: human BEC signalling, not BBB transcytosis; RCR92 preserving a calcium readout does not establish clinical safety.
- Human stroke-native PROCR isoform, luminal display, functional and systemic safety, and a noncompetitive shuttle epitope remain unproven.
