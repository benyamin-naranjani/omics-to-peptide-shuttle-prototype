OmiPep Module01 12.4k | TM4SF1 fulltext pass | 2026-11-22

This package is an additive append-only derivative of the previous PROCR v1 bundle.

Reviewed source keys and DOI:
  Lin2014_TM4SF1_glycan_species_epitope 10.1007/s10456-014-9437-2 | PDF s10456-014-9437-2.pdf
  Sciuto2015_TM4SF1_8G4_nuclear 10.1016/j.bbrc.2015.07.142 | PDF j.bbrc.2015.07.142.pdf
  Lin2024_TM4SF1_TMED_cargo 10.1002/ccs3.12031 | PDF ccs3.12031.pdf
  Lin2026_TM4SF1_ADC_angiogenic_vessels 10.3390/ijms27104437 | PDF ijms-27-04437-v3.pdf

DATA CONTRACT
- fulltext_paper_ledger_v2 contains unchanged 3 PROCR rows plus 4 new TM4SF1 rows.
- TM4SF1_experiment_ledger_v1: 31 source-specific readouts.
- cumulative_experiment_ledger_v1: 16 unchanged PROCR + 31 TM4SF1 readouts.
- shortlist_WORKING_v4: exactly 6 relationship rows, original 26 columns; only TM4SF1 record updated from v3.
- manuscript_comparison_PROVISIONAL_v2: exactly 6 rows x 8 columns, only TM4SF1 row updated.
- followup_TO_VERIFY: 4 additional leads, NOT represented as reviewed primary papers.
- All PDFs saved under data/processed/review_sources/TM4SF1/.
- No changes to 125-row decision table, original master, gene-expression, orthology, or PROCR files.
- This package does NOT approve TM4SF1; native brain luminal location, short-peptide BBB transcytosis and functional safety remain unproven.
- Appendix note: repeated measured off-organ bulk and EC nTPM are not directly comparable to the brain EC CPM.

INTERPRETATION CHECKPOINTS
1. Lin 2014: human 8G4 epitope is glycosylated EL2; antibody lacks murine cross-reactivity.
2. Sciuto 2015: dynamin-dependent uptake/near nuclear signal is intracellular not abluminal release.
3. Lin 2024: TMED signaling complex trafficking and wound-healing deficit on genetic reduction, not an externally dosed peptide.
4. Lin 2026: mouse ear luminal targeting; systemic lethality at high ADC doses; naked antibody and ADC are not equivalent; not BBB.

INSTALL
conda activate ml-gpu
cd ~/projects/omipep-omics-human-v0.2
S=/mnt/c/Users/bnara/Downloads
python -m zipfile -e "$S/OmiPep_M01_12_4k_TM4SF1_INSTALL_bundle_v1.zip" .
python notebooks/12_4k_verify_TM4SF1_review.py data/processed/review_outputs

After new per-receptor packages, always use highest cumulative ledger and working shortlist version; never sum or concatenate duplicate cumulative files together.
