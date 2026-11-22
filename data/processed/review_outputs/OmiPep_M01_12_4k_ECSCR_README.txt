OmiPep Module01 12.4k | ECSCR three-paper full-text review | 2026-11-22

Reviewed: Verma et al 2010 (Blood); Kilari et al 2013 (JBC); Akakabe et al 2013 (Nature Communications).
Alias notice: ECSCR=ECSM2=ARIA, NOT separate receptors; molecular KDR=VEGFR2.

DATA CONTRACT
- Paper ledger v3: ten rows = unchanged PROCR 3 + unchanged TM4SF1 4 + new ECSCR 3.
- ECSCR experiment ledger v1: 33 new figure/table readouts. Cumulative experiment ledger v2: unchanged previous 47 + 33 = 80.
- shortlist WORKING v5: exactly six receptor relationships, 26 columns, only ECSCR row amended from v4.
- manuscript comparison PROVISIONAL v3: exactly six rows, eight columns, only ECSCR row amended from v2.
- followup_TO_VERIFY has outside leads, NOT source-reviewed claims. Source PDFs saved under data/processed/review_sources/ECSCR/.
- No changes to preceding master data, original 125-row decision table or previous source bundles. No target approved.
- File SHA256s and prior-prefix preservation validated by notebook script.

THREE DECISIVE DISTINCTIONS
1. Blood 2010: signaling, vascular migration and surface protrusions, NOT BBB transcytosis.
2. JBC 2013: ECSCR associates with KDR in intracellular HRS compartments, NOT endosomal release of peptide cargo across BBB.
3. Nat Commun 2013: systemic insulin available to skeletal muscle with Ecscr loss (eNOS/perfusion-dependent), NOT ECSCR binding and transporting insulin across BBB.

REVIEWER GATES: human brain luminal epitope, ECSCR-targeted ligand uptake, intact directional BBB flux, no VEGF/PI3K/NO interference, ECD isoform/glycan conservation, peripheral sink selectivity.
IMPORTANT: prior GSE225948 paper mentions Ecscr as D02 EC2 marker; do not claim novel original identification. Integrated prioritization/shuttle design is the paper-specific contribution.
SUPPLEMENTAL material for three papers was cited but not provided; unresolved experiments remain flagged.
NEVER compare human brain EC CPM numerically to peripheral nTPM as a selectivity ratio.

INSTALL
conda activate ml-gpu
cd ~/projects/omipep-omics-human-v0.2
S=/mnt/c/Users/bnara/Downloads
python -m zipfile -e "$S/OmiPep_M01_12_4k_ECSCR_INSTALL_bundle_v1.zip" .
python notebooks/12_4k_verify_ECSCR_review.py data/processed/review_outputs

CUMULATIVE RULE: use highest version of cumulative ledgers and working tables; NEVER concatenate cumulative files with earlier versions. New per-receptor rows append once.
