OmiPep Module 01 / Cell 12.4k ACKR1 (DARC) Pass 2 | 2026-11-22

Contents:
- 3 primary full-text PDFs, preserved byte-identical in data/processed/review_sources/ACKR1/
- 3 paper-level ACKR1 records and 46 figure/method-level ACKR1 experimental records
- Cumulative 16-paper / 169-experiment ledgers: previous PROCR/TM4SF1/ECSCR/SELP entries UNMODIFIED
- Six-receptor shortlist v7 and provisional manuscript comparison v5: ONLY ACKR1 record updated
- Follow-up table and SHA-256 validation report
- Idempotent validator script notebooks/12_4k_verify_ACKR1_review.py

Key biological boundary:
ACKR1-mediated chemokine flux is predominantly abluminal->luminal (tissue->blood), including in primary murine BBB. The humanized ligand-exposed 20-aa peptide blood->brain route is not established. Strong evidence for physiological chemokine export and mature RBC Duffy-protein sequestration concerns. Duffy-null promoter genotype affects erythrocytes selectively, not necessarily vascular endothelium.

Full text limitations:
Pruenster: non-physiological overexpressed polarized MDCK/HUVEC; supplementary figures not supplied.
Minten: EAE/MS rather than stroke; no direct structural peptide integrity analysis in receiver fraction.
Saha: structural human receptor in vitro; no direction-resolved transport, unmodeled proximal N-terminus, Nb52 density absent; data supplements not supplied.

Install in WSL after downloading zip:
conda activate ml-gpu
cd ~/projects/omipep-omics-human-v0.2
S=/mnt/c/Users/bnara/Downloads
python -m zipfile -e "$S/OmiPep_M01_12_4k_ACKR1_INSTALL_bundle_v1.zip" .
python notebooks/12_4k_verify_ACKR1_review.py data/processed/review_outputs

Never overwrite or discard the original 125-row master, 15-col decision table, earlier review bundles, or initial RNA/orthology outputs. No reviewer or wet-lab expert has approved a final receptor.
