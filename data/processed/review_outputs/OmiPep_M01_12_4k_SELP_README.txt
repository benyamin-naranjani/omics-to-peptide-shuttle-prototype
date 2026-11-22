OmiPep v0.2 | Module 01 Cell 12.4k | SELP/P-selectin full-text Pass 2 | 2026-11-22

1) STATUS: Retained as discovery candidate with NOT-APPROVED evidence; do not select final target or approve docking epitope.
2) Source PDFs: Tylawsky 2023 (Nature Materials); Jin 2010 (BMC Neuroscience); Green 1994 (Journal of Cell Biology).
3) Three matching keys are ALREADY in Master Reference Library. No duplicate library entries are necessary.
4) Disease-model transfer: ~80nm multivalent fucoidan nanoparticle crossing via Cav1-dependent pathway in SHH medulloblastoma, including control Selp-/-, Cav1-/- and control dextran-sulfate NP; demonstrated only for this modality/disease setting. No short peptide shuttle.
5) Stroke safety: Jin stroke Selp KO lower IgG/Gd leak and PMN infiltration. Gd-DTPA-sLeX binds both SELP and SELE, and targeted MRI signal in Selp-/- does not prove SELP. Direct stroke leakage is NOT transcytosis.
6) Trafficking: Green human P-selectin in CHO/PC12 rapidly lysosomally sorted after surface internalization (2.3h half-life plated CHO; deleting 10 aa increases to 9.5h). This does NOT disprove alternate carrier/Cav1 routing.
7) Human translation bottleneck: M01 healthy human brain endothelial RNA median 0.303 CPM; D02 mouse log2FC +7.10555, FDR 0.00355, LOSO 100%; requires human ischemic luminal protein validation.
8) Peripheral sink: M01 bladder bulk RNA max 13.9 nTPM; lung EC 158.2 nTPM; immune 8.8 nTPM; activated platelets and other vascular beds cannot be ignored. Do not compare nTPM with brain EC CPM as a ratio.
9) Three original PDFs and the associated extended data figs INCLUDED in Tylawsky PDF were inspected. The referenced Tylawsky Supplementary Figures 1-19/videos are NOT part of this uploaded PDF; their outcomes are flagged as author-reported only.
10) Files are append-only newer cumulative snapshots, not overwrites of v3/v2/v5/v3 predecessor names; previous 10 paper and 80 experiment records retained. Only SELP row changes in six-receptor working and manuscript tables.
11) WSL use:
    conda activate ml-gpu
    cd ~/projects/omipep-omics-human-v0.2
    S=/mnt/c/Users/bnara/Downloads
    python -m zipfile -e "$S/OmiPep_M01_12_4k_SELP_INSTALL_bundle_v1.zip" .
    python notebooks/12_4k_verify_SELP_review.py data/processed/review_outputs
12) Remaining: ACKR1 3 priority papers; LRP1 benchmark 3 priority papers, then noncompensatory 5-to-2 target adjudication and final visualization/checkpoints. 
