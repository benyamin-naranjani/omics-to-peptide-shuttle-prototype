# Omics-to-Peptide-Shuttle Prototype

Compact end-to-end prototype for an omics-guided peptide-shuttle design workflow.

## Aim

This repository demonstrates a computational decision chain for stroke-context peptide-shuttle design:

stroke endothelial omics → receptor prioritisation → peptide generator training → receptor-aware structure triage → reward-guided peptide optimisation

## Workflow

1. `01_gse225948_endothelial_receptor_mining.ipynb`  
   Mines stroke-responsive endothelial receptor candidates from public single-cell RNA-seq data.

2. `02_b3pdb_data_downloader.ipynb`  
   Retrieves and cleans B3PDB and CPPsite2 peptide datasets.

3. `03_peptide_transformer_generation_and_scoring.ipynb`  
   Trains a peptide transformer: CPPsite2 pretraining followed by B3PDB fine-tuning.

4. `04_receptor_aware_peptide_triage.ipynb`  
   Generates peptides and performs Ly6a-aware ColabFold triage.

5. `05_reward_guided_peptide_rl_update.ipynb`  
   Demonstrates single-receptor reward-guided generator update.

6. `06_multi_receptor_structure_triage.ipynb`  
   Extends the workflow to multi-receptor structure-aware reward-guided optimisation.

## Key features

- Single-cell endothelial receptor mining
- UniProt-based surface-accessibility annotation
- Peptide transformer generation
- Cheap peptide developability and safety heuristics
- ColabFold-Multimer receptor-peptide triage
- Structure-aware reward scoring
- Reward-guided generator update
- Multi-receptor reward aggregation

## Scope

This is a proof-of-concept prototype. Generated peptides are computational candidates only. ColabFold-derived scores are used as structure-aware triage signals, not binding affinities. No experimental binding, BBB transport, toxicity, or in vivo validation is claimed.

## Main outputs

Key outputs are generated under:

- `data/processed/`
- `models/`

Large ColabFold result folders are excluded from version control where appropriate.

## Reproducibility

Run notebooks in order from `01` to `06`.
