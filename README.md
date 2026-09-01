# Omics-to-Peptide-Shuttle Prototype

A version-controlled computational prototype for omics-guided design and prioritisation of brain-targeting peptides.

The workflow connects stroke endothelial transcriptomics, target prioritisation, generative peptide modelling, receptor-aware structural triage and reward-guided optimisation.

## Computational workflow

**stroke endothelial omics → target prioritisation → peptide generation → receptor-aware structural triage → reward-guided optimisation**

## Preliminary computational feasibility

![Preliminary computational feasibility](figures/computational_feasibility.png)

**Current prototype results.** Analysis of 7,627 brain endothelial cells identified eight primary stroke-responsive target hypotheses. The peptide generator produced 100 valid and unique peptides, of which 99 were novel relative to the training data. A subsequent 1,000-peptide run yielded 997 unique sequences, with 10 diverse candidates advanced to receptor-aware structural triage. A proof-of-principle reward-guided update demonstrated iterative adaptation of the generator.

These results establish computational feasibility of the workflow but do not demonstrate target binding, BBB transport or therapeutic activity.

## Workflow implementation

1. `01_gse225948_endothelial_receptor_mining.ipynb`  
   Mines stroke-responsive endothelial candidates from public single-cell RNA-seq data and applies surface-accessibility prioritisation.

2. `02_b3pdb_data_downloader.ipynb`  
   Retrieves and cleans B3PDB and CPPsite2 peptide datasets.

3. `03_peptide_transformer_generation_and_scoring.ipynb`  
   Trains a peptide transformer using CPPsite2 pretraining followed by B3PDB fine-tuning and generates candidate sequences.

4. `04_receptor_aware_peptide_triage.ipynb`  
   Performs physicochemical analysis and Ly6a-aware ColabFold structural triage.

5. `05_reward_guided_peptide_rl_update.ipynb`  
   Demonstrates reward-guided updating of peptide sequence probabilities using receptor-aware computational signals.

6. `06_multi_receptor_structure_triage.ipynb`  
   Extends the workflow to multi-receptor structure-aware triage and reward aggregation.

## Scope

This repository is a proof-of-concept computational prototype. Generated peptides are computational candidates only. ColabFold-derived metrics are used as structure-aware triage signals rather than binding affinities. No experimental binding, BBB transport, toxicity or in vivo efficacy is claimed.

## Outputs

Key processed outputs are generated under:

- `data/processed/`
- `models/`

Large ColabFold result directories are excluded from version control where appropriate.

## Reproducibility

Run notebooks sequentially from `01` to `06`.
