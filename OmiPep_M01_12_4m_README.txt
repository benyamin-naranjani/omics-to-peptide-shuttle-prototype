OmiPep v0.2 — Module 01 Cell 12.4m standalone receptor-selection table

Install after having installed the previous LRP1 (12.4k) and 12.4l selection audit bundles.
The required inputs already exist from those bundles:
  data/processed/review_outputs/EC_receptor_M01_12_4l_six_receptor_selection_v1.csv
  data/processed/review_outputs/EC_receptor_M01_12_4l_125_gate_audit_v1.csv
  data/processed/review_outputs/EC_receptor_M01_12_4k_fulltext_paper_ledger_v6.csv
  data/processed/review_inputs/EC_receptor_M01_expert_master_WITH_ENDPOINT_EVIDENCE_v2.csv

Installation from the WSL project root:
  conda activate ml-gpu
  cd ~/projects/omipep-omics-human-v0.2
  python -m zipfile -e "/mnt/c/Users/bnara/Downloads/OmiPep_M01_12_4m_PUBLICATION_TABLE_INSTALL_bundle_v1.zip" .

Rebuild from source, validate and export:
  python notebooks/12_4m_publication_receptor_selection_table.py

Notebook Cell 12.4m (when notebook kernel cwd is notebooks):
  %run -i 12_4m_publication_receptor_selection_table.py

If kernel cwd is repo root instead:
  %run -i notebooks/12_4m_publication_receptor_selection_table.py

Outputs generated/rebuilt:
  data/processed/review_outputs/EC_receptor_M01_12_4m_six_receptor_publication_table_v1.csv
  data/processed/review_outputs/EC_receptor_M01_12_4m_reviewed_DOI_map_v1.csv
  data/processed/review_outputs/EC_receptor_M01_12_4m_table_caption_v1.txt
  data/processed/review_outputs/EC_receptor_M01_12_4m_validation_v1.json
  results/figures/module_01/M01_receptor_selection_table_12_4m_v1.{svg,pdf,png}

NOTE: This is a table as an independent interim publication figure, NOT yet assigned
as a final Nature Communications figure/table. Table DOI text is intentionally retained;
publication-ready reference-number formatting will happen at final manuscript assembly.
The two selected receptors are provisional computational targets; expert and experimental
approvals remain open. This script never writes to the source evidence master.
