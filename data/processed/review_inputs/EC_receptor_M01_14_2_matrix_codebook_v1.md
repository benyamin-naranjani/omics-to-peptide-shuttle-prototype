# Cell 14.2 — Evidence and translational-liability matrix (v1)

This is an **evidence-linked, post-hoc and manually adjudicated** depiction of the five provisional discovery receptors (PROCR, TM4SF1, SELP, ECSCR, ACKR1) and the independent comparator LRP1. No expert sign-off, human stroke luminal target validation, structural fragment approval, peptide toxicology or intact engineered BBB cargo delivery is asserted.

## Evidence columns

- `access`: Blood-facing protein presentation **in the stated species/model**. Human ischemic BBB luminal availability is **not assumed**.
- `uptake`: Cellular uptake/trafficking evidence **for the reported ligand/construct**, including indirect associations where flagged.
- `brainward`: Blood-to-brain transport in the **stated experimental model and cargo modality**; brain tracer or a native-ligand result is not interchangeable with verified free-peptide delivery.
- `peptide`: Chemically verified **intact engineered peptide/cargo on the brainward side**. No entry meets this standard in the reviewed evidence. LRP1's receiver-side intactness claim is unverified in the supplied full-text evidence.

`observed` = demonstrated in a specified source model and cargo class, not a human-stroke/peptide claim; `contextual` = indirect or model-specific evidence insufficient for the named mechanism; `reverse` = published ligand flux in the opposite direction (not a universal exclusion); `unestablished` = endpoint not established by the reviewed sources (not a demonstrated biological negative).

## Translational liabilities (distinct from toxicity)

- `function`: Receptor signaling, vascular function, homeostasis, inflammation, ligand competition or construct-specific interference.
- `barrier`: Evidence about BBB permeability/integrity, distinct from transcytosis; a binder-specific experiment is generally absent.
- `epitope`: Glycans, species-dependent binding, receptor-domain conservation, endogenous ligand-site constraints, and transport format dependency.

`observed` = pathway/structural concern experimentally documented under particular intervention/conditions; **NOT toxicity of any proposed short peptide**. `potential` = plausible, unresolved mechanistic concern. `unknown` = relevant binder-specific liability not assessed. Never interpret grey as low risk or safety.

## Peripheral expression track

`HPA_peripheral_EC_max_nTPM` is the reported highest peripheral endothelial cell-cluster RNA expression within Human Protein Atlas context. The bar length is `log10(1 + nTPM)`, and the number shown is the untransformed nTPM value. It is not a drug distribution measurement, ligand sink quantification, toxicity endpoint, or brain-specificity ratio. Do not divide HPA nTPM by the independent human-brain atlas CPM values.

## Traceability

The accompanying adjudication CSV contains exactly 42 rows (6 receptors × 7 columns), with a unique receptor/dimension key, visible label, detailed interpretation boundary, and reviewed source citation keys. The script checks the citation keys against the immutable 19-paper full-text ledger, joins corresponding DOIs, checks 12.4l/12.4m consistency, and writes a detailed source-data CSV and SHA-256 manifest.

The classification is a reviewer-facing **visual summary of the evidence base**, not an automatic scoring algorithm or numerical selection gate. A full scientific manuscript must describe differences in experimental model, species, ligand/format, and endpoint, preferably with direct references to the original study figures.
