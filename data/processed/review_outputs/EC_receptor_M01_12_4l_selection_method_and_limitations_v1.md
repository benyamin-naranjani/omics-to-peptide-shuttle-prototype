# Module 01: auditable receptor selection — provisional, post-hoc (v1)

## Exact workflow

1. **Universe:** Preserve all 125 mouse→human relationships (114 mouse genes), including many-to-many mappings; never replace missing observations by zero.
2. **Stroke evidence gate:** Flag D02 log2FC > 0 and FDR < 0.05. Among the uploaded records this gives 94 relationships. Record LOSO direction stability without manufacturing a binary robustness distinction when most are 100%.
3. **Human and topology context:** Record human brain EC abundance/enrichment, extracellular conservation, membrane localization and peripheral expression. Do NOT equate RNA abundance with native luminal receptor density. A zero/low healthy brain reference does not rule out stroke-induced luminal expression. Explicitly mark organelle-resident and secreted-only source classifications as provisional screening deferrals, not as full-text validated negatives.
4. **Numerical triage, NOT target selection:** Within 94 D02-up relationships, calculate empirical percentile ranks for four independent quantities: D02 logFC, human brain EC median CPM, human EC/non-EC enrichment, and mouse-human extracellular identity. Balanced weights = 0.25/0.30/0.25/0.20. Three alternative weightings stress-test the dependence on chosen weights. Missing fields are omitted from the partial score, and coverage is reported; never rank a missing-data row as though complete. All weights are analytic conventions selected AFTER seeing the cohort, not preregistered or validated therapeutic cutoffs.
5. **Manual endpoint-level adjudication:** Use 19 source-reviewed papers/213 experiment records to separately evaluate the *five already shortlisted* receptor mechanisms (native ligand vs nanocarrier vs antibody uptake vs indirect trafficking; transport direction; intact peptide or only isotope signal); **do not extrapolate those endpoint judgments to the other 120 relationships**. Count screening-only alternative hits independently to identify possible omissions. No quantity of citations substitutes for an endpoint.
6. **Noncompensatory constraints:** A documented opposite-direction transport result, physiologic ligand competition, major vascular safety issue, human luminal uncertainty or erythrocyte/platelet binding cannot be cancelled by high expression or weighted score. Safety must be tested with the actual *nonblocking* binder and cargo, not inferred from knockouts/ADCs.
7. **Decision:** PROCR + TM4SF1 are provisional computational design arms, distinct in mechanism and quantitative expression; SELP is the highest-priority conditional reserve; ECSCR is a reserve; ACKR1 has an opposite-direction warning. LRP1 is a separate benchmark arm (Angiopep-2, L57; Angiopep-7 for published comparison), **not one of the stroke-upregulated discovery receptors**. No final receptor/epitope or clinical approval.

## Why these two?

- **PROCR:** native mouse blood-to-brain APC precedent, moderate human EC expression/enrichment; must demonstrate short-peptide transport and avoid APC/protein C anticoagulant interference and soluble-EPCR sink.
- **TM4SF1:** high human brain EC expression/enrichment and surface antibody internalization, but **no proven BBB peptide transcytosis**; glycan-dependent species difference, peripheral skin EC exposure and angiogenesis/repair liability are major translational hurdles.
- **SELP:** nanocarrier transport in a brain tumour model and strong mouse stroke induction but 0.303 healthy-human brain EC CPM; patient-stroke protein abundance and platelet binding/safety are gating experiments.
- **ECSCR:** human EC enrichment but no demonstrated receptor-dependent shuttle; 46% ECD identity and vascular signalling liabilities.
- **ACKR1:** intact native chemokine transport is brain/abluminal→blood/luminal, the reverse of the desired brain-influx route; erythrocyte sink depends on genotype.
- **LRP1:** positive peptide benchmark evidence is not a validated clinical endpoint; L57 in-situ 0.220±0.052% input vs Angiopep-7 0.053±0.019%, with IV differences not established; Angiopep-2 vs Angiopep-7 are different sequences.

## Reviewer disclosures

- Initial five-receptor shortlist was exploratory/literature-informed, not derived uniquely by the retrospective quantitative rule. Describe it honestly as *two-stage evidence triage with decision audit*.
- This implementation is a documented reproducibility artifact, not a machine-learned classifier or clinically calibrated scoring model. LLM-assisted note extraction does not mean independent validation; claims must retain paper and experiment IDs.
- The 20 highest quantitative **alternative** membrane-annotated relationships, and a deduplicated human-target list, are retained as counterfactual stress tests, not described as negative without full-text adjudication. Some top-scoring IFITM mappings are mostly intracellular despite their screening topology label, highlighting the need to cross-check source narratives.
- Numerical weights change ranks markedly, including PROCR, underscoring why the numerical diagnostic alone cannot justify the final target decision.
- No numerical score is allowed to override unresolved native luminal epitope, actual blood→brain intact peptide transfer, or essential physiologic safety.
- The selected targets are **provisional computational design arms**; expert confirmation and experiment design remain mandatory.

## Planned minimum experiments for claims beyond computational selection

Native human ischemic brain microvascular immunoluminal access; receptor-specific binder affinity/competition; polarized human BBB integrity plus directional mass-balanced intact peptide basolateral recovery (LC-MS, not just fluorescence or radiotracer); matched knockout/competition and paracellular controls; platelet/coagulation/angiogenesis tests; human cross-reactivity, membrane, glycan and off-organ tissue panel.
