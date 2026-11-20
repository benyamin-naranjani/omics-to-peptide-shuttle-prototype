# OmiPep v0.2 | 11.5b - Evidence reconciliation before prioritisation
# Run with %run -i after 11.5. Additive audit; no ranking or fragment approval.
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

RECONCILIATION_VERSION = '1.6'
RECONCILIATION_DATE = '2026-11-19'
# Manually checked public primary-source abstracts / accessible article text.
# One scoped statement per record, not a blanket endorsement of previous claims.
# Existing evidence rows remain unchanged and are linked via source identifier.
rows = [
 ('Ecscr','20086248',2010,'Endothelial cell-specific chemotaxis receptor promotes angioblast migration during vasculogenesis and enhances VEGF receptor sensitivity','Endothelial surface and migration context','Endothelial cell surface expression, migration and VEGFR2 responses were investigated.','Not native adult brain luminal localisation or ECSCR-targeted cargo transport.'),
 ('Ecscr','23393131',2013,'Endothelial cell-specific chemotaxis receptor enhances VEGF receptor-2/KDR activation and promotes proteolysis of internalized KDR','Associated receptor trafficking','ECSCR regulates KDR activation and degradation after KDR internalisation.','KDR trafficking is not direct ECSCR-targeting ligand uptake.'),
 ('Tfrc','24470444',2014,'Transferrin receptor trafficking determines brain uptake of TfR antibody affinity variants','Antibody brain uptake and sorting','High-affinity bispecific antibodies promoted receptor lysosomal trafficking/degradation in the studied models.','Antibody-specific affinity, epitope and valency context; not generated peptide validation.'),
 ('Adam15','20189953',2010,'ADAM15 regulates endothelial permeability and neutrophil migration via Src/ERK1/2 signalling','Endothelial barrier function','ADAM15 depletion reduced HUVEC permeability and thrombin-associated dysfunction.','Peripheral EC model; no cargo transport demonstrated.'),
 ('Bst2','19837671',2009,'HIV-1 accessory protein Vpu internalizes cell-surface BST-2/tetherin through transmembrane interactions leading to lysosomes','Viral-protein-induced surface turnover','Vpu triggered BST2 internalisation and lysosomal degradation.','Transmembrane viral interaction, not evidence of targeting by an extracellular peptide.'),
 ('Orai1','24376610',2014,'Antibody-mediated targeting of the Orai1 calcium channel inhibits T cell function','Extracellular epitope and channel inhibition','An extracellular-loop-binding anti-human ORAI1 antibody inhibited T-cell function.','Not a BBB transport assay; functional perturbation liability.'),
 ('Itgb1','28787238',2018,'Beta1-integrin-matrix interactions modulate cerebral microvessel endothelial cell tight junction expression and permeability','BBB barrier function','Interference with integrin-matrix adhesion increased endothelial monolayer permeability.','Leakage is not productive transcytosis; 2017 online, 2018 issue.'),
 ('Praf2','26033241',2016,'GABAB receptor cell-surface export is controlled by an endoplasmic reticulum gatekeeper','ER retention','PRAF2 regulates GABAB receptor surface export.','Regulating another protein trafficking is not PRAF2 luminal access; 2015 online, 2016 issue.'),
 ('Procr','18841163',2009,'Endothelial protein C receptor-assisted transport of activated protein C across the mouse blood-brain barrier','Native APC blood-to-brain transport','EPCR binding-site blockade and severe EPCR deficiency reduced APC brain entry in mouse perfusion experiments.','Native ligand and mouse model; not human transport or generated peptide evidence.'),
 ('Procr','19587380',2009,'Endothelial cell protein C receptor cellular localization and trafficking: potential functional implications','Native ligand uptake and polarised trafficking','EPCR ligands APC and FVIIa promoted receptor internalisation; FVIIa apical-to-basal transport was reported.','Model and ligand dependent; coagulation pathway liability.'),
 ('Tnfrsf1a','11922661',2002,'TNFalpha transport across the blood-brain barrier is abolished in receptor knockout mice','Native TNF transport','Both TNF receptors contributed to transport; double knockout abolished it.','Not TNFR1-exclusive or peptide transport; inflammatory signalling risk.'),
 ('Tm4sf1','26241677',2015,'Intracellular distribution of TM4SF1 and internalization of TM4SF1-antibody complex in vascular endothelial cells','Peripheral endothelial uptake','Anti-TM4SF1 antibody internalisation and intracellular redistribution were observed in cultured vascular ECs.','No demonstrated BBB transcytosis; membrane loop and trafficking context.'),
 ('Tm4sf1','38946725',2024,'TM4SF1 is a molecular facilitator that distributes cargo proteins intracellularly in endothelial cells in support of blood vessel formation','Intracellular trafficking','TM4SF1-associated protein complexes and intracellular distribution were investigated.','No brain parenchymal cargo release established.'),
 ('Ace','8967503',1996,'Endothelial cells internalize monoclonal antibody to angiotensin-converting enzyme','Peripheral endothelial uptake','Anti-ACE antibody 9B9 underwent endothelial internalisation.','Not BBB transport.'),
 ('Ace','15777608',2005,'Selective rat lung endothelial targeting with a new set of monoclonal antibodies to angiotensin I-converting enzyme','Peripheral binding sink','Systemically administered ACE antibodies targeted lung endothelium in rats.','Antibody- and species-dependent; does not quantify peptide biodistribution.'),
 ('Itm2a','35313913',2022,'Exploring ITM2A as a new potential target for brain delivery','Uptake and transport test','Antibody uptake in overexpression models was studied; productive BBB transport was not established.','Native surface protein expression and ligand-specific trafficking remain unresolved.'),
 ('Lrp1','18489712',2008,'Involvement of the low-density lipoprotein receptor-related protein in the transcytosis of the brain delivery vector angiopep-2','Peptide delivery reference','Competition and transport experiments supported LRP1 involvement in Angiopep-2 transport.','Not universal luminal access or proof of every LRP1 ligand transport.'),
 ('Lrp1','29090274',2017,'A novel LRP1-binding peptide L57 that crosses the blood brain barrier','Peptide delivery reference','L57 binding to LRP1 cluster IV and brain delivery were reported.','Ligand-specific results; requires independent benchmark reproduction.'),
 ('Selp','9285631',1997,'Exacerbation of cerebral injury in mice that express the P-selectin gene','Stroke vascular targeting','Radiolabelled antibody accumulation and microvascular endothelial staining supported early postischaemic P-selectin expression in mice.','Vascular accessibility does not establish parenchymal transport.'),
 ('Selp','24564342',2014,'Imaging early endothelial inflammation following stroke by core shell silica superparamagnetic glyconanoparticles that target selectin','Stroke vascular targeting','E/P-selectin-targeted particles accumulated in brain vasculature after stroke in an animal model.','Combined selectin targeting; vascular accumulation, not receptor-specific transcytosis.'),
 ('Selp','34482087',2021,'Fucoidan-functionalized polysaccharide submicroparticles loaded with alteplase for efficient targeted thrombolytic therapy','Thrombus targeting / peripheral binding','Fucoidan-based particles targeted activated platelet and endothelial P-selectin context in experimental thromboembolic stroke.','Thrombolysis and vascular targeting are not BBB crossing; platelet binding matters.'),
 ('Cd74','22014447',2011,'Role for CD74 and CXCR4 in clathrin-dependent endocytosis of the cytokine MIF','Native ligand internalisation','MIF endocytosis depended on CD74 and CXCR4 and was associated with endosomal signalling.','Multi-receptor, non-BBB context; no standalone shuttle evidence.'),
 ('Mcam','28827364',2017,'CD146 coordinates brain endothelial cell-pericyte communication for blood-brain barrier development','BBB developmental function','CD146 participates in brain EC-pericyte communication and BBB development.','Developmental localisation and barrier function do not establish adult luminal transport.'),
 ('Mcam','31903117',2020,'Soluble CD146, a cerebrospinal fluid marker for neuroinflammation, promotes blood-brain barrier dysfunction','Soluble target and barrier risk','Soluble CD146 was investigated in neuroinflammation and BBB dysfunction.','Not proof of peptide binding or quantitative systemic sink.'),
]
cols=['gene','pmid','issue_publication_year','title_search_label','reviewed_endpoint','source_supported_statement','interpretation_limit']
fi_reconciled_sources=pd.DataFrame(rows,columns=cols)
fi_reconciled_sources['source_id']='PMID:'+fi_reconciled_sources.pmid
fi_reconciled_sources['source_url']='https://pubmed.ncbi.nlm.nih.gov/'+fi_reconciled_sources.pmid+'/'
fi_reconciled_sources['review_date']=RECONCILIATION_DATE
fi_reconciled_sources['review_scope']='targeted abstract or accessible article text; scoped statement only'
assert not fi_reconciled_sources.duplicated(['gene','source_id','reviewed_endpoint']).any()
assert set(fi_reconciled_sources.gene)<=set(structural_scope_genes)

# Preserve the retrieval failures; add manual source resolution separately.
fi_endpoint_audit_v16=receptor_literature_endpoint_review.copy()
manual=fi_reconciled_sources[['gene','source_id','issue_publication_year','source_url']].drop_duplicates()
assert not manual.duplicated(['gene','source_id']).any()
manual=manual.rename(columns={'source_url':'manual_source_url'})
fi_endpoint_audit_v16=fi_endpoint_audit_v16.merge(manual,on=['gene','source_id'],how='left',validate='many_to_one')
fi_endpoint_audit_v16['manual_source_review_available']=fi_endpoint_audit_v16.manual_source_url.notna()
fi_endpoint_audit_v16['combined_identity_status']=np.where(fi_endpoint_audit_v16.manual_source_review_available,'direct_source_located',fi_endpoint_audit_v16.bibliographic_status)
# Claim-level audit is conservative: the additional scoped statement does not certify
# every original row using that same reference.
fi_endpoint_audit_v16['original_claim_review_status']='original_claim_reconciliation_pending'
fi_endpoint_audit_v16['year_used_for_reviewed_source']=fi_endpoint_audit_v16.issue_publication_year.combine_first(pd.to_numeric(fi_endpoint_audit_v16.resolved_year,errors='coerce'))

# HPA audit: original code used exact 'brain'. Inspect explicit CNS label variants.
# Same original support requirements and exact stable-ID policy are maintained.
brain_labels={'brain','cerebral cortex'}
brain_cluster_audit=cluster_metadata.loc[cluster_metadata.tissue_key.isin(brain_labels)].copy()
brain_cluster_audit['exact_blood_EC_label']=brain_cluster_audit.cell_type_key.eq('endothelial cells')
brain_cluster_audit['passes_existing_support_policy']=(brain_cluster_audit.cluster_n_cells.ge(HPA_MIN_CLUSTER_CELLS)&brain_cluster_audit.cluster_annotation_reliability.isin(HPA_ACCEPTED_ANNOTATIONS)&brain_cluster_audit.cluster_in_aggregation)
# No rescue based on partial matches such as mixed vascular/pericyte populations.
brain_cluster_context_v16=hpa_supported_clusters.loc[hpa_supported_clusters.tissue_key.isin(brain_labels)&hpa_supported_clusters.blood_vascular_EC].copy()
brain_summary_v16=hpa_rna_summary(brain_cluster_context_v16,'HPA_audited_brain_EC_cluster','context',hpa_target_ids)
assert brain_summary_v16.human_ensembl_gene_id.is_unique
assert brain_summary_v16.loc[brain_summary_v16.HPA_audited_brain_EC_cluster_n_observed.eq(0),'HPA_audited_brain_EC_cluster_max_nTPM'].isna().all()

# Full integration: immutable v1.5 base and additive evidence fields.
b= receptor_feasibility_evidence_v15.copy()
key=['gene','human_ensembl_gene_id']
assert not b.duplicated(key).any()
notes=fi_reconciled_sources.groupby('gene',as_index=False).agg(n_directly_reviewed_PMIDs=('pmid','nunique'),directly_reviewed_sources_json=('source_id',lambda x:json.dumps(sorted(set(x)))))
receptor_feasibility_evidence_v16=b.merge(notes,on='gene',how='left',validate='many_to_one').merge(brain_summary_v16,on='human_ensembl_gene_id',how='left',validate='many_to_one')
v=receptor_feasibility_evidence_v16
v['n_directly_reviewed_PMIDs']=v.n_directly_reviewed_PMIDs.fillna(0).astype(int)
v['HPA_audited_brain_and_peripheral_context_available']=v.HPA_audited_brain_EC_cluster_n_observed.gt(0)&v.HPA_peripheral_EC_cluster_n_observed.gt(0)
# Explicit scoped observations for newly reviewed candidate genes; no promotion to
# established shuttles, no assumption of functional equivalence with human genes.
context={
 'Selp':('stroke_vascular_targeting_context','Activated platelet and endothelial targeting; human stroke protein and intact cargo transport unresolved'),
 'Cd74':('native_ligand_internalisation_context','MIF uptake involves CD74/CXCR4; immune expression, signalling and productive BBB sorting require review'),
 'Mcam':('barrier_development_and_soluble_target_context','Pericyte/EC context and soluble receptor; adult blood-facing targeting unresolved'),
 'Ace':('endothelial_uptake_with_lung_sink_context','Peripheral lung targeting reported for antibodies; BBB transport unestablished'),
 'Procr':('mouse_native_ligand_transport_context','Native APC transport; preserve EPCR/protein C and coagulation function'),
 'Tnfrsf1a':('mouse_multi_receptor_native_transport_context','TNFR1 and TNFR2 contribute to TNF transport; inflammatory signalling risk'),
 'Tm4sf1':('peripheral_endothelial_uptake_context','Internalisation not abluminal release; four-pass receptor context'),
 'Ecscr':('endothelial_surface_associated_trafficking_context','Direct ECSCR-targeted uptake unresolved'),
 'Itm2a':('exploratory_negative_or_inconclusive_transport_context','Overexpression uptake not successful native BBB transport'),
 'Lrp1':('peptide_delivery_benchmark_context','Angiopep-2 and L57; ligand-specific domain and polarity review required'),
 'Tfrc':('antibody_delivery_benchmark_context','Affinity-dependent trafficking and iron-homeostasis risk'),
}
v['direct_review_context']=v.gene.map({g:x[0] for g,x in context.items()}).fillna('no_additional_scoped_context')
v['direct_review_caveat']=v.gene.map({g:x[1] for g,x in context.items()})
v['additional_targeted_primary_source_review_available']=v.n_directly_reviewed_PMIDs.gt(0)
# Do not equate targeted source review with closed endpoint audit.
v['endpoint_reconciliation_complete']=False
v['final_selection_performed']=False
assert len(v)==len(b)
pd.testing.assert_frame_equal(b.reset_index(drop=True),v[b.columns].reset_index(drop=True),check_dtype=False)
assert not v.duplicated(key).any()
scope=v.loc[v.structural_assessment_scope].copy()
assert set(scope.gene)==set(structural_scope_genes)
assert not scope.fragment_approved.any()

# Keep all relationship rows and show quantitative review information, not ranks.
review_fields=['gene','human_gene_symbol','human_ensembl_gene_id','review_lane','stroke_upregulation_supported','D02_vs_Sham_logFC','D02_vs_Sham_FDR','D14_vs_Sham_logFC','D14_vs_Sham_FDR','D02_vs_Sham_candidate_robust_minimum_effect','D14_vs_Sham_candidate_robust_minimum_effect','accessibility_review_context','external_identity_pct','unsorted_median_CPM','EC_vs_non_EC_median_paired_log2_ratio','HPA_peripheral_EC_cluster_max_nTPM','HPA_audited_brain_EC_cluster_n_observed','human_template_context','delivery_review_context','direct_review_context','direct_review_caveat','n_directly_reviewed_PMIDs','additional_primary_review_required','fragment_approved']
fi_review_packet_v16=scope[review_fields].sort_values(['review_lane','gene','human_ensembl_gene_id'])
# Pending uses original status plus additional evidence; this is NOT a systematic review completion flag.
fi_pending_v16=scope.loc[scope.additional_primary_review_required & ~scope.additional_targeted_primary_source_review_available].copy()
# Preserve any source/claim requiring manual reconciliation even for reviewed genes.
fi_audit_summary_v16=pd.DataFrame([
 ('structural_scope_mouse_genes',scope.gene.nunique()),('structural_scope_relationships',len(scope)),
 ('direct_review_scoped_records',len(fi_reconciled_sources)),('direct_review_mouse_genes',fi_reconciled_sources.gene.nunique()),
 ('previous_failed_identifiers_directly_located',fi_endpoint_audit_v16.loc[fi_endpoint_audit_v16.bibliographic_status.eq('retrieval_failed')&fi_endpoint_audit_v16.manual_source_review_available,'source_id'].nunique()),
 ('previous_failed_identifiers_still_unresolved',fi_endpoint_audit_v16.loc[fi_endpoint_audit_v16.bibliographic_status.eq('retrieval_failed')&~fi_endpoint_audit_v16.manual_source_review_available,'source_id'].nunique()),
 ('HPA_brain_label_metadata_rows',len(brain_cluster_audit)),('HPA_supported_exact_brain_EC_clusters',brain_cluster_context_v16[['tissue_key','cluster_key']].drop_duplicates().shape[0]),
 ('HPA_audited_both_context_relationships',int(scope.HPA_audited_brain_and_peripheral_context_available.sum())),('pending_targeted_primary_review_genes',fi_pending_v16.gene.nunique()),('approved_fragments',int(scope.fragment_approved.sum()))],columns=['metric','value'])
outputs={'EC_direct_source_reconciliation_v1_6':fi_reconciled_sources,'EC_original_literature_endpoint_audit_v1_6':fi_endpoint_audit_v16,'HPA_brain_cluster_label_audit_v1_6':brain_cluster_audit,'HPA_brain_EC_context_audited_v1_6':brain_summary_v16,'EC_integrated_receptor_feasibility_evidence_v1_6':v,'EC_integrated_feasibility_review_v1_6':scope,'EC_feasibility_review_packet_v1_6':fi_review_packet_v16,'EC_remaining_targeted_primary_review_v1_6':fi_pending_v16,'EC_evidence_reconciliation_summary_v1_6':fi_audit_summary_v16}
for name,table in outputs.items(): table.to_csv(Path(out_dir)/f'{name}.csv',index=False)
manifest={'version':RECONCILIATION_VERSION,'generated_utc':datetime.now(timezone.utc).isoformat(),'direct_source_review_date':RECONCILIATION_DATE,'source_review':'Targeted public abstract/accessible article review. Original records retained; scoped statements do not certify all earlier claims.','HPA_policy':'Exact brain/cerebral cortex and blood endothelial label plus original support and identity filters; no mixed-cell rescue or cross-resource expression ratios.','selection':'No ranking, final selection or coordinate approval.','limitations':['Not systematic review; additional sources and original endpoint reconciliation remain pending.','Normal atlas does not establish human stroke expression or blood-facing protein availability.','HPA brain/peripheral expression context, if available, does not establish specificity or pharmacokinetics.']}
(Path(out_dir)/'EC_evidence_reconciliation_manifest_v1_6.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
receptor_feasibility_evidence=v
feasibility=v
receptor_integrated_feasibility_review=scope
receptor_biological_review_worklist=scope
print('Directly reviewed earlier failed identifiers:')
display(fi_endpoint_audit_v16.loc[fi_endpoint_audit_v16.bibliographic_status.eq('retrieval_failed'),['gene','source_id','combined_identity_status','manual_source_url']].drop_duplicates())
print('\nHPA brain/cerebral cortex annotation inventory:')
display(brain_cluster_audit)
print('\nSelected extended candidate context:')
display(fi_review_packet_v16.loc[fi_review_packet_v16.gene.isin(['Selp','Cd74','Mcam','Procr','Tm4sf1','Ecscr','Ace','Itm2a','Lrp1','Tfrc'])].round(4))
print('\nReconciliation summary:'); display(fi_audit_summary_v16)
print('\nEvidence reconciliation exported. No final ranking or fragment approval performed.')
