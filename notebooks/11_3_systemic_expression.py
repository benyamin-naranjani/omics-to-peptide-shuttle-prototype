# 11.3 — Systemic tissue and cellular expression context (HPA v24.1)
# Run with %run -i from the existing OmiPep notebook.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile, time, re
import numpy as np
import pandas as pd
import requests

HPA_RELEASE = '24.1'
HPA_ENSEMBL_RELEASE = 109
HPA_BASE = 'https://v24.proteinatlas.org/download/tsv/'
HPA_DIR = Path(human_data_dir) / 'HPA_v24_1'
HPA_DIR.mkdir(parents=True, exist_ok=True)
HPA_FILES = {
    'tissue': 'rna_tissue_consensus.tsv.zip',
    'tissue_metadata': 'rna_tissue_consensus_tissues.tsv.zip',
    'cluster': 'rna_single_cell_cluster.tsv.zip',
    'cluster_metadata': 'rna_single_cell_clusters.tsv.zip',
    'immune': 'rna_immune_cell.tsv.zip',
    'ihc': 'normal_ihc_data.tsv.zip',
    'ihc_metadata': 'normal_ihc_tissues.tsv.zip',
}
HPA_RNA_FLOORS = (1, 10, 100)  # descriptive only
HPA_MIN_CLUSTER_CELLS = 30
HPA_ACCEPTED_ANNOTATIONS = {'high','medium'}
HPA_IHC_HIGHER_SUPPORT = {'enhanced','supported'}

# A. Download archive files; preserve exact URLs, hashes and HTTP metadata.
def hpa_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()

def hpa_acquire(filename):
    path=HPA_DIR/filename
    manifest=path.with_suffix(path.suffix+'.json')
    url=HPA_BASE+filename
    if path.exists() and manifest.exists():
        record=json.loads(manifest.read_text())
        assert record['url']==url and record['sha256']==hpa_hash(path), f'Cache mismatch: {path}'
        assert record['bytes']==path.stat().st_size
        print('Verified cache:', filename,flush=True)
        return path,record
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(1,5):
        try:
            with requests.get(url,stream=True,timeout=(15,120),headers={'Accept-Encoding':'identity'}) as response:
                response.raise_for_status()
                assert response.status_code==200 and response.url.startswith('https://'),response.url
                n=0; next_report=32*1024**2
                with partial.open('wb') as f:
                    for block in response.iter_content(1024**2):
                        if not block: continue
                        f.write(block); n+=len(block)
                        if n>=next_report:
                            print(f'  {filename}: {n/1024**2:.1f} MiB',flush=True); next_report=n+32*1024**2
                length=response.headers.get('Content-Length')
                if length is not None and response.headers.get('Content-Encoding','identity').lower()=='identity':
                    assert n==int(length),f'Incomplete download: {filename}'
                record={'url':url,'resolved_url':response.url,'bytes':n,'last_modified':response.headers.get('Last-Modified'),'etag':response.headers.get('ETag'),'retrieved_utc':datetime.now(timezone.utc).isoformat()}
            with zipfile.ZipFile(partial) as z:
                assert z.testzip() is None, f'ZIP CRC failure: {filename}'
                record['members']=z.namelist()
            record['sha256']=hpa_hash(partial)
            partial.replace(path)
            manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
            print('Acquired:',filename, f'{record["bytes"]/1024**2:.1f} MiB',flush=True)
            return path,record
        except (requests.RequestException,zipfile.BadZipFile,AssertionError) as e:
            if attempt==4: raise
            if isinstance(e,requests.HTTPError) and e.response is not None and e.response.status_code not in {408,429,500,502,503,504}: raise
            print(f'Retry {attempt}/4: {filename}: {e}',flush=True); time.sleep(min(5*2**(attempt-1),30))

# B. Parse actual TSV headers; preserve exact source identifiers and labels.
def hpa_read(path, required, ids=None):
    with zipfile.ZipFile(path) as z:
        members=[m for m in z.namelist() if m.lower().endswith('.tsv') and not m.startswith('__MACOSX/')]
        assert len(members)==1,(path,members)
        with z.open(members[0]) as f:
            frames=[]; n_rows=0; observed_ids=set()
            for block in pd.read_csv(f,sep='\t',dtype=str,keep_default_na=False,chunksize=250000,encoding='utf-8-sig'):
                block.columns=block.columns.str.strip()
                assert set(required).issubset(block.columns),(path,list(block.columns),required)
                n_rows+=len(block)
                if ids is not None:
                    block['Gene']=block['Gene'].str.strip()
                    observed_ids.update(block['Gene'].unique())
                    block=block.loc[block['Gene'].isin(ids)]
                frames.append(block)
    assert n_rows>0, path
    frame=pd.concat(frames,ignore_index=True)
    print(f'Parsed {path.name}: {n_rows:,} source rows; {len(frame):,} retained',flush=True)
    return frame, {'source_rows':n_rows,'retained_rows':len(frame),'source_unique_gene_ids':len(observed_ids),'columns':list(frame.columns)}

def hpa_label(series):
    return series.astype(str).str.strip().str.casefold()

def hpa_numeric(frame, col):
    # Unavailable values remain missing; malformed numeric fields fail.
    raw=frame[col].replace({'':np.nan,'NA':np.nan,'N/A':np.nan})
    x=pd.to_numeric(raw,errors='raise')
    assert np.isfinite(x.dropna()).all() and x.dropna().ge(0).all(),col
    return x

def hpa_rna_summary(frame, prefix, label_col, universe):
    assert 'nTPM' in frame and label_col in frame
    records=[]
    for gene_id, g in frame.groupby('human_ensembl_gene_id',sort=False):
        valid=g.loc[g['nTPM'].notna()]
        r={'human_ensembl_gene_id':gene_id, f'{prefix}_n_observed':len(valid)}
        if len(valid):
            peak=valid['nTPM'].max()
            r.update({f'{prefix}_max_nTPM':peak,f'{prefix}_median_nTPM':valid['nTPM'].median(),f'{prefix}_peak_context':'; '.join(sorted(set(valid.loc[valid.nTPM.eq(peak),label_col].astype(str))))})
            for floor in HPA_RNA_FLOORS:
                r[f'{prefix}_n_ge_{floor}']=int(valid.nTPM.ge(floor).sum())
        records.append(r)
    cols=['human_ensembl_gene_id',f'{prefix}_n_observed',f'{prefix}_max_nTPM',f'{prefix}_median_nTPM',f'{prefix}_peak_context']+[f'{prefix}_n_ge_{x}' for x in HPA_RNA_FLOORS]
    result=pd.DataFrame(records,columns=cols).set_index('human_ensembl_gene_id').reindex(universe).reset_index()
    result[f'{prefix}_n_observed']=result[f'{prefix}_n_observed'].fillna(0).astype(int)
    return result

# Synthetic test of missing versus measured zero and tied maxima.
_t=pd.DataFrame({'human_ensembl_gene_id':['A','A','B'],'nTPM':[0.,0.,12.],'context':['x','y','z']})
_s=hpa_rna_summary(_t,'test','context',['A','B','C']).set_index('human_ensembl_gene_id')
assert _s.loc['A','test_max_nTPM']==0 and _s.loc['A','test_n_ge_1']==0
assert _s.loc['A','test_peak_context']=='x; y'
assert _s.loc['C','test_n_observed']==0 and pd.isna(_s.loc['C','test_max_nTPM'])
print('Synthetic systemic-expression controls passed.')

# C. Establish full target universe. Preserve exact-ID matches only; no symbol rescue.
_hpa_base=receptor_feasibility_evidence_v12.copy()
hpa_target_ids=sorted(_hpa_base.human_ensembl_gene_id.dropna().unique())
assert all(re.fullmatch(r'ENSG\d+',x) for x in hpa_target_ids)
assert not _hpa_base.duplicated(feasibility_pair_keys).any()
_hpa_ids=set(hpa_target_ids)
_hpa_paths={}; _hpa_manifest={}
for key, filename in HPA_FILES.items():
    _hpa_paths[key],_hpa_manifest[key]=hpa_acquire(filename)

schemas={
 'tissue':['Gene','Tissue','nTPM'],
 'tissue_metadata':['Tissue','Organ'],
 'cluster':['Gene','Tissue','Cell type','Cluster','nTPM'],
 'cluster_metadata':['Tissue','Cluster','Cell type','Cell type group','Cell count','Included in aggregation','Annotation reliability'],
 'immune':['Gene','Immune cell','nTPM'],
 'ihc':['Gene','Tissue','Cell type','Level','Reliability'],
 'ihc_metadata':['Tissue','Organ'],
}
hpa_data={}
for key, required in schemas.items():
    hpa_data[key],audit=hpa_read(_hpa_paths[key],required,_hpa_ids if 'Gene' in required else None)
    _hpa_manifest[key].update(audit)

# D. Audit version differences and name mismatches, never map by a symbol alone.
# A stable ID with a changed name is retained but flagged for human review.
identity_rows=[]
for key in ('tissue','cluster','immune','ihc'):
    data=hpa_data[key]
    data.rename(columns={'Gene':'human_ensembl_gene_id'},inplace=True)
    names=(data.groupby('human_ensembl_gene_id')['Gene name'].agg(lambda s:sorted(set(s)-{''})).to_dict() if 'Gene name' in data else {})
    present=set(data.human_ensembl_gene_id)
    for gene_id in hpa_target_ids:
        current=sorted(human_id_to_symbols.get(gene_id,set()))
        observed=names.get(gene_id,[])
        status=('ID_not_in_source' if gene_id not in present else 'exact_ID_symbol_not_reported' if not observed else 'exact_ID_symbol_agrees' if set(observed)<=set(current) else 'exact_ID_symbol_review_required')
        identity_rows.append({'source':key,'human_ensembl_gene_id':gene_id,'identity_status':status,'HPA_symbols_json':json.dumps(observed),'current_symbols_json':json.dumps(current),'HPA_ensembl_release':HPA_ENSEMBL_RELEASE,'current_ensembl_release':ENSEMBL_RELEASE})
hpa_identity_audit=pd.DataFrame(identity_rows)
# Exclude name-discordant rows from quantitative summaries until reviewed.
for key in ('tissue','cluster','immune','ihc'):
    review=set(hpa_identity_audit.loc[hpa_identity_audit.source.eq(key)&hpa_identity_audit.identity_status.eq('exact_ID_symbol_review_required'),'human_ensembl_gene_id'])
    hpa_data[key]['identity_review_required']=hpa_data[key].human_ensembl_gene_id.isin(review)

# E. Bulk normal-tissue RNA: within-source descriptive context, not a brain-EC specificity measure.
def annotate_organ(frame, metadata):
    x=frame.copy(); m=metadata[['Tissue','Organ']].copy()
    x['tissue_key']=hpa_label(x.Tissue); m['tissue_key']=hpa_label(m.Tissue)
    m['organ_key']=hpa_label(m.Organ)
    m=m[['tissue_key','organ_key']].drop_duplicates()
    assert m.tissue_key.is_unique
    x=x.merge(m,on='tissue_key',how='left',validate='many_to_one')
    x['anatomical_context']=np.select([x.organ_key.isna(),x.organ_key.eq('brain'),x.organ_key.eq('eye')|x.tissue_key.eq('retina')],['unclassified','CNS','ocular'],default='non_CNS_non_ocular')
    return x

hpa_tissue=annotate_organ(hpa_data['tissue'],hpa_data['tissue_metadata'])
hpa_tissue['nTPM']=hpa_numeric(hpa_tissue,'nTPM')
assert not hpa_tissue.duplicated(['human_ensembl_gene_id','tissue_key']).any()
hpa_tissue['context']=hpa_tissue.Tissue
peripheral_tissue=hpa_tissue.loc[hpa_tissue.anatomical_context.eq('non_CNS_non_ocular')&~hpa_tissue.identity_review_required].copy()
hpa_tissue_summary=hpa_rna_summary(peripheral_tissue,'HPA_peripheral_tissue','context',hpa_target_ids)
critical_tissue_names=['lung','liver','kidney','heart muscle','spleen','bone marrow','pancreas','small intestine','skin','skeletal muscle']
critical_tissue=peripheral_tissue.loc[peripheral_tissue.tissue_key.isin(critical_tissue_names)]
critical_tissue_matrix=critical_tissue.pivot(index='human_ensembl_gene_id',columns='tissue_key',values='nTPM').reindex(index=hpa_target_ids,columns=critical_tissue_names)
critical_tissue_matrix.columns=['HPA_tissue_'+x.replace(' ','_')+'_nTPM' for x in critical_tissue_matrix.columns]
critical_tissue_matrix=critical_tissue_matrix.reset_index()

# F. Tissue-resolved single-cell cluster context.
# HPA annotations, not new independent donor-level inference.
hpa_cluster=hpa_data['cluster'].copy()
cluster_metadata=hpa_data['cluster_metadata'].copy()
for df in (hpa_cluster,cluster_metadata):
    df['tissue_key']=hpa_label(df.Tissue); df['cluster_key']=hpa_label(df.Cluster); df['cell_type_key']=hpa_label(df['Cell type'])
cluster_metadata['cluster_n_cells']=hpa_numeric(cluster_metadata,'Cell count')
cluster_metadata['cluster_annotation_reliability']=hpa_label(cluster_metadata['Annotation reliability'])
cluster_metadata['cluster_in_aggregation']=hpa_label(cluster_metadata['Included in aggregation']).eq('yes')
cluster_metadata['cell_type_group_key']=hpa_label(cluster_metadata['Cell type group'])
cluster_meta_fields=['tissue_key','cluster_key','cell_type_key','cluster_n_cells','cluster_annotation_reliability','cluster_in_aggregation','cell_type_group_key']
cluster_metadata=cluster_metadata[cluster_meta_fields].drop_duplicates()
assert not cluster_metadata.duplicated(['tissue_key','cluster_key']).any()
hpa_cluster=hpa_cluster.merge(cluster_metadata,on=['tissue_key','cluster_key','cell_type_key'],how='left',validate='many_to_one',indicator='metadata_match')
hpa_cluster['nTPM']=hpa_numeric(hpa_cluster,'nTPM')
assert not hpa_cluster.duplicated(['human_ensembl_gene_id','tissue_key','cluster_key']).any()
hpa_cluster['cluster_supported']=(hpa_cluster.metadata_match.eq('both')&hpa_cluster.cluster_n_cells.ge(HPA_MIN_CLUSTER_CELLS)&hpa_cluster.cluster_annotation_reliability.isin(HPA_ACCEPTED_ANNOTATIONS)&hpa_cluster.cluster_in_aggregation.eq(True))
hpa_cluster['non_CNS_non_ocular']=~hpa_cluster.tissue_key.isin({'brain','eye','retina','spinal cord','cerebral cortex'})
hpa_cluster['blood_vascular_EC']=hpa_cluster.cell_type_key.eq('endothelial cells')
hpa_cluster['context']=hpa_cluster.Tissue+' | '+hpa_cluster['Cell type']+' | '+hpa_cluster.Cluster
hpa_supported_clusters=hpa_cluster.loc[hpa_cluster.cluster_supported&~hpa_cluster.identity_review_required].copy()
# Summarize all covered extra-CNS vascular endothelial clusters, excluding lymphatic ECs.
hpa_peripheral_EC=hpa_supported_clusters.loc[hpa_supported_clusters.non_CNS_non_ocular&hpa_supported_clusters.blood_vascular_EC].copy()
assert len(hpa_peripheral_EC)>0,'No supported peripheral EC clusters; inspect metadata and labels.'
hpa_peripheral_EC_summary=hpa_rna_summary(hpa_peripheral_EC,'HPA_peripheral_EC_cluster','context',hpa_target_ids)
# Preserve the tissue-specific maximum, not an average over arbitrary cluster counts.
hpa_peripheral_EC_tissue=(hpa_peripheral_EC.groupby(['human_ensembl_gene_id','tissue_key'],as_index=False).agg(max_cluster_nTPM=('nTPM','max'),n_observed_clusters=('nTPM','count')))
hpa_peripheral_EC_tissue.loc[hpa_peripheral_EC_tissue.n_observed_clusters.eq(0),'max_cluster_nTPM']=np.nan
hpa_peripheral_EC_tissue_matrix=hpa_peripheral_EC_tissue.pivot(index='human_ensembl_gene_id',columns='tissue_key',values='max_cluster_nTPM').reindex(hpa_target_ids)
hpa_other_cluster_summary=hpa_rna_summary(hpa_supported_clusters.loc[hpa_supported_clusters.non_CNS_non_ocular&~hpa_supported_clusters.blood_vascular_EC],'HPA_other_peripheral_cluster','context',hpa_target_ids)
# Optional brain cluster evidence remains separate from Wälchli CPM.
hpa_brain_EC_summary=hpa_rna_summary(hpa_supported_clusters.loc[hpa_supported_clusters.tissue_key.eq('brain')&hpa_supported_clusters.blood_vascular_EC],'HPA_brain_EC_cluster','context',hpa_target_ids)

# G. Immune-cell transcript context. Total PBMC is retained in source tables but excluded from maxima.
hpa_immune=hpa_data['immune'].copy()
hpa_immune['nTPM']=hpa_numeric(hpa_immune,'nTPM')
hpa_immune['context']=hpa_immune['Immune cell']
assert not hpa_immune.duplicated(['human_ensembl_gene_id','context']).any()
hpa_immune_summary=hpa_rna_summary(hpa_immune.loc[~hpa_label(hpa_immune.context).eq('total pbmc')&~hpa_immune.identity_review_required],'HPA_immune_cell','context',hpa_target_ids)

# H. IHC: separate categories and reliability. Never interpret negative staining as no target.
hpa_ihc=annotate_organ(hpa_data['ihc'],hpa_data['ihc_metadata'])
hpa_ihc['level_key']=hpa_label(hpa_ihc.Level)
hpa_ihc['reliability_key']=hpa_label(hpa_ihc.Reliability)
hpa_ihc['staining_ordinal']=hpa_ihc.level_key.map({'not detected':0,'low':1,'medium':2,'high':3})
hpa_ihc['higher_support']=hpa_ihc.reliability_key.isin(HPA_IHC_HIGHER_SUPPORT)
hpa_ihc['context']=hpa_ihc.Tissue+' | '+hpa_ihc['Cell type']
_ihc_valid=hpa_ihc.loc[hpa_ihc.anatomical_context.eq('non_CNS_non_ocular')&hpa_ihc.higher_support&hpa_ihc.staining_ordinal.notna()&~hpa_ihc.identity_review_required].copy()
# Use max annotated category, not an average or a quantitative protein abundance estimate.
_ihc_rows=[]
for gid,g in _ihc_valid.groupby('human_ensembl_gene_id'):
    peak=g.staining_ordinal.max()
    _ihc_rows.append({'human_ensembl_gene_id':gid,'HPA_IHC_n_higher_support_records':len(g),'HPA_IHC_n_higher_support_tissues':g.tissue_key.nunique(),'HPA_IHC_n_medium_high_tissues':g.loc[g.staining_ordinal.ge(2),'tissue_key'].nunique(),'HPA_IHC_max_level':{0:'Not detected',1:'Low',2:'Medium',3:'High'}[peak],'HPA_IHC_peak_context':'; '.join(sorted(set(g.loc[g.staining_ordinal.eq(peak),'context'])))})
_ihc_cols=['human_ensembl_gene_id','HPA_IHC_n_higher_support_records','HPA_IHC_n_higher_support_tissues','HPA_IHC_n_medium_high_tissues','HPA_IHC_max_level','HPA_IHC_peak_context']
hpa_ihc_summary=pd.DataFrame(_ihc_rows,columns=_ihc_cols).set_index('human_ensembl_gene_id').reindex(hpa_target_ids).reset_index()
for col in ['HPA_IHC_n_higher_support_records','HPA_IHC_n_higher_support_tissues']:
    hpa_ihc_summary[col]=hpa_ihc_summary[col].fillna(0).astype(int)

# I. Merge evidence WITHOUT ranking, weighting, off-target gate or cross-resource ratios.
hpa_systemic_gene_evidence=pd.DataFrame({'human_ensembl_gene_id':hpa_target_ids})
for frame in [hpa_tissue_summary,critical_tissue_matrix,hpa_peripheral_EC_summary,hpa_other_cluster_summary,hpa_brain_EC_summary,hpa_immune_summary,hpa_ihc_summary]:
    assert frame.human_ensembl_gene_id.is_unique
    hpa_systemic_gene_evidence=hpa_systemic_gene_evidence.merge(frame,on='human_ensembl_gene_id',how='left',validate='one_to_one')
assert len(hpa_systemic_gene_evidence)==len(hpa_target_ids)
hpa_systemic_gene_evidence['HPA_systemic_context_status']=np.select([
    hpa_systemic_gene_evidence.HPA_peripheral_tissue_n_observed.eq(0),
    hpa_systemic_gene_evidence.HPA_peripheral_EC_cluster_n_observed.eq(0),
    hpa_systemic_gene_evidence.HPA_immune_cell_n_observed.eq(0)],
    ['bulk_tissue_unassessable','peripheral_EC_unassessable','immune_context_unassessable'],default='RNA_context_available')
hpa_systemic_gene_evidence['HPA_IHC_context_available']=hpa_systemic_gene_evidence.HPA_IHC_n_higher_support_records.gt(0)
receptor_feasibility_evidence_v13=_hpa_base.merge(hpa_systemic_gene_evidence,on='human_ensembl_gene_id',how='left',validate='many_to_one')
assert len(receptor_feasibility_evidence_v13)==len(_hpa_base)
assert not receptor_feasibility_evidence_v13.duplicated(feasibility_pair_keys).any()
pd.testing.assert_frame_equal(_hpa_base.reset_index(drop=True),receptor_feasibility_evidence_v13[_hpa_base.columns].reset_index(drop=True),check_dtype=False)
receptor_feasibility_evidence_v13.loc[receptor_feasibility_evidence_v13.human_ensembl_gene_id.isna(),'HPA_systemic_context_status']='human_mapping_unresolved'
receptor_feasibility_evidence_v13['systemic_off_target_evidence']=np.where(receptor_feasibility_evidence_v13.HPA_systemic_context_status.eq('RNA_context_available'),'HPA systemic RNA context available; protein accessibility and ligand exposure unvalidated','HPA systemic context incomplete; see individual source coverage')
receptor_biological_review_worklist_v13=receptor_feasibility_evidence_v13.loc[receptor_feasibility_evidence_v13.structural_assessment_scope].copy()
assert set(receptor_biological_review_worklist_v13.gene)==set(structural_scope_genes)

# J. Diagnostic coverage, including every omitted/mismatched identity.
hpa_coverage_audit=hpa_identity_audit.groupby(['source','identity_status']).size().rename('n_target_genes').reset_index()
hpa_cluster_support_audit=(hpa_cluster[['tissue_key','cluster_key','cell_type_key','cluster_supported','cluster_n_cells','cluster_annotation_reliability','metadata_match']].drop_duplicates().sort_values(['tissue_key','cluster_key']))
hpa_tissue_coverage_audit=hpa_tissue[['Tissue','organ_key','anatomical_context']].drop_duplicates()
hpa_ihc_coverage_audit=hpa_ihc[['Tissue','organ_key','anatomical_context']].drop_duplicates()
print('\nHPA exact-ID mapping coverage:'); display(hpa_coverage_audit)
print('\nSupported non-CNS/non-ocular blood vascular EC clusters by tissue:')
display(hpa_peripheral_EC[['tissue_key','cluster_key']].drop_duplicates().groupby('tissue_key').size().rename('n_clusters'))
print('\nCluster metadata matching:'); display(hpa_cluster_support_audit.metadata_match.value_counts())
print('\nTissue taxonomy exceptions:'); display(hpa_tissue_coverage_audit.loc[hpa_tissue_coverage_audit.anatomical_context.eq('unclassified')]); display(hpa_ihc_coverage_audit.loc[hpa_ihc_coverage_audit.anatomical_context.eq('unclassified')])
print('\nIHC level and reliability source labels:'); display(pd.crosstab(hpa_ihc.Level,hpa_ihc.Reliability))
print('\nStructural-scope systemic expression evidence:'); display(receptor_biological_review_worklist_v13.HPA_systemic_context_status.value_counts(dropna=False))
selected_systemic_genes=['Procr','Tnfrsf1a','Ackr1','Tm4sf1','Itm2a','Ecscr','Ace','Cd93','Bst2','Entpd1','Adam15','Il4ra','Cd200','Itgb1','Ly6e','Ly6a','Ly6c1','Tfrc','Lrp1','Insr','Igf1r','Slc3a2','Lrp8','Bsg','Car4']
selected_systemic_evidence=receptor_biological_review_worklist_v13.loc[receptor_biological_review_worklist_v13.gene.isin(selected_systemic_genes)].copy()
print('\nSelected tissue and peripheral EC expression (separate HPA nTPM scales):')
display(selected_systemic_evidence[['gene','human_gene_symbol','HPA_peripheral_tissue_max_nTPM','HPA_peripheral_tissue_peak_context','HPA_peripheral_tissue_n_ge_10','HPA_peripheral_EC_cluster_max_nTPM','HPA_peripheral_EC_cluster_peak_context','HPA_peripheral_EC_cluster_n_observed']].round(2))
print('\nSelected immune and protein evidence:')
display(selected_systemic_evidence[['gene','human_gene_symbol','HPA_immune_cell_max_nTPM','HPA_immune_cell_peak_context','HPA_IHC_max_level','HPA_IHC_n_medium_high_tissues','HPA_IHC_n_higher_support_tissues']].round(2))
print('\nSelected peripheral endothelial tissue-specific maxima:')
_display_ids=selected_systemic_evidence.human_ensembl_gene_id.dropna().unique()
display(hpa_peripheral_EC_tissue_matrix.reindex(_display_ids).round(2))

# K. Export provenance and complete source tables before activating results.
hpa_tables={'HPA_identity_audit':hpa_identity_audit,'HPA_identity_coverage':hpa_coverage_audit,'HPA_tissue_RNA_source':hpa_tissue,'HPA_cluster_RNA_source':hpa_cluster,'HPA_cluster_support_audit':hpa_cluster_support_audit,'HPA_tissue_coverage_audit':hpa_tissue_coverage_audit,'HPA_IHC_coverage_audit':hpa_ihc_coverage_audit,'HPA_peripheral_EC_tissue_expression':hpa_peripheral_EC_tissue,'HPA_immune_RNA_source':hpa_immune,'HPA_IHC_source':hpa_ihc,'HPA_systemic_gene_evidence':hpa_systemic_gene_evidence,'EC_integrated_receptor_feasibility_evidence_v1_3':receptor_feasibility_evidence_v13,'EC_biological_review_worklist_v1_3':receptor_biological_review_worklist_v13,'EC_selected_systemic_expression_context':selected_systemic_evidence}
for name,frame in hpa_tables.items(): frame.to_csv(out_dir/f'{name}.csv',index=False)
_hpa_provenance={'HPA_release':HPA_RELEASE,'HPA_Ensembl_release':HPA_ENSEMBL_RELEASE,'current_Ensembl_release':ENSEMBL_RELEASE,'files':_hpa_manifest,'policy':'Exact stable IDs only; source symbol conflicts withheld from summaries pending review. No missing-to-zero imputation.','descriptive_nTPM_floors':list(HPA_RNA_FLOORS),'min_cluster_cells':HPA_MIN_CLUSTER_CELLS,'accepted_cluster_reliability':sorted(HPA_ACCEPTED_ANNOTATIONS),'IHC_higher_support_categories':sorted(HPA_IHC_HIGHER_SUPPORT),'limitations':['HPA consensus bulk RNA, cluster RNA and immune RNA have distinct normalization and sampling; no ratios across resources or against Wälchli CPM.','Cluster maxima are descriptive; not independent donors or systemic vascular coverage.','Normal tissue datasets do not capture stroke-induced peripheral changes.','IHC categories are ordinal and do not establish membrane localization or blood-facing accessibility.','Blood cells such as platelets and mature erythrocytes and circulating soluble receptor pools are not exhaustively assessed.','No receptor-specific safety, pharmacokinetics, or ligand off-target binding established.','No automatic exclusion or prioritization.'],'generated_utc':datetime.now(timezone.utc).isoformat()}
(out_dir/'HPA_systemic_expression_manifest.json').write_text(json.dumps(_hpa_provenance,indent=2),encoding='utf-8')
receptor_feasibility_evidence=receptor_feasibility_evidence_v13
feasibility=receptor_feasibility_evidence
receptor_biological_review_worklist=receptor_biological_review_worklist_v13
print('\nHuman genes:',len(hpa_systemic_gene_evidence),'| integrated rows:',len(receptor_feasibility_evidence))
print('Systemic expression evidence exported. No final ranking performed.')
