# 11.4 — Structural feasibility / template inventory (no docking)
# Run with %run -i in the OmiPep notebook after 11.3.
import hashlib, json, re, time
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

STRUCTURAL_FEASIBILITY_VERSION = '1.0'
STRUCTURE_CACHE = Path(human_data_dir).parent / 'structure_metadata_cache'
STRUCTURE_CACHE.mkdir(parents=True, exist_ok=True)
STRUCTURE_ENDPOINTS = {
    'pdbe': 'https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/',
    'afdb': 'https://alphafold.ebi.ac.uk/api/prediction/'
}
STRUCTURE_WORKERS = 4
SHORT_REGION_AA = 20  # review flag only
PROXIMITY_AA = 10    # sequence distance, not spatial distance


def sf_json(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def sf_records(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip():
        y = json.loads(x)
        if y is None: return []
        assert isinstance(y, list), type(y)
        return y
    return []


def sf_positions(intervals):
    return {p for a,b in intervals for p in range(a,b+1)}


def sf_segments(positions):
    runs = []
    for p in sorted(set(positions)):
        if runs and p == runs[-1][1]+1: runs[-1][1] = p
        else: runs.append([p,p])
    return runs


def sf_exact(feature, length):
    loc = feature.get('location', {})
    if loc.get('sequence'): return None
    a,b = loc.get('start',{}),loc.get('end',{})
    if any(x.get('modifier','EXACT') != 'EXACT' for x in [a,b]): return None
    a,b = a.get('value'),b.get('value')
    if type(a) is int and type(b) is int and 1 <= a <= b <= length: return a,b
    return None


def sf_pairs(features,length):
    return [(f,sf_exact(f,length)) for f in features]


def sf_overlap(a,b):
    return max(0, min(a[1],b[1])-max(a[0],b[0])+1)


def sf_af_interval(record,sequence):
    # June 2026 API field changes; older cached data may use legacy names.
    a=record.get('sequenceStart',record.get('uniprotStart'))
    b=record.get('sequenceEnd',record.get('uniprotEnd'))
    s=record.get('sequence',record.get('uniprotSequence'))
    if type(a) is not int or type(b) is not int or not 1<=a<=b<=len(sequence):
        return None,'invalid_or_missing_coordinates'
    if not isinstance(s,str): return None,'sequence_unavailable'
    s=''.join(s.split()).upper()
    if s == sequence[a-1:b] or (s==sequence and a==1 and b==len(sequence)):
        return (a,b),'exact_canonical_sequence_match'
    # Do not infer a fragment's sequence from full-length metadata.
    return None,'sequence_or_fragment_mismatch'


def sf_pdbe_interval(rec,length):
    a,b=rec.get('unp_start'),rec.get('unp_end')
    if type(a) is int and type(b) is int and 1<=a<=b<=length:
        return a,b
    return None


def sf_disulfide_scope(records, region, external, length):
    # Only exact two-residue endpoint annotations (not every residue inside a bond).
    internal=[]; outside=[]; single=[]; unresolved=[]
    for f in records:
        iv=sf_exact(f,length)
        if iv is None: unresolved.append(f); continue
        a,b=iv
        if a==b:
            if a in region: single.append(a)
        elif a in region and b in region: internal.append([a,b])
        elif (a in region) != (b in region):
            outside.append({'positions':[a,b],'other_endpoint_external':a in external and b in external})
    return internal,outside,single,unresolved

# A. Synthetic geometry, coordinate and sequence controls.
assert sf_segments({1,2,4,5})==[[1,2],[4,5]]
assert sf_overlap((2,5),(5,8))==1
assert sf_af_interval({'sequenceStart':2,'sequenceEnd':4,'sequence':'CDE'},'ACDEF')[0]==(2,4)
assert sf_af_interval({'sequenceStart':2,'sequenceEnd':4,'sequence':'ACDEF'},'ACDEF')[0] is None
assert sf_pdbe_interval({'unp_start':1,'unp_end':8},7) is None
_sf_f=lambda a,b: {'location':{'start':{'value':a},'end':{'value':b}}}
assert sf_disulfide_scope([_sf_f(2,4),_sf_f(3,7),_sf_f(2,2)],set(range(1,6)),set(range(1,9)),8)[:3]==([[2,4]],[{'positions':[3,7],'other_endpoint_external':True}],[2])
print('Synthetic structural boundary controls passed.')

# B. Freeze 11.3 inputs. Retain one protein record per species and identity;
# relationship-specific evidence is merged back only after this audit.
_sf_base=receptor_feasibility_evidence_v13.copy()
_sf_scope=_sf_base.loc[_sf_base.structural_assessment_scope].copy()
assert not _sf_base.duplicated(feasibility_pair_keys).any()
_sf_proteins=[]; _sf_ann={}; _sf_region_tables={}
for species,annot,regions,key,ids in [
    ('mouse',uniprot_annotations_v13,extracellular_regions,'gene',set(_sf_scope.gene)),
    ('human',human_protein_annotations,human_extracellular_regions,'human_ensembl_gene_id',set(_sf_scope.human_ensembl_gene_id.dropna()))
]:
    sub=annot.loc[annot[key].isin(ids)].copy()
    assert sub[key].is_unique and set(sub[key])==ids
    _sf_region_tables[species]=(regions,key)
    for row in sub.to_dict('records'):
        identity=row[key]; _sf_ann[(species,identity)]=row
        found=row['uniprot_status']=='found'
        seq=row.get('protein_sequence') if found else None
        acc=row.get('uniprot_accession') if found else None
        digest=row.get('protein_sequence_sha256') if found else None
        if found:
            assert isinstance(acc,str) and re.fullmatch(r'[A-Z0-9]+',acc),acc
            assert isinstance(seq,str) and len(seq)==int(row['protein_length'])
            assert hashlib.sha256(seq.encode()).hexdigest()==digest
        _sf_proteins.append({'species':species,'identity':identity,'uniprot_accession':acc,'sequence_sha256':digest,'sequence_version':row.get('sequence_version'),'protein_length':len(seq) if found else np.nan,'protein_identity_resolved':found})
_sf_proteins=pd.DataFrame(_sf_proteins)
print('Structural-scope protein identities:',len(_sf_proteins))

# C. Recover subunit / oligomer / complex context from the pinned UniProt cache.
# Cache payload schema is checked, and accession AND sequence must agree.
_sf_wanted=set(_sf_proteins.uniprot_accession.dropna())
_sf_cached={}; _sf_cache_conflicts=set()
_sf_root=Path(uniprot_cache_dir)
for path in sorted(_sf_root.rglob('*.json')):
    try:
        payload=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,OSError): continue
    if not isinstance(payload,dict) or payload.get('release')!=EXPECTED_UNIPROT_RELEASE: continue
    entries=payload.get('entries',[])
    if not isinstance(entries,list): continue
    for entry in entries:
        if not isinstance(entry,dict): continue
        acc=entry.get('primaryAccession')
        if acc not in _sf_wanted: continue
        seq=entry.get('sequence',{}).get('value')
        if not isinstance(seq,str): continue
        digest=hashlib.sha256(seq.encode()).hexdigest()
        k=(acc,digest)
        if k in _sf_cached and sf_json(_sf_cached[k][0])!=sf_json(entry):
            _sf_cache_conflicts.add(k)
        else: _sf_cached[k]=(entry,str(path))
# Selected human entries are also available in memory; use only if matching sequence.
# Existing 9.5 records were release-checked at acquisition.
for entry in globals().get('selected_human_entries',{}).values():
    if not isinstance(entry,dict): continue
    acc=entry.get('primaryAccession'); seq=entry.get('sequence',{}).get('value')
    if acc in _sf_wanted and isinstance(seq,str):
        k=(acc,hashlib.sha256(seq.encode()).hexdigest())
        if k in _sf_cached and sf_json(_sf_cached[k][0])!=sf_json(entry): _sf_cache_conflicts.add(k)
        elif k not in _sf_cached: _sf_cached[k]=(entry,'selected_human_entries (9.5)')

# D. Retrieve cached, timestamped structure metadata (no coordinate downloads).
def sf_fetch(source,accession):
    url=STRUCTURE_ENDPOINTS[source]+accession
    path=STRUCTURE_CACHE/f'{source}_{accession}.json'
    if path.exists():
        obj=json.loads(path.read_text(encoding='utf-8'))
        assert obj['url']==url and obj['payload_sha256']==hashlib.sha256(sf_json(obj['payload']).encode()).hexdigest()
        return obj
    error=None
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=(10,35),headers={'Accept':'application/json','User-Agent':'OmiPep/0.2 structural evidence'})
            if r.status_code==404:
                status='HTTP_404_no_record_returned'; payload=None
            else:
                r.raise_for_status(); payload=r.json(); status='success'
            result={'source':source,'accession':accession,'url':url,'resolved_url':r.url,'status':status,'http_status':r.status_code,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'payload':payload,'payload_sha256':hashlib.sha256(sf_json(payload).encode()).hexdigest()}
            tmp=path.with_suffix('.tmp'); tmp.write_text(sf_json(result),encoding='utf-8'); tmp.replace(path)
            time.sleep(0.25)
            return result
        except (requests.RequestException,ValueError) as exc:
            error=str(exc)
            if attempt<2: time.sleep(2**attempt)
    return {'source':source,'accession':accession,'url':url,'status':'retrieval_failed','error':error,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'payload':None}

_sf_jobs=[(source,acc) for acc in sorted(_sf_wanted) for source in STRUCTURE_ENDPOINTS]
_sf_api={}
with ThreadPoolExecutor(max_workers=STRUCTURE_WORKERS) as pool:
    for i,result in enumerate(pool.map(lambda job: sf_fetch(*job),_sf_jobs),1):
        _sf_api[(result['source'],result['accession'])]=result
        if i%20==0 or i==len(_sf_jobs): print(f'Structure metadata queries: {i}/{len(_sf_jobs)}',flush=True)
structure_metadata_audit=pd.DataFrame([{k:v for k,v in r.items() if k!='payload'} for r in _sf_api.values()])
print('\nStructure API retrieval status:'); display(pd.crosstab(structure_metadata_audit.source,structure_metadata_audit.status))

# E. Build exact-coordinate structural worklists; no region is a declared docking domain.
_sf_summary=[]; _sf_segments=[]; _sf_templates=[]; _sf_coverage=[]
for prot in _sf_proteins.to_dict('records'):
    species,identity=prot['species'],prot['identity']; ann=_sf_ann[(species,identity)]
    core=dict(prot)
    if not prot['protein_identity_resolved']:
        _sf_summary.append({**core,'structural_review_status':'protein_identity_unresolved'}); continue
    acc=prot['uniprot_accession']; seq=ann['protein_sequence']; length=len(seq)
    region_table,key=_sf_region_tables[species]
    rows=region_table.loc[region_table[key].eq(identity)]
    if not rows.empty:
        assert rows.uniprot_accession.eq(acc).all() and rows.protein_sequence_sha256.eq(prot['sequence_sha256']).all()
    ext=set(); explicit=set()
    for r in rows.to_dict('records'):
        a,b=int(r['start']),int(r['end']); assert 1<=a<=b<=length and seq[a-1:b]==r['segment_sequence']
        ext.update(range(a,b+1))
        if r['evidence_type']=='explicit_UniProt_topology': explicit.update(range(a,b+1))
    if ext: assert len(ext)==int(ann['n_total_external_residues'])
    all_features={name:sf_records(ann.get(col)) for name,col in {
        'tm':'transmembrane_json','intramembrane':'intramembrane_json','lipid':'lipidation_json','glycan':'glycosylation_json','ss':'disulfide_json','chain':'mature_chains_json','signal':'signal_peptide_json','propeptide':'propeptides_json'}.items()}
    iv={name:[v for _,v in sf_pairs(fs,length) if v is not None] for name,fs in all_features.items()}
    tm=iv['tm']; membrane=sf_positions(tm+iv['intramembrane'])
    anchors=[v for f,v in sf_pairs(all_features['lipid'],length) if v is not None and re.search(r'GPI|glycosylphosphatidylinositol',str(f.get('description','')),re.I)]
    unknown_glycan=sum(v is None for _,v in sf_pairs(all_features['glycan'],length))
    glycan_sites={a for a,b in iv['glycan'] if a==b}
    glycan_intervals_unresolved=sum(a!=b for a,b in iv['glycan'])
    sequons={i for i in range(1,length-1) if seq[i-1]=='N' and seq[i]!='P' and seq[i+1] in 'ST' and {i,i+1,i+2}<=ext}
    _key=(acc,prot['sequence_sha256']); cache=_sf_cached.get(_key)
    cache_status='conflicting_cache_records' if _key in _sf_cache_conflicts else 'exact_sequence_cache_match' if cache else 'cached_entry_unavailable'
    comments=[]; xrefs=[]; cache_path=None
    if cache_status=='exact_sequence_cache_match':
        entry,cache_path=cache
        comments=[c for c in entry.get('comments',[]) if c.get('commentType') in ['SUBUNIT','INTERACTION','DOMAIN','PTM','ALTERNATIVE PRODUCTS']]
        xrefs=[x for x in entry.get('uniProtKBCrossReferences',[]) if x.get('database') in ['PDB','AlphaFoldDB']]
    subunit=[c for c in comments if c.get('commentType') in ['SUBUNIT','INTERACTION']]
    # Experimental chain ranges are accession-level SIFTS spans, not resolved coordinates.
    p_rec=_sf_api[('pdbe',acc)]; p_payload=p_rec.get('payload')
    p_list=p_payload.get(acc) if isinstance(p_payload,dict) else None
    if p_rec['status']=='success':
        p_status='reported_mappings' if isinstance(p_list,list) and p_list else 'empty_mapping_response' if isinstance(p_list,list) else 'unexpected_schema'
    else: p_status=p_rec['status']
    p_models=[]
    for item in p_list if isinstance(p_list,list) else []:
        if not isinstance(item,dict): continue
        span=sf_pdbe_interval(item,length)
        valid_id=isinstance(item.get('pdb_id'),str) and isinstance(item.get('chain_id'),str)
        t={**core,'source':'PDBe_SIFTS','template_id':item.get('pdb_id'),'chain_id':item.get('chain_id'),'method':item.get('experimental_method',item.get('method')),'resolution_A':item.get('resolution'),'start':span[0] if span else np.nan,'end':span[1] if span else np.nan,'template_status':'accession_mapped_span_sequence_unverified' if span and valid_id else 'invalid_mapping_fields','source_record_json':sf_json(item)}
        _sf_templates.append(t)
        if span and valid_id: p_models.append((t,span))
    # AFDB metadata is accepted only for an explicitly identical sequence and range.
    af_rec=_sf_api[('afdb',acc)]; af_list=af_rec.get('payload')
    if af_rec['status']=='success': af_status='reported_models' if isinstance(af_list,list) and af_list else 'empty_model_response' if isinstance(af_list,list) else 'unexpected_schema'
    else: af_status=af_rec['status']
    af_models=[]
    for item in af_list if isinstance(af_list,list) else []:
        if not isinstance(item,dict): continue
        span,status=sf_af_interval(item,seq)
        model_id=item.get('modelEntityId',item.get('entryId'))
        t={**core,'source':'AFDB','template_id':model_id,'chain_id':None,'method':'prediction','resolution_A':np.nan,'start':span[0] if span else np.nan,'end':span[1] if span else np.nan,'template_status':status,'model_version':item.get('latestVersion'),'cif_url':item.get('cifUrl'),'pae_url':item.get('paeDocUrl'),'confidence_url':item.get('confidenceUrl'),'source_record_json':sf_json(item)}
        # A missing model ID is not an identifiable template.
        if not model_id: t['template_status']='model_identifier_unavailable'
        _sf_templates.append(t)
        if span and model_id: af_models.append((t,span))
    # Preserve continuous topology intervals, without merging across mature-chain cuts.
    cuts={a for a,b in iv['chain']}|{b+1 for a,b in iv['chain']}
    runs=[]
    for a,b in sf_segments(ext):
        points=[a]+sorted(x for x in cuts if a<x<=b)+[b+1]
        runs.extend((x,y-1) for x,y in zip(points,points[1:]))
    assert sf_positions(runs)==ext and sum(b-a+1 for a,b in runs)==len(ext)
    best_p=[]; best_af=[]
    for a,b in runs:
        region=set(range(a,b+1)); n=len(region)
        inside,outside,single,ss_unknown=sf_disulfide_scope(all_features['ss'],region,ext,length)
        # Sequence adjacency only; no 3D membrane-plane inference.
        nearest_tm=min((min(abs(x-y) for x in (a,b) for y in (s,e)) if not sf_overlap((a,b),(s,e)) else 0 for s,e in tm),default=None)
        nearest_anchor=min((0 if sf_overlap((a,b),(s,e)) else min(abs(x-y) for x in (a,b) for y in (s,e)) for s,e in anchors),default=None)
        region_id=f'{species}:{identity}:{a}-{b}'
        p_pct=[]; af_pct=[]
        for models,kind,vals in [(p_models,'PDBe_SIFTS',p_pct),(af_models,'AFDB',af_pct)]:
            for t,span in models:
                overlap=sf_overlap((a,b),span)
                pct=100*overlap/n; vals.append(pct)
                if overlap:
                    _sf_coverage.append({'region_id':region_id,'species':species,'identity':identity,'uniprot_accession':acc,'source':kind,'template_id':t['template_id'],'chain_id':t['chain_id'],'template_start':span[0],'template_end':span[1],'region_overlap_residues':overlap,'region_span_coverage_pct':pct,'template_status':t['template_status']})
        p_best=max(p_pct,default=np.nan); af_best=max(af_pct,default=np.nan)
        best_p.append(p_best); best_af.append(af_best)
        flags=[]
        if n<=SHORT_REGION_AA: flags.append('short_region_context_review')
        if len(region&explicit)<n: flags.append('contains_inferred_topology')
        if (nearest_tm is not None and nearest_tm<=PROXIMITY_AA) or (nearest_anchor is not None and nearest_anchor<=PROXIMITY_AA): flags.append('sequence_adjacent_to_membrane_feature')
        if outside or single or ss_unknown: flags.append('disulfide_context_review')
        if region&glycan_sites or region&sequons or unknown_glycan or glycan_intervals_unresolved: flags.append('glycan_context_review')
        if subunit or cache_status!='exact_sequence_cache_match': flags.append('complex_context_review')
        if bool(ann.get('boundary_conflict',False)): flags.append('boundary_conflict_review')
        _sf_segments.append({**core,'region_id':region_id,'start':a,'end':b,'length':n,'segment_sequence':seq[a-1:b],'n_explicit_residues':len(region&explicit),'n_inferred_residues':n-len(region&explicit),'nearest_TM_sequence_distance_aa':nearest_tm,'nearest_GPI_sequence_distance_aa':nearest_anchor,'glycan_sites_json':sf_json(sorted(region&glycan_sites)),'N_linked_sequon_starts_json':sf_json(sorted(i for i in sequons if {i,i+1,i+2}<=region)),'paired_disulfides_within_region_json':sf_json(inside),'paired_disulfides_crossing_region_json':sf_json(outside),'single_endpoint_disulfides_json':sf_json(single),'n_unresolved_disulfide_annotations_protein':len(ss_unknown),'mature_chain_overlaps_json':sf_json([list(x) for x in iv['chain'] if sf_overlap((a,b),x)]),'max_single_PDBe_span_coverage_pct':p_best,'max_single_exact_AFDB_span_coverage_pct':af_best,'review_flags_json':sf_json(flags),'region_status':'topology_interval_not_approved_model_fragment'})
    _sf_summary.append({**core,'structural_review_status':'topology_intervals_available' if ext else 'external_boundaries_unresolved','n_external_residues':len(ext) if ext else np.nan,'n_topology_regions':len(runs),'n_exact_TM_intervals':len(tm),'n_unresolved_TM_annotations':len(all_features['tm'])-len(tm),'n_exact_intramembrane_intervals':len(iv['intramembrane']),'n_exact_GPI_annotations':len(anchors),'n_mature_chain_intervals':len(iv['chain']),'n_exact_external_glycan_sites':len(ext&glycan_sites) if ext else np.nan,'n_unresolved_glycan_annotations':unknown_glycan+glycan_intervals_unresolved,'n_external_N_linked_sequons':len(sequons) if ext else np.nan,'subunit_annotation_status':('subunit_or_interaction_annotated' if subunit else 'no_subunit_comment_in_cached_entry') if cache_status=='exact_sequence_cache_match' else 'not_assessed','uniprot_cache_status':cache_status,'uniprot_cache_source':cache_path,'uniprot_context_comments_json':sf_json(comments),'uniprot_structure_xrefs_json':sf_json(xrefs),'PDBe_status':p_status,'n_PDBe_valid_span_records':len(p_models) if p_status in ['reported_mappings','empty_mapping_response'] else np.nan,'AFDB_status':af_status,'n_AFDB_exact_sequence_models':len(af_models) if af_status in ['reported_models','empty_model_response'] else np.nan,'n_regions_with_full_single_PDBe_span':sum(x==100 for x in best_p) if p_status in ['reported_mappings','empty_mapping_response'] else np.nan,'n_regions_with_full_single_exact_AFDB_span':sum(x==100 for x in best_af) if af_status in ['reported_models','empty_model_response'] else np.nan,'membrane_context_review':'required_before_fragment_selection' if membrane or anchors or iv['chain'] else 'membrane_dependence_unresolved','boundary_manual_review':ann.get('boundary_manual_review')})

# F. Integrity checks and preserve relationship-specific evidence.
receptor_structural_protein_audit=pd.DataFrame(_sf_summary)
receptor_structural_region_worklist=pd.DataFrame(_sf_segments)
receptor_structure_template_inventory=pd.DataFrame(_sf_templates)
receptor_structure_region_coverage=pd.DataFrame(_sf_coverage)
assert not receptor_structural_protein_audit.duplicated(['species','identity']).any()
assert receptor_structural_region_worklist.region_id.is_unique
assert receptor_structural_region_worklist.length.gt(0).all()
if not receptor_structure_region_coverage.empty:
    assert receptor_structure_region_coverage.region_span_coverage_pct.between(0,100).all()
    assert set(receptor_structure_region_coverage.region_id)<=set(receptor_structural_region_worklist.region_id)
receptor_feasibility_evidence_v14=_sf_base.copy()
_sf_merge_cols=['structural_review_status','n_topology_regions','n_exact_TM_intervals','n_unresolved_TM_annotations','n_exact_GPI_annotations','n_mature_chain_intervals','n_exact_external_glycan_sites','n_unresolved_glycan_annotations','n_external_N_linked_sequons','subunit_annotation_status','uniprot_cache_status','PDBe_status','n_PDBe_valid_span_records','AFDB_status','n_AFDB_exact_sequence_models','n_regions_with_full_single_PDBe_span','n_regions_with_full_single_exact_AFDB_span','membrane_context_review']
for species,key in [('mouse','gene'),('human','human_ensembl_gene_id')]:
    src=receptor_structural_protein_audit.loc[receptor_structural_protein_audit.species.eq(species),['identity']+_sf_merge_cols].rename(columns={'identity':key,**{x:f'{species}_structure_{x}' for x in _sf_merge_cols}})
    receptor_feasibility_evidence_v14=receptor_feasibility_evidence_v14.merge(src,on=key,how='left',validate='many_to_one')
assert len(receptor_feasibility_evidence_v14)==len(_sf_base)
assert not receptor_feasibility_evidence_v14.duplicated(feasibility_pair_keys).any()
pd.testing.assert_frame_equal(receptor_feasibility_evidence_v14[_sf_base.columns].reset_index(drop=True),_sf_base.reset_index(drop=True),check_dtype=False)
# Structural feasibility remains unresolved until actual coordinate and biological-context review.
receptor_feasibility_evidence_v14['structural_model_feasibility']=np.where(~receptor_feasibility_evidence_v14.structural_assessment_scope,'outside_structural_scope',np.where(receptor_feasibility_evidence_v14.mouse_structure_structural_review_status.eq('topology_intervals_available') & receptor_feasibility_evidence_v14.human_structure_structural_review_status.eq('topology_intervals_available'),'topology_and_template_inventory_review_required','protein_or_topology_review_required'))
receptor_biological_review_worklist_v14=receptor_feasibility_evidence_v14.loc[receptor_feasibility_evidence_v14.structural_assessment_scope].copy()
assert set(receptor_biological_review_worklist_v14.gene)==set(structural_scope_genes)

# G. Export and inspect. No coordinates are selected automatically.
review_genes=['Tfrc','Lrp1','Insr','Igf1r','Lrp8','Slc3a2','Itm2a','Tm4sf1','Ecscr','Ly6e','Ace','Procr','Tnfrsf1a','Ackr1','Cd93','Cd200','Itgb1','Car4','Bst2','Entpd1','Il4ra','Bsg']
selected_structural_feasibility=receptor_biological_review_worklist_v14.loc[receptor_biological_review_worklist_v14.gene.isin(review_genes)].copy()
print('\nProtein topology outcomes:'); display(pd.crosstab(receptor_structural_protein_audit.species,receptor_structural_protein_audit.structural_review_status))
print('\nCached UniProt context:'); display(pd.crosstab(receptor_structural_protein_audit.species,receptor_structural_protein_audit.uniprot_cache_status))
print('\nSelected targets — structural template inventory (human):')
display(selected_structural_feasibility[['gene','human_gene_symbol','human_structure_n_exact_TM_intervals','human_structure_n_topology_regions','human_structure_n_exact_external_glycan_sites','human_structure_n_PDBe_valid_span_records','human_structure_n_AFDB_exact_sequence_models','human_structure_n_regions_with_full_single_PDBe_span','human_structure_n_regions_with_full_single_exact_AFDB_span','human_structure_subunit_annotation_status']])
print('\nSelected human topology intervals — fragment review (not approved fragments):')
_sf_selected_ids=set(selected_structural_feasibility.human_ensembl_gene_id.dropna())
_sf_region_view=receptor_structural_region_worklist.loc[receptor_structural_region_worklist.species.eq('human') & receptor_structural_region_worklist.identity.isin(_sf_selected_ids)].merge(selected_structural_feasibility[['human_ensembl_gene_id','human_gene_symbol']].drop_duplicates('human_ensembl_gene_id'),left_on='identity',right_on='human_ensembl_gene_id',validate='many_to_one')
display(_sf_region_view[['human_gene_symbol','start','end','length','nearest_TM_sequence_distance_aa','paired_disulfides_crossing_region_json','max_single_PDBe_span_coverage_pct','max_single_exact_AFDB_span_coverage_pct','review_flags_json']])
for name,frame in {'EC_structural_protein_audit':receptor_structural_protein_audit,'EC_structural_region_worklist':receptor_structural_region_worklist,'EC_structure_template_inventory':receptor_structure_template_inventory,'EC_structure_region_span_coverage':receptor_structure_region_coverage,'EC_structure_metadata_retrieval_audit':structure_metadata_audit,'EC_selected_structural_feasibility':selected_structural_feasibility,'EC_integrated_receptor_feasibility_evidence_v1_4':receptor_feasibility_evidence_v14,'EC_biological_review_worklist_v1_4':receptor_biological_review_worklist_v14}.items():
    frame.to_csv(out_dir/f'{name}.csv',index=False)
_sf_manifest={'version':STRUCTURAL_FEASIBILITY_VERSION,'uniprot_release':EXPECTED_UNIPROT_RELEASE,'endpoints':STRUCTURE_ENDPOINTS,'cache_dir':str(STRUCTURE_CACHE),'short_region_review_flag_aa':SHORT_REGION_AA,'sequence_proximity_review_flag_aa':PROXIMITY_AA,'coordinate_convention':'1-based inclusive canonical UniProt; split continuous topology intervals at annotated mature-chain boundaries','limitations':['PDBe SIFTS accession-mapped span is not verified coordinate coverage or sequence identity. Unobserved residues and internal gaps require model-level inspection.','AFDB exact sequence/range metadata does not establish structural accuracy; coordinates, pLDDT, PAE and oligomeric arrangement not yet inspected.','No annotated subunit comment does not establish monomer state.','Sequence distance is not physical membrane proximity.','N-linked sequons are sequence motifs, not glycan occupancy. Disulfide annotations require chain and mature-state context.','Extracellular topology can represent organelle lumen; no luminal BBB accessibility inferred.','No model fragments approved, no ranking, no membrane MD decision before target and binding-site selection.','API missing/error responses are not proof of model absence.'],'generated_utc':datetime.now(timezone.utc).isoformat()}
(out_dir/'EC_structural_feasibility_manifest.json').write_text(json.dumps(_sf_manifest,indent=2),encoding='utf-8')
receptor_feasibility_evidence=receptor_feasibility_evidence_v14
feasibility=receptor_feasibility_evidence
receptor_biological_review_worklist=receptor_biological_review_worklist_v14
print('\nProtein identities:',len(receptor_structural_protein_audit),'| topology intervals:',len(receptor_structural_region_worklist),'| template records:',len(receptor_structure_template_inventory),'| overlapping template spans:',len(receptor_structure_region_coverage))
print('Structural inventory exported. No fragment approved; no final ranking performed.')
