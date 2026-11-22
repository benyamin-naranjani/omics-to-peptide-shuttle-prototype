"""OmiPep v0.2 | Module 01 | Cell 12.4g -- Reviewed literature importer.

USAGE (existing notebook, after 12.4d and 12.4e):
    %run -i 12_4g_rebuild_evidence_master_from_reviewed_batches.py

Place vetted, versioned review ZIPs such as LIT-P01-01_REVIEWED_v1.zip in:
    data/processed/review_packets/

Rebuilds the cumulative publication index, paper--relationship claim ledger,
and full 125-row master with newly appended *screening* fields. Does not change
upstream scientific tables, the original 541-column dossier, citation Google
Sheet, manual decisions, approved targets, or any verified_* fields. Screening
is NOT completion of primary-paper full-text adjudication.

The input review ZIP format is produced by the reviewed batch packet generator:
    review_manifest.json (sha256_of_members, review_batch_id, relationship_ids)
    references.csv
    evidence_claims.csv
    relationship_review.csv

This cell is intentionally a generic importer, not a literature searcher and
not a receptor ranking/gating algorithm. Re-executing re-derives the same data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


REPO_ROOT_124g = Path(os.environ.get(
    "OMIPEP_REPO_ROOT", str(Path(__file__).resolve().parent.parent)
)).resolve()
PROCESSED_124g = REPO_ROOT_124g / "data" / "processed"
PACKETS_DIR_124g = PROCESSED_124g / "review_packets"
OUTPUT_DIR_124g = PROCESSED_124g / "review_outputs"
EXPERT_MASTER_SOURCE_124g = PROCESSED_124g / "EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv"
DOSSIER_SOURCE_124g = PROCESSED_124g / "EC_receptor_12_4_full_relationship_docket_v1.csv"

ZIP_PATTERN_124g = re.compile(r"^LIT-[PR]\d{2}-\d{2}_REVIEWED_v\d+\.zip$")
RELATIONSHIP_COL_124g = "relationship_id"
SOURCE_PRESERVED_COLUMNS_124g = 29

REVIEW_COLUMNS_124g = {
    "review_batch_id", "review_status", "reviewed_primary_sources",
    "native_luminal_access_summary", "binding_internalization_summary",
    "intact_BBB_transcytosis_summary", "abluminal_intact_generated_peptide_cargo_summary",
    "supporting_citation_keys", "contradictory_citation_keys",
    "n_screened_papers", "expert_uncertainties_and_next_tests",
}
REFERENCE_COLUMNS_124g = {
    "citation_key", "authors", "year", "article_title", "journal", "doi", "pmid",
    "url", "verification_status", "source_read_depth", "review_batch_id",
}
CLAIM_COLUMNS_124g = {
    "claim_id", "relationship_id", "citation_key", "endpoint", "evidence_class",
    "species_model", "ligand_or_agent", "experimental_observation",
    "applicability_limitation", "direction_or_context",
    "intact_BBB_transcytosis_test_status", "review_status", "review_batch_id",
}

DERIVED_COL_MAP_124g = {
    "review_batch_id": "literature_screened_batch_id",
    "review_status": "literature_screening_review_state",
    "reviewed_primary_sources": "literature_screened_primary_source_keys",
    "native_luminal_access_summary": "literature_screened_native_luminal_access_summary",
    "binding_internalization_summary": "literature_screened_binding_internalization_summary",
    "intact_BBB_transcytosis_summary": "literature_screened_BBB_transport_summary",
    "abluminal_intact_generated_peptide_cargo_summary": "literature_screened_abluminal_cargo_summary",
    "supporting_citation_keys": "literature_screened_contextual_support_keys",
    "contradictory_citation_keys": "literature_screened_contradictory_keys",
    "n_screened_papers": "literature_n_primary_sources_screened_NOT_verified",
    "expert_uncertainties_and_next_tests": "literature_screened_uncertainties_and_next_tests",
}


def _pass_124g(condition: bool, what: str) -> None:
    if not bool(condition):
        raise AssertionError("FAIL | " + what)
    print("PASS | " + what)


def _sha256_124g(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha_124g(path: Path) -> str:
    return _sha256_124g(path.read_bytes())


def _parse_csv_124g(content: bytes, label: str) -> pd.DataFrame:
    # Keep original IDs, DOI, FDR, zeros, missing and all source strings unchanged.
    text = content.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    _pass_124g(bool(header) and len(set(header)) == len(header),
               f"{label}: nonempty, distinct raw CSV headers")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                       low_memory=False)


def _read_file_124g(path: Path) -> pd.DataFrame:
    _pass_124g(path.is_file(), f"Required baseline exists: {path.name}")
    return _parse_csv_124g(path.read_bytes(), path.name)


def _unique_key_124g(frame: pd.DataFrame, col: str, label: str) -> None:
    _pass_124g(col in frame.columns, f"{label}: required {col} exists")
    _pass_124g(frame[col].ne("").all() and not frame[col].duplicated().any(),
               f"{label}: {col} unique and nonempty")


def _atomic_bytes_124g(path: Path, content: bytes) -> None:
    """Replace ONLY the designated derived output; never alter source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".omipep-124g-", dir=path.parent,
                                     delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def _csv_bytes_124g(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _norm_doi_124g(raw: str) -> str:
    val = str(raw).strip().lower()
    val = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", val)
    val = re.sub(r"^doi:\s*", "", val)
    return val.rstrip("/ ")


def _load_review_zip_124g(path: Path, source_ids: set[str]):
    required = {"references.csv", "evidence_claims.csv", "relationship_review.csv",
                "review_manifest.json"}
    with zipfile.ZipFile(path) as zipin:
        names = zipin.namelist()
        _pass_124g(len(names) == len(set(names)), f"{path.name}: unique ZIP member names")
        _pass_124g(required.issubset(names), f"{path.name}: required members")
        _pass_124g(not any(".." in Path(n).parts or n.startswith("/") for n in names),
                   f"{path.name}: no unsafe ZIP member paths")
        manifest = json.loads(zipin.read("review_manifest.json").decode("utf-8"))
        _pass_124g(isinstance(manifest, dict), f"{path.name}: manifest object")
        hashed = manifest.get("sha256_of_members", {})
        _pass_124g(isinstance(hashed, dict) and
                   all(n in names for n in hashed) and
                   all(_sha256_124g(zipin.read(n)) == digest
                       for n, digest in hashed.items()),
                   f"{path.name}: declared member SHA-256 digests match")
        for n in required - {"review_manifest.json"}:
            _pass_124g(n in hashed, f"{path.name}: {n} is hash-covered")
        refs = _parse_csv_124g(zipin.read("references.csv"), "references.csv")
        claims = _parse_csv_124g(zipin.read("evidence_claims.csv"), "evidence_claims.csv")
        review = _parse_csv_124g(zipin.read("relationship_review.csv"), "relationship_review.csv")

    batch_id = str(manifest.get("review_batch_id", "")).strip()
    _pass_124g(bool(re.fullmatch(r"LIT-[PR]\d{2}-\d{2}", batch_id)),
               f"{path.name}: valid review microbatch ID")
    _pass_124g(path.name.startswith(batch_id + "_REVIEWED_v"),
               f"{path.name}: filename matches manifest batch ID")
    _pass_124g(REFERENCE_COLUMNS_124g.issubset(refs.columns),
               f"{batch_id}: mandatory reference fields")
    _pass_124g(CLAIM_COLUMNS_124g.issubset(claims.columns),
               f"{batch_id}: mandatory claim evidence fields")
    _pass_124g(REVIEW_COLUMNS_124g.issubset(review.columns),
               f"{batch_id}: mandatory relationship summaries")
    _unique_key_124g(refs, "citation_key", batch_id + " references")
    _unique_key_124g(claims, "claim_id", batch_id + " claims")
    _unique_key_124g(review, RELATIONSHIP_COL_124g, batch_id + " relationships")
    review_ids = set(review[RELATIONSHIP_COL_124g])
    _pass_124g(review_ids.issubset(source_ids),
               f"{batch_id}: every relationship ID belongs to the frozen 125")
    _pass_124g(review_ids == set(manifest.get("relationship_ids", [])),
               f"{batch_id}: exact reviewed relationship membership matches manifest")
    _pass_124g(1 <= len(review_ids) <= 5,
               f"{batch_id}: review pack contains 1-5 relationships")
    _pass_124g(set(claims[RELATIONSHIP_COL_124g]).issubset(review_ids),
               f"{batch_id}: claims have known relationship membership")
    _pass_124g(set(claims["citation_key"]).issubset(set(refs["citation_key"])),
               f"{batch_id}: all claims point to references in this ZIP")
    _pass_124g(not any(str(x).strip().lower() in {"completed", "final_approved"}
                       for x in review["review_status"]),
               f"{batch_id}: no falsely finalized review status")
    for frame, label in ((refs, "references"), (claims, "claims"), (review, "review")):
        _pass_124g(frame["review_batch_id"].eq(batch_id).all(),
                   f"{batch_id}: {label} row batch tags match manifest")
    for count_key, frame in (("n_primary_papers", refs),
                             ("n_evidence_claims", claims),
                             ("n_relationships", review)):
        _pass_124g(int(manifest.get(count_key, -1)) == len(frame),
                   f"{batch_id}: {count_key} count matches manifest")
    # Each relationship citation key must be backed by a real published reference.
    for _, row in review.iterrows():
        cited = {v.strip() for col in ("reviewed_primary_sources", "supporting_citation_keys",
                                      "contradictory_citation_keys")
                 for v in str(row[col]).split(";") if v.strip()}
        _pass_124g(cited.issubset(set(refs["citation_key"])),
                   f"{batch_id}: cited keys valid for {row[RELATIONSHIP_COL_124g]}")
    return batch_id, refs, claims, review, manifest


def run_124g():
    print("\n=== OmiPep M01 | 12.4g reviewed evidence ledger + master rebuild ===")
    _pass_124g(PROCESSED_124g.is_dir(), "repo data/processed available")
    _pass_124g(PACKETS_DIR_124g.is_dir(), "review_packets directory exists")
    expert = _read_file_124g(EXPERT_MASTER_SOURCE_124g)
    dossier = _read_file_124g(DOSSIER_SOURCE_124g)
    _pass_124g(len(expert) == len(dossier) == 125,
               "baseline expert and original dossier both retain 125 relationships")
    _pass_124g(len(expert.columns) == SOURCE_PRESERVED_COLUMNS_124g,
               "exact baseline 29 expert columns preserved")
    _pass_124g(len(dossier.columns) == 541,
               "exact 541-column source dossier preserved")
    _unique_key_124g(expert, "relationship_id", "expert")
    _unique_key_124g(dossier, "relationship_id", "dossier")
    _pass_124g(set(expert.relationship_id) == set(dossier.relationship_id),
               "master and original 125 relationship IDs agree")
    _pass_124g(dossier["gene"].nunique() == 114, "114 distinct source mouse genes")
    for status, expected in (("docking_fragment_status", "not_approved"),
                             ("final_selection_status", "not_selected")):
        _pass_124g(status in dossier and dossier[status].eq(expected).all(),
                   f"all 125 {status} values remain {expected}")
    source_ids = set(dossier.relationship_id)

    files = sorted(p for p in PACKETS_DIR_124g.iterdir()
                   if p.is_file() and ZIP_PATTERN_124g.fullmatch(p.name))
    _pass_124g(bool(files), "at least one reviewed ZIP is available")

    reference_parts, claim_parts, review_parts = [], [], []
    batches = set()
    packets = []
    for p in files:
        batch_id, refs, claims, review, manifest = _load_review_zip_124g(p, source_ids)
        _pass_124g(batch_id not in batches,
                   f"no competing reviewed ZIP versions for {batch_id}")
        batches.add(batch_id)
        reference_parts.append(refs)
        claim_parts.append(claims)
        review_parts.append(review)
        packets.append({"filename": p.name, "sha256": _file_sha_124g(p),
                        "source_packet_sha256": manifest.get("source_packet_sha256", ""),
                        "batch_id": batch_id, "relationships": len(review)})

    references = pd.concat(reference_parts, ignore_index=True)
    claims = pd.concat(claim_parts, ignore_index=True)
    reviews = pd.concat(review_parts, ignore_index=True)
    _unique_key_124g(claims, "claim_id", "cumulative claims")
    _unique_key_124g(reviews, "relationship_id", "cumulative reviewed relationships")

    # Publication deduplication: DOI first, PMID fallback; collision is fatal if
    # a different citation key points to same normalized identifier.
    references["normalized_doi"] = references["doi"].map(_norm_doi_124g)
    references["publication_id"] = references.apply(
        lambda row: "DOI:" + row["normalized_doi"] if row["normalized_doi"]
                    else "PMID:" + str(row["pmid"]).strip(), axis=1)
    _pass_124g(references.publication_id.ne("PMID:").all(),
               "each reference has a DOI or PMID")
    for _, frame in references.groupby("citation_key"):
        _pass_124g(frame.publication_id.nunique() == 1,
                   f"consistent DOI/PMID for citation key {frame.citation_key.iloc[0]}")
    for _, frame in references.groupby("publication_id"):
        _pass_124g(frame.citation_key.nunique() == 1,
                   f"dedup publication ID {frame.publication_id.iloc[0]}")
    unique_refs = references.drop_duplicates("citation_key", keep="first").copy()
    _pass_124g(set(claims.citation_key).issubset(set(unique_refs.citation_key)),
               "cumulative claims link only to known publication IDs")
    _pass_124g(set(claims.relationship_id).issubset(set(reviews.relationship_id)),
               "cumulative claims link only to reviewed relationships")

    # Build 125 rows, adding evidence that is EXPLICITLY marked screened.
    # Importantly, leave baseline 29 expert columns, including all verified_*
    # scientific endpoint fields, byte-for-byte equivalent after CSV parsing.
    reviewed = reviews[["relationship_id", *DERIVED_COL_MAP_124g.keys()]].copy()
    reviewed = reviewed.rename(columns=DERIVED_COL_MAP_124g)
    complete = expert.merge(reviewed, on="relationship_id", how="left",
                            sort=False, validate="one_to_one")
    derived_cols = list(DERIVED_COL_MAP_124g.values())
    for c in derived_cols:
        complete[c] = complete[c].fillna("").astype(str)
    _pass_124g(len(complete) == 125 and complete.relationship_id.is_unique,
               "final master: 125 distinct source relationships")
    _pass_124g(complete[list(expert.columns)].equals(expert),
               "all original 29 expert columns remain exactly intact")
    _pass_124g(complete.loc[complete.relationship_id.isin(reviews.relationship_id),
                            "literature_screening_review_state"].ne("").all(),
               "reviewed relationships have explicit screening status")
    _pass_124g(complete.loc[~complete.relationship_id.isin(reviews.relationship_id),
                            "literature_screening_review_state"].eq("").all(),
               "unreviewed relationships remain unknown (not negative)")
    for c in ("n_verified_supporting_papers", "n_verified_contradictory_papers"):
        if c in expert.columns:
            _pass_124g(complete[c].equals(expert[c]),
                       f"source {c} remains unchanged, not falsified by screening")

    # Keep reproducible outputs separate from immutable originals. All files
    # below are generated from the original master and reviewed ZIPs.
    OUTPUT_DIR_124g.mkdir(parents=True, exist_ok=True)
    destinations = {
        "reference_index": OUTPUT_DIR_124g / "EC_receptor_M01_reference_index_v1.csv",
        "claim_ledger": OUTPUT_DIR_124g / "EC_receptor_M01_evidence_claim_ledger_v1.csv",
        "relationship_reviews": OUTPUT_DIR_124g / "EC_receptor_M01_relationship_review_summaries_v1.csv",
        "expert_master": OUTPUT_DIR_124g / "EC_receptor_M01_expert_master_WITH_SCREENED_EVIDENCE_v1.csv",
        "manifest": OUTPUT_DIR_124g / "EC_receptor_M01_review_import_manifest_v1.json",
    }
    outputs = {
        "reference_index": _csv_bytes_124g(unique_refs),
        "claim_ledger": _csv_bytes_124g(claims),
        "relationship_reviews": _csv_bytes_124g(reviews),
        "expert_master": _csv_bytes_124g(complete),
    }
    audit = {
        "module": "OmiPep v0.2 Module 01",
        "cell": "12.4g",
        "state": "screened_sources_not_full_text_adjudicated",
        "output_policy": "derived_only_no_manual_edits_or_source_modifications",
        "n_relationships_source": 125,
        "n_relationships_with_screening": len(reviews),
        "n_relationships_pending_further_adjudication": 125,
        "n_unique_publications_screened": len(unique_refs),
        "n_evidence_claim_rows": len(claims),
        "n_reviewed_zip_packages": len(files),
        "n_approved_receptors": 0,
        "n_approved_docking_fragments": 0,
        "original_source_dossier_sha256": _file_sha_124g(DOSSIER_SOURCE_124g),
        "original_expert_master_sha256": _file_sha_124g(EXPERT_MASTER_SOURCE_124g),
        "batch_packets": packets,
        "output_sha256": {k: _sha256_124g(v) for k,v in outputs.items()},
        "notes": [
            "Source and verified_ columns have not been edited.",
            "Blank transport fields must not be interpreted as absence.",
            "Only screened summaries are appended; manual complete adjudication awaits review.",
            "This importer does not write to Google Sheets. Sheet is managed as separate deduplicated reference library.",
            "No scoring, automatic prioritization, selection or docking handoff.",
        ],
    }
    outputs["manifest"] = (json.dumps(audit, indent=2,
                                        ensure_ascii=False) + "\n").encode("utf-8")

    for key, content in outputs.items():
        # Idempotence check: unchanged output files are not rewritten.
        path = destinations[key]
        if path.is_file() and path.read_bytes() == content:
            print("UNCHANGED |", path.relative_to(REPO_ROOT_124g))
        else:
            _atomic_bytes_124g(path, content)
            print("REBUILT |", path.relative_to(REPO_ROOT_124g))

    for path in (EXPERT_MASTER_SOURCE_124g, DOSSIER_SOURCE_124g):
        _pass_124g(_file_sha_124g(path) ==
                   audit["original_expert_master_sha256" if path == EXPERT_MASTER_SOURCE_124g
                         else "original_source_dossier_sha256"],
                   f"upstream source unchanged: {path.name}")

    print(f"\nPASS | {len(files)} reviewed ZIP(s); {len(unique_refs)} unique screened publications;"
          f" {len(claims)} evidence records; {len(reviews)}/125 reviewed relationship summaries")
    print("NOTE | 125 relationships remain pending final full-text + expert adjudication")
    print("NOTE | manuscript receptor merit rank and Cell 12.5 remain BLOCKED")
    display_cols = ["DE_review_order_NOT_merit_rank", "mouse_gene", "human_gene",
                    "literature_screening_review_state",
                    "literature_n_primary_sources_screened_NOT_verified",
                    "literature_screened_BBB_transport_summary"]
    print("\nSCREENED RELATIONSHIP PREVIEW")
    print(complete.loc[complete.literature_screening_review_state.ne(""),
                       display_cols].to_string(index=False))
    try:
        from IPython.display import display
        display(complete.loc[complete.literature_screening_review_state.ne(""),
                             display_cols])
    except ImportError:
        pass
    return complete, unique_refs, claims


OMIPEP_EXPERT_MASTER_SCREENED_124g, OMIPEP_REFERENCE_INDEX_124g, OMIPEP_EVIDENCE_LEDGER_124g = run_124g()
