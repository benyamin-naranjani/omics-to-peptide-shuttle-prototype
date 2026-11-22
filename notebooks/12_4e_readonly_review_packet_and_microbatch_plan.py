"""
OmiPep v0.2 | Module 01 | Cell 12.4e

Purpose:
  Read-only inspection of the local 12.4d 125-row expert master,
  literature worklist and source docket. Creates a NEW derived,
  gene-aware microbatch schedule (normally <=5 relationship rows
  per session) and a metadata/audit report.

Does NOT:
  alter the 12.4d tables, the 541-field dossier, manual decisions,
  Google Sheets, approved-fragment status or any receptor selection;
  verify scientific citations or claim that a published study proves BBB transport.

Run from the existing Jupyter notebook:
    %run -i 12_4e_readonly_review_packet_and_microbatch_plan.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MAX_RELATIONSHIPS_PER_SESSION = 5
REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
INPUTS = {
    "source_docket": PROCESSED / "EC_receptor_12_4_full_relationship_docket_v1.csv",
    "expert_master": PROCESSED / "EC_receptor_12_4d_expert_review_master_PRELIMINARY_v2.csv",
    "literature_worklist": PROCESSED / "EC_receptor_12_4d_literature_review_worklist_v2.csv",
    "provenance": PROCESSED / "EC_receptor_12_4d_schema_and_provenance_v2.json",
    "handoff": PROCESSED / "EC_receptor_12_4_downstream_handoff_BLOCKED_v1.json",
}
OUTPUT_CSV = PROCESSED / "EC_receptor_12_4e_literature_microbatch_plan_v1.csv"
OUTPUT_JSON = PROCESSED / "EC_receptor_12_4e_review_preflight_audit_v1.json"


def require(condition, message):
    if not bool(condition):
        raise AssertionError("FAIL | " + message)
    print("PASS | " + message)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_strict(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    require(bool(header), f"{path.name}: nonempty header")
    require(len(header) == len(set(header)),
            f"{path.name}: unique raw CSV column names")
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", low_memory=False)


def identify_column(table, candidates, label, required=True):
    found = [candidate for candidate in candidates if candidate in table.columns]
    if not found:
        if required:
            raise AssertionError(f"{label}: none of the columns {candidates!r} are present")
        return None
    return found[0]


def normalized_bool(series):
    values = series.astype(str).str.strip().str.casefold()
    possible = {"true": True, "1": True, "yes": True, "false": False,
                "0": False, "no": False, "": None, "nan": None}
    unknown = sorted(set(values) - set(possible))
    require(not unknown, f"literature flag has only recognized values: {unknown}")
    return values.map(possible)


def gene_blocks(group, mouse_column):
    """Collect all relationships of the same mouse gene within a legacy batch.

    Preserve gene ordering by the first occurrence in the review queue.
    A multi-orthology relationship group is reviewed together even if the
    original worklist contains its relationships on noncontiguous lines.
    """
    for _, block in group.groupby(mouse_column, sort=False, dropna=False):
        yield list(block.index)


def create_sessions(group, mouse_column, limit):
    """Greedy groups of <=limit, preserving original batch order."""
    blocks = list(gene_blocks(group, mouse_column))
    # A same-gene block >limit cannot fit and requires explicit split.
    sessions = []
    current = []
    oversized = []
    for block in blocks:
        if len(block) > limit:
            oversized.append({"gene": str(group.loc[block[0], mouse_column]),
                              "rows": len(block)})
            if current:
                sessions.append(current)
                current = []
            for offset in range(0, len(block), limit):
                sessions.append(block[offset: offset + limit])
            continue
        if current and len(current) + len(block) > limit:
            sessions.append(current)
            current = []
        current.extend(block)
    if current:
        sessions.append(current)
    return sessions, oversized


def main():
    print("\n=== OmiPep M01 | 12.4e: read-only evidence packet preflight ===")
    require((REPO / "notebooks").is_dir() and PROCESSED.is_dir(),
            "local OmiPep repository structure found")

    for label, path in INPUTS.items():
        require(path.is_file() and path.stat().st_size > 0,
                f"required {label} present: {path.name}")

    docket = read_csv_strict(INPUTS["source_docket"])
    expert = read_csv_strict(INPUTS["expert_master"])
    work = read_csv_strict(INPUTS["literature_worklist"])
    provenance = json.loads(INPUTS["provenance"].read_text(encoding="utf-8"))
    handoff = json.loads(INPUTS["handoff"].read_text(encoding="utf-8"))
    require(isinstance(provenance, dict), "12.4d provenance manifest is a JSON object")
    require(isinstance(handoff, dict), "handoff is a JSON object")

    for table, label, expected_columns in (
        (docket, "original source", 541),
        (expert, "expert preview", 29),
        (work, "literature worklist", 38),
    ):
        require(len(table) == 125,
                f"{label} contains exactly 125 relationship records")
        require(table.shape[1] == expected_columns,
                f"{label} contains exactly {expected_columns} columns")
        require("relationship_id" in table, f"{label} has relationship_id")
        require(table.relationship_id.ne("").all() and
                not table.relationship_id.duplicated().any(),
                f"{label} relationship IDs are complete and unique")

    source_ids = set(docket.relationship_id)
    require(set(expert.relationship_id) == source_ids,
            "expert preview preserves all original relationship IDs")
    require(set(work.relationship_id) == source_ids,
            "literature worklist preserves all original relationship IDs")

    source_gene = identify_column(docket, ["gene", "mouse_gene"], "docket mouse gene")
    expert_gene = identify_column(expert, ["mouse_gene", "gene"], "expert mouse gene")
    work_gene = identify_column(work, ["mouse_gene", "gene"], "worklist mouse gene")
    source_map = docket.set_index("relationship_id")[source_gene]
    for label, table, gene_col in (
        ("expert", expert, expert_gene), ("worklist", work, work_gene)
    ):
        mapped = table.set_index("relationship_id")[gene_col]
        require(mapped.sort_index().equals(source_map.sort_index()),
                f"{label} mouse genes match original relationships without modification")
    require(docket[source_gene].nunique() == 114,
            "original dossier contains 114 distinct mouse genes")

    human_cols = ["human_ensembl_gene_id", "human_gene_symbol", "human_gene"]
    for human_col in human_cols:
        if human_col in docket.columns:
            for label, table in (("expert", expert), ("worklist", work)):
                if human_col in table.columns:
                    require(
                        table.set_index("relationship_id")[human_col].sort_index().equals(
                            docket.set_index("relationship_id")[human_col].sort_index()
                        ),
                        f"{label} preserves {human_col} from original dossier",
                    )

    for status, expected in (("docking_fragment_status", "not_approved"),
                             ("final_selection_status", "not_selected")):
        require(status in docket and docket[status].eq(expected).all(),
                f"all original {status} values remain {expected}")

    batch_col = identify_column(work, ["review_batch"], "worklist")
    order_col = identify_column(work,
        ["DE_review_order_NOT_merit_rank", "DE_review_order", "review_order"],
        "worklist review ordering", required=False)
    if order_col:
        order = pd.to_numeric(work[order_col], errors="coerce")
        require(order.notna().all() and order.is_unique,
                "review ordering is numeric and unique; not receptor merit")
        work = work.assign(_sort_order=order).sort_values(
            ["_sort_order", "relationship_id"], kind="stable")
    else:
        work = work.assign(_sort_order=range(1, len(work) + 1))
        print("NOTICE | No recognized review-order column: keeping file row order.")

    batches = Counter(work[batch_col])
    p = sum(n for code, n in batches.items() if str(code).startswith("LIT-P"))
    r = sum(n for code, n in batches.items() if str(code).startswith("LIT-R"))
    require(p == 73 and r == 52, "source review partition: P=73, R=52")
    require(all(1 <= size <= 10 for size in batches.values()),
            "original 12.4d batches are at most 10 relationships")
    require(len(batches) == 14, "original 14 review batch IDs are preserved")

    pending_col = identify_column(docket,
        ["primary_literature_review_pending"], "docket pending flag",
        required=False)
    if pending_col:
        pending = normalized_bool(docket[pending_col])
        require((pending == True).sum() == 73,
                "original docket explicitly flags 73 primary reviews")
        join_pending = pending.copy()
        join_pending.index = docket.relationship_id
        joined = work["relationship_id"].map(join_pending)
        p_mask = work[batch_col].str.startswith("LIT-P")
        require((joined[p_mask] == True).all() and
                (joined[~p_mask] == False).all(),
                "P/R batch assignment agrees with source primary-review flags")
    else:
        print("NOTICE | Pending flag not present in 541-column dossier; "
              "P/R partition verified against the worklist only.")

    # Read-only category audit: present source columns, not inferred evidence claims.
    category_tokens = {
        "mouse_D02": ["D02", "logFC", "FDR"],
        "human_brain_EC": ["CPM", "EC_", "human_"],
        "orthology_and_topology": ["ortholog", "orthology", "external", "topology"],
        "HPA_peripheral_tissue": ["HPA_tissue", "HPA_peripheral", "organ"],
        "HPA_peripheral_EC": ["HPA_peripheral_EC"],
        "HPA_immune": ["HPA_immune"],
        "HPA_protein_IHC": ["HPA_IHC", "IHC_", "protein"],
        "evidence_and_citations": ["citation", "DOI", "PMID", "transcytosis", "claim"],
    }
    observed_groups = {}
    for group, tokens in category_tokens.items():
        matched = [col for col in expert.columns
                   if any(tok.casefold() in col.casefold() for tok in tokens)]
        observed_groups[group] = matched
        print(f"FEATURE CHECK | {group}: {len(matched)} matching expert columns")
        if matched:
            print("    " + ", ".join(matched[:12]))

    # Generate a session schedule, not a scientific rank. Original IDs retained.
    work = work.copy()
    session_field = pd.Series(index=work.index, dtype="object")
    session_index = pd.Series(index=work.index, dtype="object")
    oversized_blocks = []
    session_summary = []
    for legacy_code in sorted(batches):
        group = work.loc[work[batch_col].eq(legacy_code)]
        groups, oversized = create_sessions(group, work_gene,
                                             MAX_RELATIONSHIPS_PER_SESSION)
        for item in oversized:
            oversized_blocks.append({"legacy_batch":legacy_code, **item})
        for local_number, indices in enumerate(groups, start=1):
            session_code = f"{legacy_code}-{local_number:02d}"
            session_field.loc[indices] = session_code
            session_index.loc[indices] = list(range(1, len(indices) + 1))
            session_summary.append({
                "session_id": session_code,
                "legacy_batch": legacy_code,
                "relationship_rows": len(indices),
                "mouse_genes": int(work.loc[indices, work_gene].nunique()),
                "relationship_ids": work.loc[indices, "relationship_id"].tolist(),
                "review_status": "not_started",
            })
    require(session_field.notna().all(),
            "all 125 records allocated exactly once to a session")
    require(all(s["relationship_rows"] <= MAX_RELATIONSHIPS_PER_SESSION
                for s in session_summary),
            f"every microbatch <= {MAX_RELATIONSHIPS_PER_SESSION} relationships")

    result = work.drop(columns=["_sort_order"]).copy()
    result["review_microbatch_id"] = session_field
    result["review_microbatch_row"] = session_index.astype("int64")
    result["source_review_batch_preserved"] = result[batch_col]
    result["literature_adjudication_status"] = "not_started"
    result["literature_adjudication_note"] = ""    # unknown, not negative evidence

    require(len(result) == 125 and result.relationship_id.is_unique,
            "derived microbatch plan retains 125 unique original IDs")
    source_files = {label: {"path": str(path.relative_to(REPO)),
                            "sha256": file_hash(path),
                            "bytes": path.stat().st_size}
                    for label, path in INPUTS.items()}
    audit = {
        "module": "OmiPep v0.2 Module 01",
        "cell": "12.4e",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "read_only_preflight_and_derived_literature_microbatches",
        "original_relationship_rows": 125,
        "original_mouse_genes": 114,
        "P_pending_relationships": 73,
        "R_existing_annotation_rechecks": 52,
        "original_review_batch_counts": dict(sorted(batches.items())),
        "microbatch_max_relationship_rows": MAX_RELATIONSHIPS_PER_SESSION,
        "microbatch_count": len(session_summary),
        "microbatch_sessions": session_summary,
        "oversized_same_gene_blocks": oversized_blocks,
        "expert_preview_feature_columns_by_category": observed_groups,
        "source_file_digests": source_files,
        "original_12_4d_manifest_top_level_keys": sorted(provenance.keys()),
        "original_handoff": handoff,
        "policy": {
            "no_data_driven_exclusion": True,
            "no_final_receptor_ranking": True,
            "no_automated_literature_verification": True,
            "no_docking_approval": True,
            "unknown_is_not_negative": True,
            "do_not_merge_microbatches_with_receptor_merit": True,
            "references_sheet_not_modified": True,
        },
    }
    # Protect earlier work: on repeat, verify outputs instead of rewriting.
    if OUTPUT_CSV.exists() or OUTPUT_JSON.exists():
        require(OUTPUT_CSV.exists() and OUTPUT_JSON.exists(),
                "both 12.4e outputs exist together, or neither does")
        previous = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        require(previous.get("source_file_digests") == source_files,
                "existing 12.4e input hashes agree; no stale audit")
        existing_csv = read_csv_strict(OUTPUT_CSV)
        require(existing_csv.fillna("").equals(
                result.astype(str).fillna("").reset_index(drop=True)),
                "existing 12.4e microbatch plan matches; no rewrite")
        print(f"VERIFIED EXISTING | {OUTPUT_CSV.relative_to(REPO)}")
        print(f"VERIFIED EXISTING | {OUTPUT_JSON.relative_to(REPO)}")
    else:
        result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        OUTPUT_JSON.write_text(json.dumps(audit, indent=2,
                                          ensure_ascii=False), encoding="utf-8")
        print(f"\nCREATED | {OUTPUT_CSV.relative_to(REPO)}")
        print(f"CREATED | {OUTPUT_JSON.relative_to(REPO)}")
    print(f"PASS | {len(session_summary)} literature microbatch sessions")
    print("NOTE | These are REVIEW sessions, not receptor evidence grades.")
    print("NOTE | No DOI, paper or primary biological endpoint reviewed in this cell.")
    print("NOTE | 12.5 / downstream receptor-model handoff remain BLOCKED.")
    print("\nFIRST SESSIONS")
    for session in session_summary[:8]:
        print(f"  {session['session_id']:<14} | rows="
              f"{session['relationship_rows']} | mouse_genes="
              f"{session['mouse_genes']}")
    print("\nNEXT | Review the audit JSON and first microbatch membership; "
          "then inspect original PMIDs/DOIs before scientific adjudication.")


if __name__ == "__main__":
    main()
