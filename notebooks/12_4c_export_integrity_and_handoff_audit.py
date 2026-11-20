"""OmiPep v0.2 | Module 01 | Cell 12.4c.

Read-only audit of REAL 12.4 exports. Requires 12.4 executed in the same
Jupyter kernel. Does NOT regenerate exports, re-run analyses, approve targets,
or touch the manual adjudication sheets.

Jupyter:
    %run -i 12_4c_export_integrity_and_handoff_audit.py
"""

from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import pandas as pd

print("\n=== OmiPep Module 01 | Cell 12.4c: export integrity and handoff audit ===")
_errors_124c = []
_warnings_124c = []

def _check_124c(condition, detail):
    if bool(condition):
        print("PASS |", detail)
    else:
        print("FAIL |", detail)
        _errors_124c.append(detail)

def _warn_124c(detail):
    print("REVIEW |", detail)
    _warnings_124c.append(detail)

_repo_124c = Path(__file__).resolve().parent.parent
_processed_124c = _repo_124c / "data" / "processed"
_check_124c((_repo_124c / "notebooks").is_dir() and _processed_124c.is_dir(),
            "Repository layout and data/processed directory present")

# An explicitly enumerated packet, rather than a broad filename glob.
_files_124c = {
    "docket": "EC_receptor_12_4_full_relationship_docket_v1.csv",
    "focal": "EC_receptor_12_4_focal_comparison_v1.csv",
    "provisional": "EC_receptor_12_4_provisional_exploratory_hypotheses_v1.csv",
    "alternative": "EC_receptor_12_4_alternative_hypotheses_v1.csv",
    "reference": "EC_receptor_12_4_reference_and_context_v1.csv",
    "decision": "EC_receptor_12_4_manuscript_decision_table_v1.csv",
    "summary": "EC_receptor_12_4_shortlist_packet_summary_v1.csv",
    "manifest": "EC_receptor_12_4_shortlist_packet_manifest_v1.json",
    "handoff": "EC_receptor_12_4_downstream_handoff_BLOCKED_v1.json",
    "markdown": "EC_receptor_12_4_manuscript_decision_table_v1.md",
}
for _label_124c, _name_124c in _files_124c.items():
    _path_124c = _processed_124c / _name_124c
    _check_124c(_path_124c.is_file() and _path_124c.stat().st_size > 0,
                f"Required packet file present and nonempty: {_label_124c}")

if any(not (_processed_124c / name).is_file() for name in _files_124c.values()):
    raise AssertionError("Missing packet artifacts: audit stopped before file parsing")

# Check the REAL CSV header before pandas can rename duplicate names to *.1.
_tables_124c = {}
for _label_124c, _name_124c in _files_124c.items():
    if not _name_124c.endswith(".csv"):
        continue
    _path_124c = _processed_124c / _name_124c
    with _path_124c.open("r", encoding="utf-8-sig", newline="") as _handle_124c:
        _headers_124c = next(csv.reader(_handle_124c), [])
    _check_124c(bool(_headers_124c) and len(_headers_124c) == len(set(_headers_124c)),
                f"{_label_124c}: raw CSV header has no duplicates")
    _df_124c = pd.read_csv(_path_124c, dtype=str, keep_default_na=False,
                             encoding="utf-8-sig", low_memory=False)
    _tables_124c[_label_124c] = _df_124c
    print(f"       {_label_124c}: {_df_124c.shape[0]} rows x {_df_124c.shape[1]} cols")

# Shape and distinct-entity invariants established by actual Cell 12.4 output.
for _label_124c, _expected_124c in (
    ("docket", 125), ("focal", 22), ("provisional", 3), ("reference", 9)
):
    _check_124c(len(_tables_124c[_label_124c]) == _expected_124c,
                f"{_label_124c}: exactly {_expected_124c} exported relationships")

for _label_124c, _expected_124c in (
    ("docket", 114), ("alternative", 10),
    ("reference", 9), ("provisional", 3)
):
    _tab_124c = _tables_124c[_label_124c]
    if "gene" in _tab_124c:
        _check_124c(_tab_124c["gene"].nunique() == _expected_124c,
                    f"{_label_124c}: {_expected_124c} distinct mouse genes")
    else:
        _warn_124c(f"{_label_124c}: 'gene' field absent; gene uniqueness not testable")

_p_124c = _tables_124c["provisional"]
if "gene" in _p_124c:
    _check_124c(set(_p_124c["gene"]) == {"Ecscr", "Tm4sf1", "Procr"},
                "Exported provisional gene set exactly Ecscr/Tm4sf1/Procr")
else:
    _errors_124c.append("Provisional file lacks required gene field")

# Status invariants must also be true in the exported files (not just memory).
_d_124c = _tables_124c["docket"]
for _field_124c, _required_124c in (
    ("docking_fragment_status", "not_approved"),
    ("final_selection_status", "not_selected"),
):
    _check_124c(_field_124c in _d_124c.columns,
                f"Full docket includes {_field_124c}")
    if _field_124c in _d_124c:
        _check_124c(_d_124c[_field_124c].eq(_required_124c).all(),
                    f"All 125 exported {_field_124c} values are {_required_124c}")
    for _label_124c in ("focal", "provisional", "alternative", "reference", "decision"):
        _tab_124c = _tables_124c[_label_124c]
        if _field_124c in _tab_124c:
            _check_124c(_tab_124c[_field_124c].eq(_required_124c).all(),
                        f"{_label_124c}: {_field_124c} remains {_required_124c}")

if "manual_review_status" in _p_124c:
    _check_124c(_p_124c["manual_review_status"].str.casefold().eq("pending").all(),
                "All three exported receptor hypotheses retain pending manual review")
else:
    _warn_124c("Provisional export omits manual_review_status; see source dossier")

# Compare exported gene occurrence counts with existing in-kernel source tables.
for _label_124c, _memory_name_124c in (
    ("docket", "dossier_124"),
    ("focal", "focus_124"),
    ("provisional", "provisional_124"),
    ("reference", "reference_124"),
):
    _memory_124c = globals().get(_memory_name_124c)
    _export_124c = _tables_124c[_label_124c]
    _check_124c(isinstance(_memory_124c, pd.DataFrame),
                f"Existing in-kernel source table {_memory_name_124c} present")
    if isinstance(_memory_124c, pd.DataFrame) and "gene" in _memory_124c and "gene" in _export_124c:
        _check_124c(Counter(_memory_124c["gene"].fillna("").astype(str)) ==
                    Counter(_export_124c["gene"].astype(str)),
                    f"{_label_124c}: gene-by-gene relationship multiplicities match memory")

# The compact summary is the key manuscript-facing set of numeric claims.
_summary_124c = _tables_124c["summary"]
_expected_metrics_124c = {
    "full_relationship_rows_retained": 125,
    "structural_scope_mouse_genes": 114,
    "provisional_exploratory_mouse_genes_not_selected": 3,
    "provisional_exploratory_relationship_rows": 3,
    "alternative_exploratory_mouse_genes_not_excluded": 10,
    "reference_or_context_mouse_genes": 9,
    "focal_relationship_rows": 22,
    "completed_documented_manual_relationship_rows_not_claim_verified": 0,
    "manual_entries_requiring_correction": 0,
    "primary_literature_review_pending_relationship_rows": 73,
    "approved_receptors": 0,
    "approved_docking_fragments": 0,
    "validated_generated_peptide_transport": 0,
}
_check_124c({"metric", "value"}.issubset(_summary_124c.columns),
            "Summary CSV contains metric and value columns")
if {"metric", "value"}.issubset(_summary_124c.columns):
    _metrics_124c = dict(zip(_summary_124c["metric"], _summary_124c["value"]))
    for _key_124c, _expected_124c in _expected_metrics_124c.items():
        _raw_124c = _metrics_124c.get(_key_124c)
        try:
            _actual_124c = int(_raw_124c)
        except (ValueError, TypeError):
            _actual_124c = None
        _check_124c(_actual_124c == _expected_124c,
                    f"Summary { _key_124c } = {_expected_124c}")

# Parse and DISPLAY the full structured blockade and manifest: no guessed schema.
for _label_124c in ("handoff", "manifest"):
    _path_124c = _processed_124c / _files_124c[_label_124c]
    try:
        _payload_124c = json.loads(_path_124c.read_text(encoding="utf-8"))
        _check_124c(isinstance(_payload_124c, dict),
                    f"{_label_124c}: parseable JSON object")
        print(f"\n--- {_label_124c.upper()} JSON (read-only) ---")
        print(json.dumps(_payload_124c, indent=2, ensure_ascii=False)[:4000])
        if _label_124c == "handoff":
            _text_124c = json.dumps(_payload_124c, ensure_ascii=False).lower()
            if "blocked" in _text_124c:
                _check_124c(True, "Handoff JSON explicitly contains BLOCKED status")
            else:
                _warn_124c("No literal 'blocked' in handoff payload; manually verify its gate logic")
    except (OSError, ValueError) as _exc_124c:
        _check_124c(False, f"{_label_124c}: JSON parse failed ({_exc_124c})")

# Markdown content and safety language (absence of word 'blocked' alone is NOT a failed gate).
_md_124c = (_processed_124c / _files_124c["markdown"]).read_text(encoding="utf-8")
for _symbol_124c in ("ECSCR", "TM4SF1", "PROCR"):
    _check_124c(_symbol_124c in _md_124c,
                f"Manuscript Markdown mentions {_symbol_124c}")
_check_124c("pending" in _md_124c.lower(),
            "Manuscript Markdown discloses pending reviews")
print("\n--- MANUSCRIPT MARKDOWN (first 2800 characters) ---")
print(_md_124c[:2800])
if len(_md_124c) > 2800:
    print(f"[... {len(_md_124c) - 2800} additional characters omitted]")

# Can document that the manual sheet predates export, but WITHOUT a trusted
# PRE-12.4 content digest cannot prove byte-for-byte preservation.
_template_124c = _processed_124c / "EC_receptor_manual_adjudication_template_v1.csv"
_export_times_124c = [(_processed_124c / name).stat().st_mtime for name in _files_124c.values()]
if _template_124c.exists():
    _digest_124c = hashlib.sha256(_template_124c.read_bytes()).hexdigest()
    _stamp_124c = datetime.fromtimestamp(_template_124c.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    print(f"\nMANUAL SHEET | timestamp {_stamp_124c} | SHA256 {_digest_124c}")
    _check_124c(_template_124c.stat().st_mtime <= min(_export_times_124c),
                "Manual adjudication template modification time predates current packet")
else:
    _warn_124c("Manual adjudication template not found")
print("LIMITATION | No verified pre-run SHA256 baseline: do not claim cryptographic preservation.")
print("LIMITATION | This checks content consistency, not correctness of external biological evidence.")
print("LIMITATION | Downstream docking/model handoff remains prohibited.")

if _errors_124c:
    raise AssertionError("Cell 12.4c audit failed: " + "; ".join(_errors_124c))
print(f"\nPASS | Export integrity checks passed; {_warnings_124c.__len__()} items need interpretive review.")
print("NEXT | Share output, especially HANDOFF JSON, MANIFEST JSON, and Markdown excerpt.")
