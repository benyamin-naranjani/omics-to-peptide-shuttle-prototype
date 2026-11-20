"""OmiPep v0.2 / Module 01 / Cell 12.4b.

Read-only post-execution audit for Cell 12.4.
Run *after* the corrected Cell 12.4 in the SAME Jupyter kernel:
    %run -i 12_4b_readonly_artifact_verification.py

No file writes, state changes, re-analysis or biological approvals.
"""

from pathlib import Path
from datetime import datetime
import hashlib
import pandas as pd

print("\n=== OmiPep Module 01 | Cell 12.4b: read-only verification ===")
errors_124b = []

def verify_124b(condition, message):
    ok = bool(condition)
    print(f"{'PASS' if ok else 'FAIL'} | {message}")
    if not ok:
        errors_124b.append(message)

# Test only the existing 12.4 tables: never rebuild them.
required_124b = ("provisional_124", "focus_124", "reference_124", "dossier_124")
tables_124b = {}
for name_124b in required_124b:
    obj_124b = globals().get(name_124b)
    verify_124b(isinstance(obj_124b, pd.DataFrame), f"{name_124b} exists as DataFrame")
    if isinstance(obj_124b, pd.DataFrame):
        tables_124b[name_124b] = obj_124b
        verify_124b(obj_124b.columns.is_unique, f"{name_124b} has unique columns")
        print(f"       {name_124b}: {obj_124b.shape[0]} rows x {obj_124b.shape[1]} cols")

provisional_table_124b = tables_124b.get("provisional_124")
if provisional_table_124b is not None:
    verify_124b("gene" in provisional_table_124b.columns, "provisional_124 contains mouse gene symbols")
    if "gene" in provisional_table_124b.columns:
        observed_124b = set(provisional_table_124b["gene"].dropna().astype(str))
        verify_124b(observed_124b == {"Ecscr", "Tm4sf1", "Procr"},
                    "exactly Ecscr/Tm4sf1/Procr are provisional (not selected)")

focus_table_124b = tables_124b.get("focus_124")
if focus_table_124b is not None:
    verify_124b(len(focus_table_124b) == 22, "focal relationship table retains 22 rows")

review_table_124b = tables_124b.get("dossier_124")
if review_table_124b is not None:
    for field_124b, required_124b_value in (
        ("docking_fragment_status", "not_approved"),
        ("final_selection_status", "not_selected"),
    ):
        present_124b = field_124b in review_table_124b.columns
        verify_124b(present_124b, f"{field_124b} column exists")
        if present_124b and review_table_124b.columns.is_unique:
            verify_124b(review_table_124b[field_124b].eq(required_124b_value).all(),
                        f"all {field_124b} values are {required_124b_value}")
    verify_124b("manuscript_discussion_role" in review_table_124b.columns,
                "manuscript_discussion_role column exists")
    if "manuscript_discussion_role" in review_table_124b.columns and review_table_124b.columns.is_unique:
        verify_124b(review_table_124b["manuscript_discussion_role"].notna().all(),
                    "every dossier row has a manuscript discussion role")

# Identify export files without writing/changing them. Modification time is reported
# because an existing stale file must not be mistaken for this execution's export.
repo_124b = Path(__file__).resolve().parent.parent
verify_124b((repo_124b / "notebooks").is_dir(), "script is located inside the expected repo/notebooks directory")

match_terms_124b = ("12_4", "12.4", "provisional", "shortlist", "receptor_comparison",
                    "manuscript_packet", "adjudication")
exclude_dirs_124b = {".git", ".ipynb_checkpoints", "__pycache__", ".venv",
                    "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
found_124b = []
if repo_124b.is_dir():
    import os
    for current_124b, dirs_124b, files_124b in os.walk(repo_124b):
        dirs_124b[:] = [d for d in dirs_124b if d not in exclude_dirs_124b]
        for file_124b in files_124b:
            path_124b = Path(current_124b) / file_124b
            if path_124b.suffix.lower() in {".csv", ".tsv", ".md", ".json", ".xlsx"}:
                if any(t in file_124b.lower() for t in match_terms_124b):
                    found_124b.append(path_124b)

found_124b.sort(key=lambda p: (p.suffix.lower() != ".md", str(p)))
print(f"\nCandidate 12.4 exported tables / manuscript / adjudication files: {len(found_124b)}")
for path_124b in found_124b[:50]:
    stat_124b = path_124b.stat()
    digest_124b = hashlib.sha256(path_124b.read_bytes()).hexdigest()[:16] if stat_124b.st_size <= 15_000_000 else "large-file"
    when_124b = datetime.fromtimestamp(stat_124b.st_mtime).astimezone().isoformat(timespec="seconds")
    print(f"  {path_124b.relative_to(repo_124b)} | {stat_124b.st_size} B | modified {when_124b} | sha256 {digest_124b}")
    if path_124b.suffix.lower() == ".md" and stat_124b.st_size <= 1_000_000:
        contents_124b = path_124b.read_text(encoding="utf-8", errors="replace")
        print("    manuscript content markers:", {
            "ECSCR": "ECSCR" in contents_124b,
            "TM4SF1": "TM4SF1" in contents_124b,
            "PROCR": "PROCR" in contents_124b,
            "pending": "pending" in contents_124b.lower(),
            "blocked": "blocked" in contents_124b.lower(),
        })
if len(found_124b) > 50:
    print(f"  ... {len(found_124b) - 50} more candidates omitted")

# Listing a file, or recording its present hash, does NOT prove that it was
# exported in this run or that an earlier manual sheet was unchanged.
print("\nNOTE | File presence and current hashes do not establish prior-sheet preservation.")
print("NOTE | No receptor, docking fragment or peptide transport result is approved here.")
if errors_124b:
    raise AssertionError("Cell 12.4b audit failed: " + "; ".join(errors_124b))
print("\nPASS | All available in-memory 12.4 safety assertions passed.")
print("NEXT | Review the printed export names/timestamps/contents before progressing to 12.5.")
