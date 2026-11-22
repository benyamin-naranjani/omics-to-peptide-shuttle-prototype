"""12.4r — read-only, schedule-driven selection of next OmiPep review microbatch.

Run inside Module 01 notebook BEFORE existing 12.4f exporter. Does not edit files.
Validates the frozen 12.4e schedule and installed reviewed bundles; never
increments session names or guesses whether P08-02 exists.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, BadZipFile

# Works from the notebook current directory or a script in notebooks/.
_here = Path.cwd().resolve()
_candidates = [_here, *_here.parents]
if "__file__" in globals():
    _script = Path(__file__).resolve()
    _candidates = [_script.parent, *_script.parents, *_candidates]
ROOT = next((p for p in _candidates if (p / "data" / "processed").is_dir()
             and (p / "notebooks").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Cannot find OmiPep repository root (data/processed + notebooks).")
PROCESSED = ROOT / "data" / "processed"
SCHEDULE = PROCESSED / "EC_receptor_12_4e_literature_microbatch_plan_v1.csv"
PACKETS = PROCESSED / "review_packets"
if not SCHEDULE.is_file():
    raise FileNotFoundError(f"12.4e schedule missing: {SCHEDULE}; run preflight first")

with SCHEDULE.open(newline="", encoding="utf-8-sig") as fp:
    reader = csv.DictReader(fp)
    assert reader.fieldnames is not None, "Empty 12.4e schedule"
    assert len(reader.fieldnames) == len(set(reader.fieldnames)), "Nonunique 12.4e columns"
    required = {"relationship_id", "review_microbatch_id"}
    assert required.issubset(reader.fieldnames), f"Missing schedule fields {required - set(reader.fieldnames)}"
    schedule_rows = list(reader)
assert len(schedule_rows) == 125, f"Expected 125 frozen relationships; got {len(schedule_rows)}"
ids = [r["relationship_id"].strip() for r in schedule_rows]
assert all(ids) and len(ids) == len(set(ids)), "Missing/duplicate relationship IDs"

sessions = defaultdict(list)
pattern = re.compile(r"^LIT-([PR])(\d{2})-(\d{2})$")
for r in schedule_rows:
    session = r["review_microbatch_id"].strip()
    assert pattern.fullmatch(session), f"Unexpected review microbatch ID: {session!r}"
    sessions[session].append(r["relationship_id"].strip())
for name, rows in sessions.items():
    assert 1 <= len(rows) <= 5, f"{name}: schedule contains {len(rows)} rows, expected 1–5"
assert sum(map(len, sessions.values())) == 125, "Schedule not a 125-row partition"

def order_key(name):
    lane, major, minor = pattern.fullmatch(name).groups()
    return (0 if lane == "P" else 1, int(major), int(minor))

# A completed session requires an actual reviewed ZIP, not an exported source ZIP.
# Importer 12.4i is authoritative; run it after each installed reviewed packet.
completed = set()
if PACKETS.is_dir():
    for path in sorted(PACKETS.glob("LIT-*_REVIEWED_v*.zip")):
        try:
            with ZipFile(path, "r") as z:
                names = set(z.namelist())
                expected = {"references.csv", "evidence_claims.csv", "relationship_review.csv", "review_manifest.json"}
                assert expected.issubset(names), f"Missing entries {expected - names}"
                manifest = json.loads(z.read("review_manifest.json"))
                sid = manifest["review_batch_id"]
                assert sid in sessions, f"Reviewed packet {path.name} not in schedule: {sid}"
                assert set(manifest["relationship_ids"]) == set(sessions[sid]), (
                    f"{path.name}: reviewed relationship IDs disagree with frozen schedule")
                assert manifest.get("n_relationships") == len(sessions[sid]), "Manifest count mismatch"
                assert all(hashlib.sha256(z.read(name)).hexdigest() == digest
                           for name, digest in manifest.get("sha256_of_members", {}).items()), (
                    f"{path.name}: member checksum mismatch")
                for member in ["references.csv", "evidence_claims.csv", "relationship_review.csv"]:
                    assert z.read(member), f"Empty {member} in {path.name}"
                assert sid in path.name, f"Wrong filename/session for {path.name}"
                completed.add(sid)
        except (BadZipFile, ValueError, KeyError, AssertionError) as exc:
            raise RuntimeError(f"Refusing to skip invalid review package {path}: {exc}") from exc

pending = [s for s in sorted(sessions, key=order_key) if s not in completed]
OMIPEP_REVIEW_SESSION_ID = pending[0] if pending else None
print("=== OmiPep M01 | 12.4r: schedule-driven next review session ===")
print(f"PASS | Frozen schedule: {len(schedule_rows)} relationships in {len(sessions)} sessions; max five per session")
print(f"PASS | Reviewed ZIP sessions validated: {len(completed)}")
print(f"PASS | Pending sessions: {len(pending)}")
if OMIPEP_REVIEW_SESSION_ID:
    print(f"NEXT | {OMIPEP_REVIEW_SESSION_ID} ({len(sessions[OMIPEP_REVIEW_SESSION_ID])} relationships)")
else:
    print("COMPLETE | All 125 relationships have reviewed packets; no next export")
