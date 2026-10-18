"""Sanity checks for the public GitHub repository.

Run from anywhere with: python tools/repo_check.py
No third-party packages are required.
"""
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DOCS = {
    "Bengali": 4,
    "Hindi": 9,
    "Kannada": 4,
    "Marathi": 6,
    "Odia": 5,
    "Punjabi": 5,
    "Tamil": 10,
    "Telugu": 7,
}
GENERATED = {"target", "store", "runs", "hr_rag_work", "__pycache__"}

errors = []

for p in ROOT.rglob("*"):
    rel = p.relative_to(ROOT)
    if any(part.startswith(".") for part in rel.parts):
        errors.append(f"dot path committed: {rel}")
    if any(part in GENERATED for part in rel.parts):
        errors.append(f"generated directory committed: {rel}")

for name in ["index.html", "medium.html", "tutorial_body.html", "styles.css", "PUBLISHING.md"]:
    if not (ROOT / "docs" / name).is_file():
        errors.append(f"missing docs/{name}")

counts = {}
facts = 0
for lang, expected in EXPECTED_DOCS.items():
    folder = ROOT / "data" / "corpus" / lang
    docs = sorted(folder.glob("[0-9][0-9]_*.md"))
    counts[lang] = len(docs)
    if len(docs) != expected:
        errors.append(f"{lang}: expected {expected} documents, found {len(docs)}")
    for doc in docs:
        ledger = doc.with_suffix(".facts.json")
        if not ledger.is_file():
            errors.append(f"missing ledger for {doc.relative_to(ROOT)}")
            continue
        try:
            rows = json.loads(ledger.read_text(encoding="utf-8"))
            facts += len(rows)
        except Exception as exc:
            errors.append(f"cannot parse {ledger.relative_to(ROOT)}: {exc}")

try:
    test = json.loads((ROOT / "data" / "testset" / "suryodaya_qa_testset_1000.json").read_text(encoding="utf-8"))
    ids = [x["id"] for x in test]
    if len(test) != 1000:
        errors.append(f"expected 1000 questions, found {len(test)}")
    if len(ids) != len(set(ids)):
        errors.append("test-set IDs are not unique")
    hops = Counter(x["hop_type"] for x in test)
except Exception as exc:
    errors.append(f"cannot parse test set: {exc}")
    test = []
    hops = Counter()

if facts != 6315:
    errors.append(f"expected 6315 fact records, found {facts}")

print("documents:", sum(counts.values()), counts)
print("facts:", facts)
print("questions:", len(test), dict(hops))

if errors:
    print("\nFAILED")
    for e in errors:
        print("-", e)
    sys.exit(1)

print("\nOK: repository structure, Pages files and corpus cardinalities are consistent")
