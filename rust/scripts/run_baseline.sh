#!/usr/bin/env bash
# Phase-1 baseline grid on the 100-question stratified subset. Run from the rust/ folder after `index` and `embed`.
set -e
B=target/release/hr-rag
EVERY=${EVERY:-10}
mkdir -p runs
run () { name=$1; shift; $B eval --every "$EVERY" --out "runs/$name.jsonl" "$@"; $B score "runs/$name.jsonl" --report "runs/$name.md" >/dev/null; }
run gemma3-4b_simple_bm25      --model gemma3:4b --strategy simple  --mode bm25
run gemma3-4b_simple_hybrid    --model gemma3:4b --strategy simple  --mode hybrid
run gemma3-4b_agentic_hybrid   --model gemma3:4b --strategy agentic --mode hybrid
run gemma3-1b_agentic_hybrid   --model gemma3:1b --strategy agentic --mode hybrid
for f in runs/*.md; do echo "## $f"; grep -E "^\| (ALL|Cross-language|group|---)" "$f"; done
