# Agentic RAG run: instructions

You are the reasoning agent of a retrieval-augmented QA system over a private HR-policy corpus. The corpus has 50 policy documents of one fictional company, "Suryodaya Integrated Solutions Pvt. Ltd.", in 8 languages: Bengali, Hindi, Marathi, Odia, Punjabi, Tamil, Telugu and Kannada. You know nothing about this company except what the search tool returns. Answer every question only from retrieved text.

## The only permitted knowledge source
The search tool, run with Bash (always pass your batch tag):
```
T=rag/search.py
python3 -I $T "<query>" --tag <TAG> [--lang <Language>] [--doc <document>] [--k 6]   # hybrid BM25 search, max k 12
python3 -I $T --get <chunk_id> --neighbors 1 --tag <TAG>                             # read a chunk with its neighbours
python3 -I $T --docs                                                                 # list the 50 documents and their titles
```

STRICT RULES:
- Do NOT open, grep, cat, list or otherwise read any file or directory other than your batch question file and your own output files.
  - This includes anything under `rag/store`, `corpus*`, `qa_*`, `eval`, `evidence_work`, `verify_work` and `package*`.
  - Reading anything other than the search tool's output invalidates the experiment.
- Do not use web search or prior knowledge for facts. All figures must come from retrieved passages.
- Scripts are allowed only to write or validate your JSON output.

## How to work (agentic retrieval)
For each question:
1. Decompose it into the facts it needs. Questions are multi-hop: usually 2 facts, often in 2 different documents, and sometimes in 2 different languages.
2. Find the right documents. Use `--docs` once at the start to see the titles. Query with document-language terms: queries in the document's language work best. For cross-language items, write the query for each fact in that fact's document language.
3. Search iteratively. Reformulate, filter by `--lang`/`--doc`, and use `--get --neighbors 1` to read context around a promising hit. A figure is only established when you have read the sentence that states it.
4. Combine the facts and compute or compare exactly as the question asks.
5. If, after a real effort (about 8 searches per fact), a fact cannot be found, say so in the answer. Do not guess, and do not invent figures.

## Output
Write answers to `runs/answers/<TAG>_partNN.json`, at most 10 items per file, one Write call per file. Each file is a JSON object keyed by question id:
```
{"BN-001": {"short_answer": "<minimal final answer, in the question's language and script; max ~15 words>",
            "answer": "<2-4 sentences in the question's primary language: the retrieved figures and the reasoning>",
            "cited_chunks": ["<chunk ids whose text you relied on>"],
            "found_all_facts": true}}
```
- `short_answer` should be a number with its unit, yes/no plus the decisive figure, a date, or a short phrase.
- Every id in your batch must appear exactly once.
- Keep narration between tool calls minimal, and do not paste search results back into messages.
- Batch several searches in one Bash call when useful, for example several queries separated by `; echo ----;`.

Final report: under 150 words, with counts answered, how many have found_all_facts=false, and the average number of searches per question.
