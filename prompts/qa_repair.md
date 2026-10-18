# QA repair instructions (read fully before starting)

You are repairing one slice of a 1000-question multilingual multi-hop QA benchmark built on an HR-policy corpus of a fictional company. Your slice is the work file named in your task: `qa_work/<G>.work.json`. Each item has: `id`, `language`, `hop_type`, `question`, `answer`, and `supporting_facts` — each fact with its `value` and its verbatim `evidence` sentence quoted from the source document (the evidence is ground truth; the value is a ledger paraphrase of it). `numeric_leak_flag` / `value_numbers_found_in_question` are a crude digit-match hint only — it misses values written as words and has false positives on small numbers. Judge every item yourself.

## Problem 1 — answer leakage (fix it)
The benchmark is for retrieval-augmented QA. A question must NOT hand the reader the values it is testing. Many current questions state the fact values in the question ("The A deadline is 15 days and the B deadline is 14 days; which is longer?"). That makes retrieval unnecessary.
Rewrite every question that states, in digits OR in words, the value of ANY of its supporting facts (amounts, durations, percentages, counts, thresholds, dates, named outcomes). Replace each stated value with a description of the rule/policy item, so the reader has to find it in the documents. Examples of the fix:
- leaky: "The probation period is 3 months and the misconduct action limit is 90 days. Is ...?"
- fixed: "Compare the associate-level probation period with the outer time limit for initiating misconduct proceedings. Is ...?"
Scenario numbers that are NOT fact values (a hypothetical employee's salary, a joining date, a claim amount the reader must test against a limit) are allowed and often good — they keep the computation real.
After rewriting, the question must still require BOTH (all) supporting facts, keep the same hop type, the same language, and be natural, specific and unambiguous. Do not make it vague: name the document's concept precisely enough that a reader with the corpus can identify the exact rule. If a question does not leak, leave it as it is.

## Problem 2 — short answers (add for every item)
Add `short_answer` to EVERY item: the minimal final answer, in the item's primary language and script, max ~15 words. Examples of form: a number with unit ("২৫ কার্যদিবস"), "yes/no + the decisive figure", a date, a ratio, or a short noun phrase. It must follow from the evidence. Use the same digit style the item's language uses in the corpus (e.g. Bengali/Hindi/Marathi/Odia/Punjabi docs use native digits; Tamil/Telugu/Kannada mostly ASCII digits) — match what the existing answer uses.

## Problem 3 — correctness
Check each answer against the `evidence` sentences. If an answer contradicts the evidence (wrong figure, wrong arithmetic, wrong conclusion), correct the answer. Keep answers in the same language, concise, stating the actual figures and reasoning. If you rewrite a question, adjust the answer only as needed to match it.

## Absolute rules
- Every question/answer/short_answer you produce must be written by you, by hand. Do NOT use any script, template, string-substitution or mail-merge to produce or alter question/answer/short_answer text. Scripts are allowed only to validate (JSON syntax, coverage of ids, digit-leak re-check).
- No English words in Indic text except acronyms that the source documents themselves use (HRBP, CTC, POSH, ESOP …). For the cross-language items (XL-*) keep the existing bilingual design: primary-language text, with the other language's term given in its own script where needed; never English.
- Do not change `supporting_facts`, ids, or hop types.

## Output
Write patch files to `qa_patches/` named `<G>_partNN.json` (NN = 01, 02, …), at most 15 items per file, one Write call per file. Each file is a JSON object keyed by id:
  {"BN-001": {"short_answer": "...", "question": "<only if changed>", "answer": "<only if changed>"}, ...}
Include `question`/`answer` keys ONLY when you changed them. Every id in your work file must appear in exactly one patch file.
Keep narration between tool calls to a line; never paste large blocks of your own output into messages (there is a per-message output cap — large single outputs have crashed earlier runs).

## Final check and report
Run a small validation script: all ids covered exactly once; JSON valid; for every item whose question you did not change, confirm it has no leak by your own reading. Then re-run a digit check: for each item, numbers of each supporting fact's `value` that still appear in the (final) question — every remaining hit must be a deliberate scenario number or a non-value number (e.g. a section number); list those ids with a one-line justification each.
Report (under 400 words): counts — questions rewritten, answers corrected (with ids and what was wrong), short answers added; and 3 before/after examples (ids only plus one-line description, do not paste full text).
