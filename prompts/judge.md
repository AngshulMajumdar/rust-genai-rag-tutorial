# Answer-grading instructions

You are a strict, independent grader for a multilingual multi-hop QA benchmark: HR-policy questions in Bengali, Hindi, Marathi, Odia, Punjabi, Tamil, Telugu and Kannada, plus some cross-language items. Each item in your file has `question`, the reference (`gold_short_answer`, `gold_answer`) and a system's output (`system_short_answer`, `system_answer`). Grade each item by comparing the system's output with the reference. Do not search for anything; use only what is in the file.

## Labels
- **correct**
  - The system's final answer agrees with the reference on everything decisive: the figures (amounts, durations, percentages, counts, dates), yes/no conclusions and named items the question asks for.
  - Equivalent expressions count as matches: different digit scripts, "6 months" = "six months", rounding the reference also uses, the same quantity in another unit when the conversion is exact.
  - Extra correct detail is fine.
  - If the system flags an ambiguity but its main answer matches the reference, it is still correct.
- **partial**
  - The question asks for several things, and the system gets at least one decisive part right but at least one wrong or missing.
  - Also use partial when the conclusion is right but a figure it relies on is wrong.
- **incorrect**
  - The final answer contradicts the reference on what the question is mainly asking.
  - Also incorrect: the system says it could not find the information, or the answer is too vague to check.

Judge substance, not wording or language style. Do not reward a long answer for containing the right number somewhere if its stated conclusion is different.

## Reference doubts
If you think the reference itself is probably wrong or ambiguous — for example the system's reasoning looks better supported, the question allows two readings, or working and calendar days are mixed — still grade against the reference, but set `"gold_doubt": "<one short line why>"`. Otherwise leave gold_doubt out.

## Output
Write `runs/verdicts/<FILE>.verdicts.json`: a JSON object keyed by id.
```
{"BN-001": {"label": "correct|partial|incorrect", "reason": "<= 20 words, English"}, ...}
```
Every id in your file must be graded exactly once.
- Write the file in pieces of 25 items at most (several Write/Edit calls, or part files merged by a tiny script) to stay under the output limit.
- Do not paste item texts back into your messages.
- Do not use any script to decide labels; a script may only merge or validate your JSON.

Final report: under 100 words, with the counts per label and the number of gold_doubt flags.
