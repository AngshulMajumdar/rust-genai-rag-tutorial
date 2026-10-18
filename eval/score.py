"""Aggregate agentic-RAG results: judge accuracy, automatic numeric match, cited-evidence recall, search effort.
usage: python3 eval/score.py    (after eval/retrieval_baseline.py has written results/gold_chunks.json)
"""
import sys, json, glob, re, collections
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(REPO, *p)
qa = {e['id']: e for e in json.load(open(J('data', 'testset', 'suryodaya_qa_testset_1000.json'), encoding='utf-8'))}
pred = json.load(open(J('results', 'predictions_agentic_rag.json'), encoding='utf-8'))
gold = json.load(open(J('results', 'gold_chunks.json')))
ver = json.load(open(J('results', 'judge_verdicts.json'), encoding='utf-8'))
assert set(ver) == set(qa), (len(ver), set(qa) - set(ver))

DB = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DM = {chr(b + i): str(i) for b in DB for i in range(10)}
def nums(s):
    s = ''.join(DM.get(c, c) for c in str(s)).replace(',', '')
    return {float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)}

def group(e):
    return 'Cross-language' if e['hop_type'] == 'cross_language' else e['language']

rows = collections.defaultdict(lambda: collections.Counter())
for i, e in qa.items():
    v = ver[i]['label']; p = pred[i]
    g_n = nums(e['short_answer']); p_n = nums(p.get('short_answer', '')) | nums(p.get('answer', ''))
    auto = (g_n <= p_n) if g_n else None
    gl = gold[i]; cited = set(p.get('cited_chunks', []))
    hits = [bool(set(g) & cited) for g in gl if g]
    for key in ('ALL', 'lang:' + group(e), 'hop:' + e['hop_type']):
        r = rows[key]
        r['n'] += 1; r[v] += 1
        if auto is not None:
            r['auto_n'] += 1; r['auto_ok'] += auto
        r['facts'] += len(hits); r['facts_cited'] += sum(hits); r['all_cited'] += all(hits)
        r['gold_doubt'] += 'gold_doubt' in ver[i]

searches = collections.Counter()
for p in glob.glob(J('results', 'search_logs', '*.jsonl')):
    searches[p.split('/')[-1][:-6]] = sum(1 for _ in open(p))

out = {}
for k, r in rows.items():
    out[k] = {'n': r['n'], 'accuracy_strict': round(r['correct'] / r['n'], 3),
              'accuracy_lenient': round((r['correct'] + .5 * r['partial']) / r['n'], 3),
              'correct': r['correct'], 'partial': r['partial'], 'incorrect': r['incorrect'],
              'auto_numeric_match': round(r['auto_ok'] / r['auto_n'], 3) if r['auto_n'] else None,
              'cited_fact_recall': round(r['facts_cited'] / r['facts'], 3),
              'cited_all_facts': round(r['all_cited'] / r['n'], 3), 'gold_doubt': r['gold_doubt']}
out['_search_calls_total'] = sum(searches.values())
out['_search_calls_per_question'] = round(sum(searches.values()) / len(qa), 2)
json.dump(out, open(J('results', 'results_agentic_rag.json'), 'w'), indent=1)
order = ['ALL'] + sorted(k for k in out if k.startswith('lang:')) + sorted(k for k in out if k.startswith('hop:'))
print(f"{'group':28s}{'n':>5s}{'strict':>8s}{'lenient':>9s}{'C/P/I':>12s}{'autoNum':>9s}{'citeFact':>9s}{'citeAll':>8s}{'doubt':>6s}")
for k in order:
    r = out[k]
    print(f"{k:28s}{r['n']:5d}{r['accuracy_strict']:8.3f}{r['accuracy_lenient']:9.3f}{str(r['correct'])+'/'+str(r['partial'])+'/'+str(r['incorrect']):>12s}{r['auto_numeric_match']:9.3f}{r['cited_fact_recall']:9.3f}{r['cited_all_facts']:8.3f}{r['gold_doubt']:6d}")
print('search calls total', out['_search_calls_total'], 'per question', out['_search_calls_per_question'])
