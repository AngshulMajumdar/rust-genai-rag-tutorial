"""Map gold evidence to chunk ids, then run single-shot (non-agentic) retrieval baselines on all questions.
usage: python3 eval/retrieval_baseline.py      (run rag/build_index.py first)
Writes results/gold_chunks.json and results/baseline_retrieval.json.
"""
import sys, os, json, glob, re, pickle
from collections import defaultdict, Counter
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'rag'))
from ragcore import norm  # noqa
import search as S  # noqa

idx, chunks = S.load()
def clean(s):
    return re.sub(r'\s+', '', norm(s))
ctext = [clean(c['text']) for c in chunks]
by_doc = defaultdict(list)
for i, c in enumerate(chunks):
    by_doc[(c['language'], c['document'])].append(i)
facts = {}
for fp in glob.glob(os.path.join(REPO, 'data', 'corpus', '*', '*.facts.json')):
    L = fp.split('/')[-2]; D = os.path.basename(fp)[:-11]
    for f in json.load(open(fp, encoding='utf-8')):
        facts[(L, D, f['subject'], f['aspect'])] = f
qa = json.load(open(os.path.join(REPO, 'data', 'testset', 'suryodaya_qa_testset_1000.json'), encoding='utf-8'))
gold = {}
miss = 0
for e in qa:
    g = []
    for s in e['supporting_facts']:
        f = facts[(s['language'], s['document'], s['subject'], s['aspect'])]
        parts = [clean(p) for p in re.split(r'(?<=[।॥\.\?!])\s+', f['evidence']) if len(p.strip()) > 8]
        ids = sorted({chunks[i]['id'] for i in by_doc[(s['language'], s['document'])] for p in parts if p and p in ctext[i]})
        if not ids:
            # evidence split across chunk boundary or table: take chunks containing the longest evidence fragment pieces
            ev = clean(f['evidence'])
            cand = by_doc[(s['language'], s['document'])]
            best = max(cand, key=lambda i: max((len(ev[j:j+40]) for j in range(0, max(1, len(ev)-40), 20) if ev[j:j+40] in ctext[i]), default=0))
            if any(ev[j:j+40] in ctext[best] for j in range(0, max(1, len(ev)-40), 20)):
                ids = [chunks[best]['id']]
            else:
                miss += 1
        g.append(ids)
    gold[e['id']] = g
json.dump(gold, open(os.path.join(REPO, 'results', 'gold_chunks.json'), 'w'), indent=0)
print('facts without gold chunk:', miss)

pos = {c['id']: i for i, c in enumerate(chunks)}
res = defaultdict(lambda: defaultdict(Counter))
KS = [1, 3, 5, 10, 20]
for e in qa:
    langs = sorted({s['language'] for s in e['source']})
    for mode in ('global', 'lang_filtered'):
        if mode == 'global':
            top = S.search(idx, chunks, e['question'], k=20)
        else:
            top = []
            for L in langs:
                top += S.search(idx, chunks, e['question'], lang=L, k=20)
            # interleave per-language lists
            per = [S.search(idx, chunks, e['question'], lang=L, k=20) for L in langs]
            top = [x for tup in zip(*per) for x in tup] if len(per) > 1 else per[0]
        top_ids = [chunks[i]['id'] for i in top]
        for k in KS:
            hit = [bool(set(g) & set(top_ids[:k])) for g in gold[e['id']]]
            for key in ('ALL', e['hop_type'], ('XL:' + e['language']) if e['hop_type'] == 'cross_language' else e['language']):
                res[mode][key][f'fact@{k}'] += sum(hit)
                res[mode][key][f'all@{k}'] += all(hit)
        for key in ('ALL', e['hop_type'], ('XL:' + e['language']) if e['hop_type'] == 'cross_language' else e['language']):
            res[mode][key]['n_q'] += 1
            res[mode][key]['n_f'] += len(gold[e['id']])
out = {}
for mode, d in res.items():
    out[mode] = {}
    for key, c in d.items():
        out[mode][key] = {'n': c['n_q'], **{f'fact_recall@{k}': round(c[f'fact@{k}'] / c['n_f'], 3) for k in KS},
                          **{f'all_facts@{k}': round(c[f'all@{k}'] / c['n_q'], 3) for k in KS}}
json.dump(out, open(os.path.join(REPO, 'results', 'baseline_retrieval.json'), 'w'), indent=1)
for mode in out:
    print('==', mode)
    for key in sorted(out[mode], key=lambda x: (x != 'ALL', x)):
        r = out[mode][key]
        print(f"{key:15s} n={r['n']:4d}  fact@5={r['fact_recall@5']:.3f} fact@10={r['fact_recall@10']:.3f}  all@5={r['all_facts@5']:.3f} all@10={r['all_facts@10']:.3f} all@20={r['all_facts@20']:.3f}")
