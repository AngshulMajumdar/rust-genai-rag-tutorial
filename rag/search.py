"""Corpus search tool for the RAG agents.
  python3 rag/search.py "<query>" [--lang Bengali|Hindi|Marathi|Odia|Punjabi|Tamil|Telugu|Kannada] [--doc <document>] [--k 6]
  python3 rag/search.py --get <chunk_id> [--neighbors 1]
  python3 rag/search.py --docs            (list documents with titles)
Hybrid BM25 (word unigram + character 4-gram), fused by reciprocal rank.
"""
import sys, os, json, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ragcore import *  # noqa
STORE = os.environ.get('RAG_STORE', os.path.join(HERE, 'store'))

def load():
    idx = pickle.load(open(os.path.join(STORE, 'index.pkl'), 'rb'))
    chunks = [json.loads(l) for l in open(os.path.join(STORE, 'chunks.jsonl'), encoding='utf-8')]
    return idx, chunks

def search(idx, chunks, q, lang=None, doc=None, k=6):
    allowed = None
    if lang or doc:
        allowed = {i for i, c in enumerate(chunks) if (not lang or c['language'] == lang) and (not doc or c['document'] == doc)}
    fused = {}
    for name, toks in (('word', word_tokens(q)), ('char', char_grams(q))):
        sc = idx[name].score(toks, allowed)
        for r, (i, _) in enumerate(sorted(sc.items(), key=lambda x: -x[1])[:200]):
            fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]

def show(c):
    print(f"=== [{c['id']}] {c['language']} | {c['document']} | {c['doc_title']} | {c['heading']}")
    print(c['text']); print()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='?'); ap.add_argument('--lang'); ap.add_argument('--doc')
    ap.add_argument('--k', type=int, default=6); ap.add_argument('--get'); ap.add_argument('--neighbors', type=int, default=0)
    ap.add_argument('--docs', action='store_true'); ap.add_argument('--tag', default='untagged')
    a = ap.parse_args()
    if a.docs:
        print(open(os.path.join(STORE, 'documents.tsv'), encoding='utf-8').read()); sys.exit()
    import time
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    with open(os.path.join(HERE, 'logs', f'{a.tag}.jsonl'), 'a', encoding='utf-8') as lf:
        lf.write(json.dumps({'t': time.time(), 'query': a.query, 'get': a.get, 'lang': a.lang, 'doc': a.doc, 'k': a.k}, ensure_ascii=False) + '\n')
    idx, chunks = load()
    if a.get:
        pos = {c['id']: i for i, c in enumerate(chunks)}[a.get]
        for j in range(pos - a.neighbors, pos + a.neighbors + 1):
            if 0 <= j < len(chunks) and chunks[j]['document'] == chunks[pos]['document']:
                show(chunks[j])
        sys.exit()
    for i in search(idx, chunks, a.query, a.lang, a.doc, min(a.k, 12)):
        show(chunks[i])
