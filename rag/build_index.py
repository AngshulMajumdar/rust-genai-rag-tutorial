"""Build the RAG store: section-aware chunks + hybrid BM25 index (word unigrams and char 4-grams).
usage: python3 rag/build_index.py [corpus_dir] [store_dir]   (defaults: data/corpus, rag/store)
The store contains ONLY corpus text. No questions, answers or fact ledgers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ragcore import *
import json, glob, re, pickle

def main(corpus, store):
    chunks = []
    for md in sorted(glob.glob(f'{corpus}/*/*.md')):
        lang = md.split('/')[-2]; doc = os.path.basename(md)[:-3]
        if not re.match(r'\d\d_', doc):
            continue
        chunks.extend(chunk_doc(lang, doc, md))
    texts = [c['doc_title'] + ' ' + c['heading'] + ' ' + c['text'] for c in chunks]
    word = BM25([word_tokens(t) for t in texts])
    char = BM25([char_grams(t) for t in texts])
    os.makedirs(store, exist_ok=True)
    with open(f'{store}/chunks.jsonl', 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    pickle.dump({'word': word, 'char': char, 'ids': [c['id'] for c in chunks],
                 'lang': [c['language'] for c in chunks]}, open(f'{store}/index.pkl', 'wb'))
    docs = sorted({(c['language'], c['document'], c['doc_title']) for c in chunks})
    with open(f'{store}/documents.tsv', 'w', encoding='utf-8') as f:
        for d in docs:
            f.write('\t'.join(d) + '\n')
    wc = [len(c['text'].split()) for c in chunks]
    print('chunks', len(chunks), 'docs', len(docs), 'avg words', sum(wc) / len(wc), 'max', max(wc))


if __name__ == '__main__':
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, 'data', 'corpus'),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo, 'rag', 'store'))
