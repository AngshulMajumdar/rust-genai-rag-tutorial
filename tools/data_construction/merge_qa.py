"""Apply hand-written patches; rebuild source[] from corpus3 ledger sections. Mechanical only.
usage: python3 -I merge_qa.py <corpus3> <qa_in> <patch_dir> <qa_out>
"""
import json, glob, sys
from collections import Counter
c3, qa_in, pdir, qa_out = sys.argv[1:5]
facts = {}
for fp in glob.glob(f'{c3}/*/*.facts.json'):
    lang = fp.split('/')[-2]; doc = fp.split('/')[-1][:-len('.facts.json')]
    for f in json.load(open(fp, encoding='utf-8')):
        facts[(lang, doc, f['subject'], f['aspect'])] = f
patches = {}
for p in sorted(glob.glob(f'{pdir}/*_part*.json')):
    for k, v in json.load(open(p, encoding='utf-8')).items():
        assert k not in patches, f'dup patch {k}'
        assert set(v) <= {'short_answer', 'question', 'answer'}, (k, v.keys())
        patches[k] = v
qa = json.load(open(qa_in, encoding='utf-8'))
stats = Counter(); out = []
for e in qa:
    pt = patches[e['id']]
    q = pt.get('question', e['question']); a = pt.get('answer', e['answer'])
    stats['q_changed'] += 'question' in pt; stats['a_changed'] += 'answer' in pt
    langs = [s['language'] for s in e['source']]; docs = [s['document'].replace('.md', '') for s in e['source']]
    src, sfs = [], []
    for sf in e['supporting_facts']:
        hit = None
        for L, D in zip(langs, docs):
            if (L, D, sf['subject'], sf['aspect']) in facts:
                hit = (L, D); break
        assert hit, (e['id'], sf)
        f = facts[(hit[0], hit[1], sf['subject'], sf['aspect'])]
        sfs.append({'language': hit[0], 'document': hit[1], 'subject': sf['subject'], 'aspect': sf['aspect']})
        s = {'language': hit[0], 'document': hit[1], 'section': f['section']}
        if s not in src:
            src.append(s)
    hop = e['hop_type']
    if hop == 'cross_section' and len({x['section'] for x in src}) < 2:
        stats['cross_section_same_section'] += 1
    if hop == 'cross_document' and len({x['document'] for x in src}) < 2:
        stats['cross_document_single_doc'] += 1
    out.append({'id': e['id'], 'language': e['language'], 'hop_type': hop, 'question': q, 'short_answer': pt['short_answer'],
                'answer': a, 'source': src, 'supporting_facts': sfs})
json.dump(out, open(qa_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(len(out), dict(stats))
