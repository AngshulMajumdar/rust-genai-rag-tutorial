"""Build per-group QA work files: each entry + its supporting facts (value, evidence) from corpus3 + numeric leak flags.
usage: python3 -I build_qa_work.py <corpus3> <qa_json> <outdir>
"""
import json, glob, os, re, sys
from collections import defaultdict
c3, qa_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
DIGIT_BLOCKS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DMAP = {chr(b + i): str(i) for b in DIGIT_BLOCKS for i in range(10)}
def nums(s):
    s = ''.join(DMAP.get(c, c) for c in str(s)).replace(',', '')
    return set(n for n in re.findall(r'\d+(?:\.\d+)?', s))
facts = {}
for fp in glob.glob(f'{c3}/*/*.facts.json'):
    lang = fp.split('/')[-2]; doc = os.path.basename(fp)[:-len('.facts.json')]
    for f in json.load(open(fp, encoding='utf-8')):
        facts[(lang, doc, f['subject'], f['aspect'])] = f
qa = json.load(open(qa_path, encoding='utf-8'))
groups = defaultdict(list)
for e in qa:
    g = e['id'].split('-')[0] if not e['id'].startswith('XL') else 'XL'
    qn = nums(e['question'])
    sf_out = []
    for sf in e['supporting_facts']:
        hit = None
        for s in e['source']:
            k = (s['language'], s['document'].replace('.md', ''), sf['subject'], sf['aspect'])
            if k in facts:
                hit = (k, facts[k]); break
        if hit is None:
            raise SystemExit(f'missing fact {e["id"]} {sf}')
        (lang, doc, _, _), f = hit
        vn = nums(f['value']) | nums(f['evidence'] or '')
        vn_val = nums(f['value'])
        leaked = sorted(n for n in vn_val if n in qn and not (len(n) == 4 and n.startswith('20')))
        sf_out.append({'language': lang, 'document': doc, 'section': f['section'], 'subject': f['subject'], 'aspect': f['aspect'],
                       'value': f['value'], 'evidence': f['evidence'], 'value_numbers_found_in_question': leaked})
    groups[g].append({'id': e['id'], 'language': e['language'], 'hop_type': e['hop_type'], 'question': e['question'], 'answer': e['answer'],
                      'supporting_facts': sf_out,
                      'numeric_leak_flag': any(x['value_numbers_found_in_question'] for x in sf_out)})
for g, items in groups.items():
    json.dump(items, open(f'{out}/{g}.work.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(g, len(items), 'flagged', sum(i['numeric_leak_flag'] for i in items))
