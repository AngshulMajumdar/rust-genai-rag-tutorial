"""Apply agent-chosen evidence ids to corpus3 ledgers (verbatim unit text only).
usage: python3 -I apply_picks.py <corpus3> <evidence_work> <qa_json>
"""
import json, glob, os, re, sys
c3, ew, qa_path = sys.argv[1:4]
qa = json.load(open(qa_path, encoding='utf-8'))
used = set()
for e in qa:
    for s in e['source']:
        for sf in e['supporting_facts']:
            used.add((s['language'], s['document'].replace('.md', ''), sf['subject'], sf['aspect']))
log = {'applied': 0, 'mismatch_fixed': [], 'dropped_absent': [], 'absent_but_used': []}
for pk in sorted(glob.glob(f'{ew}/*.picks.json')):
    name = os.path.basename(pk)[:-len('.picks.json')]
    lang, doc = name.split('__')
    units = {}
    for ln in open(f'{ew}/{name}.units.txt', encoding='utf-8'):
        m = re.match(r'\[(\d+)\] \(sec (.*?), p(\d+)\) (.*)', ln.rstrip('\n'))
        units[int(m.group(1))] = (None if m.group(2) == 'None' else m.group(2), int(m.group(3)), m.group(4))
    picks = json.load(open(pk, encoding='utf-8'))
    lp = f'{c3}/{lang}/{doc}.facts.json'
    facts = json.load(open(lp, encoding='utf-8'))
    out = []
    for f in facts:
        k = f"{f['subject']}|{f['aspect']}"
        p = picks.get(k)
        if p is None or f.get('evidence'):
            out.append(f); continue
        if p['status'] == 'absent':
            if (lang, doc, f['subject'], f['aspect']) in used:
                log['absent_but_used'].append([lang, doc, k]); out.append(f)
            else:
                log['dropped_absent'].append([lang, doc, k])
            continue
        ids = sorted(p['ids'])
        f['section'] = units[ids[0]][0]
        f['page'] = units[ids[0]][1]
        f['evidence'] = ' '.join(units[i][2] for i in ids)
        if p['status'] == 'mismatch':
            log['mismatch_fixed'].append([lang, doc, k, f['value'], p.get('doc_value')])
            f['ledger_value_superseded'] = f['value']
            f['value'] = p.get('doc_value')
        log['applied'] += 1
        out.append(f)
    json.dump(out, open(lp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(log, open(f'{ew}/_apply_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('applied', log['applied'], 'mismatch', len(log['mismatch_fixed']), 'dropped', log['dropped_absent'], 'absent_but_used', log['absent_but_used'])
for m in log['mismatch_fixed']:
    print('MISMATCH', m)
