"""Mechanical ledger normalizer. Never writes prose: evidence is quoted verbatim from the .md.
usage: python3 -I normalize_ledgers.py <corpus_in> <corpus_out>
"""
import json, glob, os, re, sys, shutil
from collections import Counter

LANGS = ['Bengali', 'Hindi', 'Marathi', 'Odia', 'Punjabi', 'Tamil', 'Telugu', 'Kannada']
DIGIT_BLOCKS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DMAP = {chr(b + i): str(i) for b in DIGIT_BLOCKS for i in range(10)}
DIG = '0-9' + ''.join(chr(b) + '-' + chr(b + 9) for b in DIGIT_BLOCKS)


def ascii_digits(s):
    return ''.join(DMAP.get(c, c) for c in str(s))


def nums(s):
    s = ascii_digits(s).replace(',', '')
    return [n for n in re.findall(r'\d+(?:\.\d+)?', s)]


def clean(s):
    s = re.sub(r'\*\*|__|`', '', s)
    return re.sub(r'\s+', ' ', s).strip()


HEAD_RE = re.compile(r'^(#{2,6})\s+(.*)$')
SEC_RE = re.compile(r'^([' + DIG + r']+(?:\.[' + DIG + r']+)*[a-zA-Zକக]?)\.?\s')
MARK_RE = re.compile(r'^<!--\s*\S+\s+(\d+)\s*-->')


def parse_doc(path):
    """Return list of units: dict(text, section, page). Units = sentences (or table rows)."""
    lines = open(path, encoding='utf-8').read().split('\n')
    units, section, page, seen_title = [], None, 1, False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if MARK_RE.match(s):
            page += 1
            continue
        h = HEAD_RE.match(s)
        if h:
            txt = clean(h.group(2))
            m = SEC_RE.match(txt + ' ')
            if m:
                section = m.group(1)
            elif seen_title:
                section = txt  # unnumbered heading, e.g. appendix name
            seen_title = True
            continue
        if s.startswith('#') or s == '---':
            continue
        txt = clean(s)
        if txt.startswith('|'):
            parts = [txt]
        else:
            parts = re.split(r'(?<=[।॥\.\?!])\s+(?=\S)', txt)
        for p in parts:
            p = p.strip()
            if len(p) > 3:
                units.append({'text': p, 'section': section, 'page': page})
    return units


def norm_key(s):
    return re.sub(r'\s+', '', ascii_digits(clean(s)))


def section_matches(unit_sec, want):
    if not want or not unit_sec:
        return False
    a, b = ascii_digits(unit_sec), ascii_digits(str(want)).rstrip('.')
    return a == b or a.startswith(b + '.')


def locate(fact, units):
    """Return (unit, method) or (None, reason)."""
    sent = fact.get('sentence')
    if sent:
        k = norm_key(sent)
        for u in units:
            nk = norm_key(u['text'])
            if k and (k == nk or k in nk or (len(nk) > 20 and nk in k)):
                return u, 'sentence_exact'
        # fallback: best token overlap
        toks = set(clean(sent).split())
        best, bs = None, 0
        for u in units:
            o = len(toks & set(u['text'].split())) / max(1, len(toks))
            if o > bs:
                best, bs = u, o
        if best and bs >= 0.6:
            return best, 'sentence_fuzzy'
    want = fact.get('section')
    if want is not None:
        want = re.sub(r'\.0$', '', str(want))
    val = clean(str(fact.get('value', '')))
    if len(val) >= 4:
        vk = norm_key(val)
        hits = [u for u in units if vk in norm_key(u['text'])]
        insec = [u for u in hits if section_matches(u['section'], want)]
        if insec:
            return insec[0], 'value_text_in_section'
        if len(hits) == 1:
            return hits[0], 'value_text_unique_in_doc'
    vn = [n for n in nums(fact.get('value', '')) if n not in ('0',)]
    if not vn:
        return None, 'no_numeric_value'
    def cands(scope):
        out = []
        for u in scope:
            un = set(nums(u['text']))
            if all(n in un for n in vn):
                out.append(u)
        return out
    if want:
        c = cands([u for u in units if section_matches(u['section'], want)])
        if c:
            if len(c) == 1:
                return c[0], 'value_in_section'
            return None, 'value_in_section_multi'
    c = cands(units)
    if len(c) == 1:
        return c[0], 'value_unique_in_doc'
    if c:
        return None, 'value_ambiguous_in_doc'
    return None, 'value_not_found'


def main(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    report = Counter()
    unresolved = []
    for lang in LANGS:
        os.makedirs(f'{dst}/{lang}', exist_ok=True)
        for md in sorted(glob.glob(f'{src}/{lang}/*.md')):
            doc = os.path.basename(md)[:-3]
            shutil.copy(md, f'{dst}/{lang}/{doc}.md')
            units = parse_doc(md)
            facts = json.load(open(f'{src}/{lang}/{doc}.facts.json', encoding='utf-8'))
            out = []
            for f in facts:
                u, how = locate(f, units)
                report[(lang, how)] += 1
                rec = {
                    'subject': f['subject'],
                    'aspect': f['aspect'],
                    'value': f.get('value'),
                }
                if f.get('unit'):
                    rec['unit'] = f['unit']
                if u:
                    rec['section'] = u['section']
                    rec['page'] = u['page']
                    rec['evidence'] = u['text']
                else:
                    rec['section'] = f.get('section')
                    rec['page'] = None
                    rec['evidence'] = None
                    unresolved.append({'language': lang, 'document': doc, 'reason': how, **{k: f.get(k) for k in ('subject', 'aspect', 'value', 'section', 'sentence')}})
                out.append(rec)
            json.dump(out, open(f'{dst}/{lang}/{doc}.facts.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(unresolved, open(f'{dst}/_unresolved.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k in sorted(report):
        print(k, report[k])
    print('UNRESOLVED', len(unresolved))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
