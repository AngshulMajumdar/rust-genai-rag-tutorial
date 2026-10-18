"""Core: normalization, tokenization, chunking, BM25."""
import json, glob, os, re, sys, math, pickle
from collections import Counter, defaultdict

DIGIT_BLOCKS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DMAP = {chr(b + i): str(i) for b in DIGIT_BLOCKS for i in range(10)}
MAX_WORDS = 220


def norm(s):
    s = ''.join(DMAP.get(c, c) for c in s)
    s = re.sub(r'\*\*|__|`', '', s)
    return s.lower()


WORD_RE = re.compile(r'[\w\u0900-\u0963\u0966-\u0D7F\u200c\u200d]+')


def word_tokens(s):
    return WORD_RE.findall(norm(s))


def char_grams(s, n=4):
    out = []
    for w in word_tokens(s):
        w = f'#{w}#'
        if len(w) <= n:
            out.append(w)
        else:
            out.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return out


HEAD = re.compile(r'^(#{1,6})\s+(.*)$')


def chunk_doc(lang, doc, path):
    lines = open(path, encoding='utf-8').read().split('\n')
    title, heading_path = None, []
    blocks = []  # (heading_path tuple, paragraph)
    para = []
    def flush():
        if para:
            blocks.append((tuple(heading_path), ' '.join(para)))
            para.clear()
    for ln in lines:
        s = ln.strip()
        if s.startswith('<!--'):
            continue
        m = HEAD.match(s)
        if m:
            flush()
            lvl = len(m.group(1)); txt = re.sub(r'\*\*', '', m.group(2)).strip()
            if lvl <= 2 and title is None and lvl == 2:
                title = txt; heading_path = []; continue
            if lvl == 1:
                continue
            heading_path = heading_path[:max(0, lvl - 3)] + [txt]
            continue
        if not s or s == '---':
            flush(); continue
        para.append(s)
    flush()
    chunks, cur, cur_h, cur_n = [], [], None, 0
    def emit():
        if cur:
            chunks.append({'heading': ' > '.join(cur_h) if cur_h else '', 'text': '\n'.join(cur)})
    for h, p in blocks:
        n = len(p.split())
        if h != cur_h or cur_n + n > MAX_WORDS:
            emit(); cur, cur_n = [], 0; cur_h = h
        cur.append(p); cur_n += n
    emit()
    out = []
    for i, c in enumerate(chunks):
        out.append({'id': f'{lang[:2].upper()}:{doc}:{i:04d}', 'language': lang, 'document': doc,
                    'doc_title': title or doc, 'heading': c['heading'], 'text': c['text']})
    return out


class BM25:
    def __init__(self, docs_tokens, k1=1.4, b=0.75):
        self.N = len(docs_tokens); self.k1, self.b = k1, b
        self.len = [len(t) for t in docs_tokens]; self.avg = sum(self.len) / max(1, self.N)
        self.post = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for t, f in Counter(toks).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.N - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}

    def score(self, q_tokens, allowed=None):
        sc = defaultdict(float)
        for t, qf in Counter(q_tokens).items():
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, f in self.post[t]:
                if allowed is not None and i not in allowed:
                    continue
                sc[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sc


