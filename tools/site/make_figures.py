"""Builds the PNG figures in docs/assets/ from the stored results (results/*.json, data/).
Static images so the same files work on GitHub Pages and in a Medium import. Run from the repo root."""
import json, glob, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated reference palette)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
OUT = "docs/assets/"
LANGS = ["Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]

def style(ax, xmax=None):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if xmax: ax.set_xlim(0, xmax)

def hbar(ax, labels, vals, colors, fmt):
    y = range(len(labels))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62, edgecolor=SURF, linewidth=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, fmt(v), va="center", color=INK, fontsize=10)

def title(fig, t, sub):
    fig.text(0.02, 0.97, t, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.02, 0.905, sub, fontsize=10.5, va="top", color=INK2)

# 1. dataset composition
test = json.load(open("data/testset/suryodaya_qa_testset_1000.json"))
cnt = collections.Counter("Cross-language" if q["hop_type"] == "cross_language" else q["language"] for q in test)
docs = collections.Counter(p.split("/")[2] for p in glob.glob("data/corpus/*/*.md"))
order = sorted(LANGS, key=lambda l: -cnt[l]) + ["Cross-language"]
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200); fig.subplots_adjust(left=0.25, right=0.95, top=0.8, bottom=0.08)
hbar(ax, [f"{l} ({docs[l]} docs)" if l in docs else "Cross-language" for l in order], [cnt[l] for l in order],
     [BLUE] * 8 + [ORANGE], lambda v: f"{v}")
style(ax, 205)
title(fig, "1,000 multi-hop questions over 50 documents", "Questions per language; the 100 cross-language questions need facts from two languages")
fig.savefig(OUT + "fig1_dataset.png"); plt.close(fig)

# 2. single-shot retrieval collapses on cross-language questions
b = json.load(open("results/baseline_retrieval.json"))
g, lf = b["global"], b["lang_filtered"]
labels = ["Cross-section (180)", "Cross-document (710)", "Cross-language (100)", "Cross-language,\nper-language queries"]
vals = [g["cross_section"]["all_facts@10"], g["cross_document"]["all_facts@10"], g["cross_language"]["all_facts@10"], lf["cross_language"]["all_facts@10"]]
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200); fig.subplots_adjust(left=0.27, right=0.95, top=0.76, bottom=0.1)
hbar(ax, labels, [v * 100 for v in vals], [BLUE, BLUE, BLUE, ORANGE], lambda v: f"{v:.0f}%")
style(ax, 100)
title(fig, "Cross-language questions defeat a single query",
      "Questions with every gold fact in the top 10 of 5,507 chunks (BM25, word + char 4-gram)")
fig.savefig(OUT + "fig2_retrieval.png"); plt.close(fig)

# 3. retrieval by question language
fig, ax = plt.subplots(figsize=(7, 4.0), dpi=200); fig.subplots_adjust(left=0.17, right=0.95, top=0.8, bottom=0.08)
ls = sorted(LANGS, key=lambda l: -g[l]["all_facts@10"])
hbar(ax, ls, [g[l]["all_facts@10"] * 100 for l in ls], [BLUE] * 8, lambda v: f"{v:.0f}%")
style(ax, 100)
title(fig, "Single-query retrieval by language", "Share of questions with every gold fact in the top 10, single-language questions only")
fig.savefig(OUT + "fig3_retrieval_by_language.png"); plt.close(fig)

# 4. agentic RAG pipeline
fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200); fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
def box(x, y, w, h, t, c=BLUE, fc="#eef4fc"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", ec=c, fc=fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, color=INK, linespacing=1.35)
W, G, Y, H = 16.4, 3.3, 17, 16
xs = [1.8 + i * (W + G) for i in range(5)]
texts = ["Question\nin any of\n8 languages", "Plan one query\nper needed fact,\nin the language\nof its document",
         "Hybrid search\nBM25 word +\nchar 4-gram\n(+ dense)", "Answer from\nthe passages\nonly; cite\nchunk ids", "Short answer,\nanswer and\ncitations"]
for i, (x, t) in enumerate(zip(xs, texts)):
    box(x, Y, W, H, t, *( (ORANGE, "#fdf0ea") if i == 4 else (BLUE, "#eef4fc") ))
    if i < 4:
        ax.annotate("", (xs[i + 1] - 0.6, Y + H / 2), (x + W + 0.6, Y + H / 2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
ax.annotate("", (xs[1] + W / 2, Y - 0.8), (xs[3] + W / 2, Y - 0.8), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, connectionstyle="arc3,rad=-0.3"))
ax.text(50, 3.2, "a needed fact is missing: re-plan for it and search again", ha="center", fontsize=8.5, color=ORANGE)
ax.text(2, 45.5, "Agentic RAG loop", fontsize=13, fontweight="bold", color=INK, va="top")
ax.text(2, 39.5, "Run with Claude agents (results below) and with local Ollama models (Rust chatbot)", fontsize=9, color=INK2, va="top")
fig.savefig(OUT + "fig4_pipeline.png"); plt.close(fig)

# 5. table image: agentic RAG results (tables do not survive a Medium import)
r = json.load(open("results/results_agentic_rag.json"))
rows = [("All", r["ALL"])] + [(l, r["lang:" + l]) for l in LANGS] + [("Cross-language", r["lang:Cross-language"])]
cols = ["Group", "n", "Correct/partial/wrong", "Strict acc.", "Numeric match", "All facts cited"]
cells = [[n, str(v["n"]), f'{v["correct"]} / {v["partial"]} / {v["incorrect"]}', f'{v["accuracy_strict"]*100:.1f}%',
          f'{v["auto_numeric_match"]*100:.1f}%', f'{v["cited_all_facts"]*100:.1f}%'] for n, v in rows]
fig, ax = plt.subplots(figsize=(7, 3.9), dpi=200); ax.axis("off"); fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.0)
t = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center", colLoc="center",
             colWidths=[0.19, 0.07, 0.25, 0.14, 0.17, 0.18])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.42)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(GRID); c.set_linewidth(0.6)
    if i == 0: c.set_text_props(color=INK2, fontweight="bold"); c.set_facecolor("#f4f3f0")
    elif i == 1: c.set_text_props(fontweight="bold")
    if j == 0 and i > 0: c._loc = "left"; c.set_text_props(ha="left")
title(fig, "Agentic RAG on all 1,000 questions: 99.8% strictly correct",
      "Claude agents with the search tool only; graded by separate judge instances")
fig.savefig(OUT + "table_agentic.png"); plt.close(fig)
print("figures written")
