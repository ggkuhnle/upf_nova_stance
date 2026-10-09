"""Step 2b: re-classify stance from full text, for papers fetched by fetch_fulltext.py.

An 8B model cannot read a whole paper, so for each paper we build a short context:
  - every sentence mentioning NOVA / a processing classification, +/- one sentence
  - the start of the Conclusion/Discussion section
(references are removed first). That context goes to Ollama with the same prompt as the
abstract run. Checkpointed per paper (data/stance_fulltext_<model>.jsonl); re-run to resume.

Result: data/stance.csv is updated. Papers with a usable full-text context get the
full-text stance; all others keep the abstract-based one.
Columns added: stance_abstract (kept), stance_fulltext, basis ('fulltext'|'abstract').
Use --show N to print the contexts for N papers without calling the model.
"""
import argparse
import json
import re

import pandas as pd

from classify_stance import NOVA_RE, classify_ollama, sentences
from common import DATA

FT = DATA / "fulltext"
HEAD_RE = re.compile(r"^(#+\s*)?(\d+\.?\s*)?(discussion|conclusions?|concluding remarks|"
                     r"strengths and limitations|limitations)\b.*$", re.I | re.M)
REF_RE = re.compile(r"^\s*(#+\s*)?references\s*$", re.I | re.M)
MAX_NOVA, MAX_CONC = 4500, 1200


def build_context(text):
    text = re.sub(r"# ABSTRACT.*?(?=# BODY)", "", text, flags=re.S)   # abstract was already classified
    refs = list(REF_RE.finditer(text))
    if refs:                                  # PDF text: drop the reference list
        text = text[:refs[-1].start()]
    flat = re.sub(r"\s+", " ", text)
    sents = sentences(flat)
    keep = set()
    for i, s in enumerate(sents):
        if NOVA_RE.search(s):
            keep.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(sents))
    n_hits = sum(1 for s in sents if NOVA_RE.search(s))
    passages, size = [], 0
    for i in sorted(keep):
        if size + len(sents[i]) > MAX_NOVA:
            break
        passages.append(sents[i][:600])
        size += len(sents[i])
    conc = ""
    m = HEAD_RE.search(text)
    if m:
        tail = re.sub(r"\s+", " ", text[m.start():])
        conc = tail[:MAX_CONC]
    ctx = " ".join(passages)
    if conc:
        ctx += "\n\nCONCLUSION/DISCUSSION START: " + conc
    return ctx, n_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--show", type=int, help="print contexts for N papers and exit")
    a = ap.parse_args()

    st = pd.read_csv(DATA / "stance.csv")
    if "stance_abstract" not in st:            # first time: remember the abstract-based labels
        st["stance_abstract"] = st["stance"]
    st["title"] = st["title"].fillna("").astype(str)
    ids = {p.stem for p in FT.glob("*.txt")}
    cand = st[st["work_id"].str.split("/").str[-1].isin(ids)]
    if a.limit:
        cand = cand.head(a.limit)

    if a.show:
        for r in cand.head(a.show).itertuples():
            ctx, n = build_context((FT / f"{r.work_id.split('/')[-1]}.txt").read_text(errors="ignore"))
            print(f"\n=== {r.title[:90]}  [{n} NOVA sentences, abstract stance: {r.stance_abstract}]\n{ctx[:1500]}")
        return

    tag = a.model.replace(":", "-").replace("/", "-")
    ckpt = DATA / f"stance_fulltext_{tag}.jsonl"
    if a.fresh and ckpt.exists():
        ckpt.unlink()
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["work_id"]] = d
            except json.JSONDecodeError:
                pass
    todo = cand[~cand["work_id"].isin(done)]
    print(f"{len(cand)} papers with full text, {len(done)} done, {len(todo)} to do", flush=True)

    fails = 0
    try:
        with open(ckpt, "a") as fh:
            for k, r in enumerate(todo.itertuples(), 1):
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}", flush=True)
                ctx, n = build_context((FT / f"{r.work_id.split('/')[-1]}.txt").read_text(errors="ignore"))
                if n == 0:                      # NOVA never named in the full text
                    d = dict(work_id=r.work_id, stance_fulltext="not_discussed", engages_ft=False, nova_sentences_ft=0)
                else:
                    try:
                        stance, eng = classify_ollama(a.model, r.title, ctx)
                    except Exception as e:
                        fails += 1
                        print(f"  failed {r.work_id}: {e}", flush=True)
                        if fails >= 20:
                            raise RuntimeError("20 failures; is Ollama running?")
                        continue
                    fails = 0
                    d = dict(work_id=r.work_id, stance_fulltext=stance, engages_ft=bool(eng), nova_sentences_ft=n)
                done[r.work_id] = d
                fh.write(json.dumps(d) + "\n")
                fh.flush()
    except KeyboardInterrupt:
        print("Interrupted; progress saved. Re-run to resume.")
    finally:
        ft = pd.DataFrame(done.values())
        if len(ft):
            st = st.drop(columns=[c for c in ("stance_fulltext", "engages_ft", "nova_sentences_ft", "basis") if c in st])
            st = st.merge(ft, on="work_id", how="left")
            has = st["stance_fulltext"].notna()
            st["basis"] = has.map({True: "fulltext", False: "abstract"})
            st["stance"] = st["stance_fulltext"].where(has, st["stance_abstract"])
            st["engages_critique"] = st["engages_ft"].where(has, st["engages_critique"]).astype(bool)
            st.to_csv(DATA / "stance.csv", index=False)
            print(pd.crosstab(st.loc[has, "stance_abstract"], st.loc[has, "stance_fulltext"]).to_string())
            print(f"{int(has.sum())} papers on full text, {int((~has).sum())} on abstract -> {DATA / 'stance.csv'}")


if __name__ == "__main__":
    main()
