"""Step 2: classify each paper's stance towards NOVA.

Only sentences that mention NOVA / the UPF classification are scored, so generic
words ("however", "valid") elsewhere in the text no longer drive the result.

Labels: critical | mixed | supportive | neutral_use | not_discussed
Also:   engages_critique = abstract holds both critical and supportive cues
        in NOVA sentences (a crude proxy for engagement with the debate).

Backends:
  --backend rules   (default) sentence-level cue matching on abstracts
                    (plus PDFs if config 'pdf_dir' is set; needs pdfplumber)
  --backend llm     Claude via the anthropic package (needs ANTHROPIC_API_KEY)
  --backend ollama  local Llama (or any Ollama model) via http://localhost:11434;
                    free, no tokens billed. Model set with --model (default llama3.1:8b).
"""
import argparse
import json
import re

import pandas as pd
import requests

from common import BIB, CFG, DATA

NOVA_RE = re.compile(r"\bNOVA\b|ultra[- ]?processed\s+(food\s+)?(classification|definition|concept)|"
                     r"food[- ]processing\s+classification|classif\w+\s+(of\s+)?(foods?\s+)?by\s+(the\s+)?(extent|degree)", re.I)

CRITICAL = [r"limitation", r"critic", r"critique", r"flaw", r"ambigu", r"misclassif", r"inconsisten",
            r"oversimplif", r"poorly defined", r"not (well[- ])?(defined|validated)", r"reductionis",
            r"subjectiv", r"lack(s|ing)? (of )?(a )?(clear|consistent|precise|validated|scientific)",
            r"problematic", r"questionabl", r"inter-?rater", r"low (agreement|reliability)",
            r"vague", r"arbitrar", r"unclear", r"challeng\w+ the (use|validity)", r"too broad",
            r"heterogene\w+ (within|of)", r"misleading", r"fails? to", r"does not (capture|distinguish|account)",
            r"controvers", r"debate"]
SUPPORTIVE = [r"\bvalid(ated|ity)?\b", r"robust", r"reliab", r"useful", r"practical", r"well[- ]established",
              r"evidence[- ]based", r"widely (used|adopted|applied)", r"framework", r"enables?", r"allows?",
              r"supports?", r"consistent with", r"appropriate", r"informative", r"valuable", r"strong(ly)? (associat|evidence)",
              r"good (agreement|reliability)", r"effective"]
CRIT_RE = re.compile("|".join(CRITICAL), re.I)
SUPP_RE = re.compile("|".join(SUPPORTIVE), re.I)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def classify_rules(text):
    if not text:
        return "not_discussed", 0, 0, 0
    sents = [s for s in sentences(text) if NOVA_RE.search(s)]
    if not sents:
        return "not_discussed", 0, 0, 0
    # also look one sentence ahead: critique often follows the NOVA mention
    allsents = sentences(text)
    ctx = set()
    for i, s in enumerate(allsents):
        if NOVA_RE.search(s):
            ctx.update(range(i, min(i + 2, len(allsents))))
    scored = " ".join(allsents[i] for i in sorted(ctx))
    c, s_ = len(CRIT_RE.findall(scored)), len(SUPP_RE.findall(scored))
    if c and s_:
        label = "critical" if c >= 2 * s_ else "supportive" if s_ >= 2 * c else "mixed"
    elif c:
        label = "critical"
    elif s_:
        label = "supportive"
    else:
        label = "neutral_use"
    return label, len(sents), c, s_


def pdf_text(pdf_dir, doi):
    """Return full text of <pdf_dir>/<doi with / -> _>.pdf if present."""
    if not pdf_dir or not doi:
        return ""
    from pathlib import Path
    p = Path(pdf_dir) / (doi.replace("https://doi.org/", "").replace("/", "_") + ".pdf")
    if not p.exists():
        return ""
    import pdfplumber
    with pdfplumber.open(p) as pdf:
        return " ".join((pg.extract_text() or "") for pg in pdf.pages)


LLM_PROMPT = """You assess how a scientific paper treats the NOVA food classification
(ultra-processed foods). From the text below answer ONLY with JSON:
{"stance": "critical|mixed|supportive|neutral_use|not_discussed",
 "engages_critique": true|false}
critical = argues NOVA/UPF concept is flawed or limited; supportive = endorses it as valid/useful;
mixed = both; neutral_use = just applies it; not_discussed = NOVA not addressed.
engages_critique = explicitly discusses or rebuts criticisms of NOVA.

TITLE: {title}
TEXT: {text}"""


def classify_llm(client, title, text):
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=100,
                                 messages=[{"role": "user", "content": LLM_PROMPT.replace("{title}", title).replace("{text}", text[:6000])}])
    d = json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group(0))
    return d["stance"], bool(d["engages_critique"])


def classify_ollama(model, title, text):
    prompt = LLM_PROMPT.replace("{title}", title).replace("{text}", text[:6000])
    r = requests.post("http://localhost:11434/api/generate", timeout=300,
                      json={"model": model, "prompt": prompt, "format": "json", "stream": False,
                            "options": {"temperature": 0}})
    r.raise_for_status()
    d = json.loads(r.json()["response"])
    stance = d.get("stance", "")
    if stance not in ("critical", "mixed", "supportive", "neutral_use", "not_discussed"):
        stance = "neutral_use"
    return stance, bool(d.get("engages_critique"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "llm", "ollama"], default="rules")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true", help="discard the checkpoint and start over")
    a = ap.parse_args()

    papers = pd.read_csv(BIB / "papers_detail.csv").drop_duplicates("work_id")
    abst = pd.read_csv(DATA / "abstracts.csv").fillna("").drop_duplicates("work_id")
    papers["short_id"] = papers["work_id"].str.replace("https://openalex.org/", "", regex=False)
    df = papers.merge(abst.rename(columns={"work_id": "short_id"}), on="short_id", how="inner")
    df["title"] = df["title"].fillna("").astype(str)
    if a.limit:
        df = df.head(a.limit)

    client = None
    if a.backend == "llm":
        import anthropic
        client = anthropic.Anthropic()

    # Checkpoint: one JSON line per finished paper, flushed immediately, so an
    # interrupted run resumes where it stopped. Separate file per backend/model.
    tag = a.backend + (("_" + a.model.replace(":", "-").replace("/", "-")) if a.backend == "ollama" else "")
    ckpt = DATA / f"stance_{tag}.jsonl"
    if a.fresh and ckpt.exists():
        ckpt.unlink()
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["work_id"]] = d
            except json.JSONDecodeError:   # half-written last line after a crash
                pass
    todo = df[~df["work_id"].isin(done)]
    print(f"{len(df)} papers, {len(done)} already done, {len(todo)} to do (checkpoint: {ckpt.name})", flush=True)

    fails = 0
    try:
        with open(ckpt, "a") as fh:
            for k, r in enumerate(todo.itertuples(), 1):
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}", flush=True)
                full = pdf_text(CFG.get("pdf_dir"), r.doi)
                text = full or f"{r.title}. {r.abstract}"
                try:
                    if a.backend == "ollama":
                        stance, eng = classify_ollama(a.model, r.title, text)
                        n = c = s = 0
                    elif client:
                        stance, eng = classify_llm(client, r.title, text)
                        n = c = s = 0
                    else:
                        stance, n, c, s = classify_rules(text)
                        eng = c > 0 and s > 0
                except Exception as e:      # skip; retried on next run
                    fails += 1
                    print(f"  failed {r.work_id}: {e}", flush=True)
                    if fails >= 20:
                        raise RuntimeError("20 failures in a row; is the backend running?")
                    continue
                fails = 0
                d = dict(work_id=r.work_id, stance=stance, nova_sentences=n, critical_cues=c,
                         supportive_cues=s, engages_critique=bool(eng), source="pdf" if full else "abstract")
                done[r.work_id] = d
                fh.write(json.dumps(d) + "\n")
                fh.flush()
    except KeyboardInterrupt:
        print("Interrupted; progress is saved. Re-run the same command to resume.")
    finally:
        out = pd.DataFrame([done[w] for w in df["work_id"] if w in done])
        if len(out) and a.backend != "rules":
            # Gate: an LLM stance only counts if title/abstract actually mention NOVA
            # (or a processing classification). Otherwise the LLM tends to label any
            # UPF paper as applying/supporting NOVA. Original label kept in llm_stance.
            txt = df.set_index("work_id").apply(lambda r: f"{r.title}. {r.abstract}", axis=1)
            mentions = out["work_id"].map(lambda w: bool(NOVA_RE.search(txt[w])))
            out["llm_stance"] = out["stance"]
            out.loc[~mentions, "stance"] = "not_discussed"
            out.loc[~mentions, "engages_critique"] = False
        if len(out):
            res = df[["work_id", "title", "doi", "year", "journal", "citations", "study_type", "country"]].merge(out, on="work_id")
            res.to_csv(DATA / "stance.csv", index=False)
            print(res["stance"].value_counts().to_string())
            print(f"{len(res)}/{len(df)} papers -> {DATA / 'stance.csv'}")


if __name__ == "__main__":
    main()
