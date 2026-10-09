"""Step 1: fetch abstracts from OpenAlex for every paper in the bibliography corpus.

Reads  <bibliography_data_dir>/papers_detail.csv   (from upf_bibliography)
Writes data/abstracts.csv  (work_id, abstract); resumable.
"""
import sys
import time

import pandas as pd
import requests

from common import BIB, CFG, DATA

URL = "https://api.openalex.org/works"


def rebuild(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def main():
    papers = pd.read_csv(BIB / "papers_detail.csv")
    papers = papers[papers["year"] >= CFG["min_year"]]
    ids = papers["work_id"].str.replace("https://openalex.org/", "", regex=False).tolist()
    out = DATA / "abstracts.csv"
    done = set(pd.read_csv(out)["work_id"]) if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} papers, {len(todo)} to fetch")
    rows = []
    s = requests.Session()
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        for attempt in range(5):
            r = s.get(URL, params={"filter": "openalex_id:" + "|".join(batch),
                                   "select": "id,abstract_inverted_index",
                                   "per-page": 50, "mailto": CFG["mailto"]}, timeout=30)
            if r.status_code == 429:
                time.sleep(min(30 * (attempt + 1), 120))
                continue
            r.raise_for_status()
            break
        got = {w["id"].split("/")[-1]: rebuild(w.get("abstract_inverted_index"))
               for w in r.json()["results"]}
        rows += [{"work_id": i, "abstract": got.get(i, "")} for i in batch]
        if (k // 50) % 10 == 9 or k + 50 >= len(todo):
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            rows = []
            print(f"  {min(k + 50, len(todo))}/{len(todo)}", file=sys.stderr)
    print("done ->", out)


if __name__ == "__main__":
    main()
