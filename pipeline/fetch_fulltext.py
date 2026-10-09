"""Step 1b: download legally open-access full text by DOI.

Order of attempts per paper:
  1. Europe PMC  - JATS XML for PubMed Central open-access papers (cleanest)
  2. Unpaywall   - best open-access PDF location (needs mailto in config.json)

Candidates: papers whose stance in data/stance.csv is not 'not_discussed'
(use --all for the whole corpus). Resumable: finished papers are listed in
data/fulltext_manifest.csv and skipped; failures ('none') are retried only with --retry.

Writes  data/fulltext/<openalex id>.xml | .pdf   original file
        data/fulltext/<openalex id>.txt          plain text with section headings
        data/fulltext_manifest.csv               work_id, doi, source, status
Paywalled papers get status 'none': download those yourself into pdf_dir.
"""
import argparse
import csv
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from common import BIB, CFG, DATA

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2/"
FT = DATA / "fulltext"
MANIFEST = DATA / "fulltext_manifest.csv"
FIELDS = ["work_id", "doi", "source", "status"]
HEADERS = {"User-Agent": f"upf_nova_stance (mailto:{CFG['mailto']})"}
SKIP_SEC = {"ref-list", "ack", "fn-group", "app-group", "glossary"}


class Transient(Exception):
    """Service unreachable/overloaded: result unknown, so retry on the next run."""


def get(url, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise Transient(url)


def xml_to_text(xml_bytes):
    """Title/abstract/body of a JATS article as text, with '## heading' lines; no references."""
    root = ET.fromstring(xml_bytes)
    out = []

    def walk(el):
        if el.tag in SKIP_SEC:
            return
        if el.tag == "title":
            out.append("\n## " + "".join(el.itertext()).strip())
            return
        if el.tag == "p":
            out.append("".join(el.itertext()).strip())
            return
        for ch in el:
            walk(ch)

    for tag in ("abstract", "body"):
        for el in root.iter(tag):
            out.append(f"\n# {tag.upper()}")
            walk(el)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def try_europepmc(doi, stem):
    r = get(f"{EPMC}/search", params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"})
    if r.status_code != 200:
        return None
    hits = r.json().get("resultList", {}).get("result", [])
    hit = next((h for h in hits if h.get("pmcid") and h.get("isOpenAccess") == "Y"), None)
    if not hit:
        return None
    x = get(f"{EPMC}/{hit['pmcid']}/fullTextXML")
    if x.status_code != 200 or not x.content.lstrip().startswith(b"<"):
        return None
    try:
        text = xml_to_text(x.content)
    except ET.ParseError:
        return None
    if len(text) < 2000:
        return None
    (FT / f"{stem}.xml").write_bytes(x.content)
    (FT / f"{stem}.txt").write_text(text, encoding="utf-8")
    return "europepmc"


def try_unpaywall(doi, stem):
    r = get(UNPAYWALL + doi, params={"email": CFG["mailto"]})
    if r.status_code != 200:
        return None
    loc = r.json().get("best_oa_location") or {}
    url = loc.get("url_for_pdf")
    if not url:
        return None
    p = get(url)
    if p.status_code != 200 or not p.content.startswith(b"%PDF") or len(p.content) > 40e6:
        return None
    (FT / f"{stem}.pdf").write_bytes(p.content)
    try:                      # text extraction is optional (needs pdfplumber)
        import io
        import pdfplumber
        with pdfplumber.open(io.BytesIO(p.content)) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        (FT / f"{stem}.txt").write_text(text, encoding="utf-8")
    except ImportError:
        pass
    return "unpaywall"


def extract_pdfs():
    """Create the missing .txt for every saved .pdf (needs pdfplumber)."""
    import pdfplumber
    todo = [p for p in sorted(FT.glob("*.pdf")) if not p.with_suffix(".txt").exists()]
    print(f"{len(todo)} PDFs to extract", flush=True)
    for k, p in enumerate(todo, 1):
        try:
            with pdfplumber.open(p) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        except Exception as e:
            print(f"  failed {p.name}: {e}", flush=True)
            continue
        if len(text) > 2000:
            p.with_suffix(".txt").write_text(text, encoding="utf-8")
        else:
            print(f"  little/no text in {p.name} (scanned?)", flush=True)
        if k % 20 == 0:
            print(f"  {k}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-pdfs", action="store_true", help="only extract text from saved PDFs, then exit")
    ap.add_argument("--all", action="store_true", help="whole corpus, not just NOVA-discussing papers")
    ap.add_argument("--retry", action="store_true", help="retry papers previously marked 'none'")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    FT.mkdir(exist_ok=True)
    if a.extract_pdfs:
        return extract_pdfs()

    papers = pd.read_csv(BIB / "papers_detail.csv").drop_duplicates("work_id")
    papers = papers[papers["doi"].notna()]
    if not a.all:
        st = pd.read_csv(DATA / "stance.csv")
        papers = papers[papers["work_id"].isin(st.loc[st.stance != "not_discussed", "work_id"])]

    done = {}
    if MANIFEST.exists():
        done = pd.read_csv(MANIFEST).set_index("work_id")["status"].to_dict()
    skip = {w for w, s in done.items() if s == "ok" or (s == "none" and not a.retry)}
    todo = papers[~papers["work_id"].isin(skip)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(papers)} candidates, {len(skip)} done, {len(todo)} to try", flush=True)

    errors = 0
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        try:
            for k, r in enumerate(todo.itertuples(), 1):
                doi = r.doi.replace("https://doi.org/", "")
                stem = r.work_id.split("/")[-1]
                try:
                    src = try_europepmc(doi, stem) or try_unpaywall(doi, stem)
                except Transient as e:       # not recorded as a result; retried next run
                    errors += 1
                    print(f"  service error for {doi}: {e}", flush=True)
                    if errors >= 15:
                        print("Too many service errors; stopping. Re-run later to resume.")
                        break
                    continue
                w.writerow(dict(work_id=r.work_id, doi=doi, source=src or "", status="ok" if src else "none"))
                fh.flush()
                if k % 20 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Interrupted; progress saved. Re-run to resume.")

    m = pd.read_csv(MANIFEST).drop_duplicates("work_id", keep="last")
    print(m["status"].value_counts().to_string())
    print(m[m.status == "ok"]["source"].value_counts().to_string())
    print("manifest ->", MANIFEST)


if __name__ == "__main__":
    main()
