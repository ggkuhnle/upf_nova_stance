"""Step 4: how much does supportive literature cite critical literature, and vice versa?

Restricted to papers that discuss NOVA (stance != not_discussed) and to citations inside the corpus.
For every ordered pair of stances (citing -> cited) it reports
  - n citations, and the share of the citing group's citations of NOVA-discussing papers
  - share of citing papers that cite >=1 paper of the cited stance
  - enrichment: observed share / share expected if papers cited NOVA-discussing papers at random
    among those published up to the citing paper's year (controls for critical papers being few and
    mostly early or late).
Writes output/cross_citation.md and output/cross_citation_pairs.csv (every supportive<->critical link).
"""
import pandas as pd

from common import BIB, DATA, OUT

GROUPS = ["critical", "mixed", "supportive", "neutral_use"]


def main():
    st = pd.read_csv(DATA / "stance.csv")
    st = st[st.stance.isin(GROUPS)].copy()
    info = st.set_index("work_id")
    e = pd.read_csv(BIB / "citation_edges_work.csv").drop_duplicates()
    e = e[e.citing_work_id.isin(info.index) & e.cited_work_id.isin(info.index) & (e.citing_work_id != e.cited_work_id)]
    e = e.assign(cs=e.citing_work_id.map(info.stance), ds=e.cited_work_id.map(info.stance),
                 cy=e.citing_work_id.map(info.year), dy=e.cited_work_id.map(info.year))

    # expected share of each cited stance, per citing year: stance mix of papers published <= that year
    yrs = sorted(st.year.dropna().unique())
    mix = {y: st[st.year <= y].stance.value_counts(normalize=True).reindex(GROUPS, fill_value=0) for y in yrs}

    rows = []
    for a in GROUPS:
        papers_a = st[st.stance == a]
        ea = e[e.cs == a]
        for b in GROUPS:
            eab = ea[ea.ds == b]
            exp = ea.cy.map(lambda y: mix[y][b] if y in mix else 0).sum() / max(1, len(ea))
            obs = len(eab) / max(1, len(ea))
            rows.append(dict(citing=a, cited=b, n_papers_citing_group=len(papers_a), n_citations=len(eab),
                             share_of_citations=round(obs, 3), expected_share=round(exp, 3),
                             enrichment=round(obs / exp, 2) if exp else float("nan"),
                             citing_papers_with_link=eab.citing_work_id.nunique(),
                             pct_papers_with_link=round(100 * eab.citing_work_id.nunique() / max(1, len(papers_a)), 1)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cross_citation_matrix.csv", index=False)

    pairs = e[((e.cs == "supportive") & (e.ds == "critical")) | ((e.cs == "critical") & (e.ds == "supportive"))].copy()
    for side, col in (("citing", "citing_work_id"), ("cited", "cited_work_id")):
        pairs[f"{side}_title"] = pairs[col].map(info.title).str[:90]
        pairs[f"{side}_year"] = pairs[col].map(info.year)
    pairs[["cs", "citing_year", "citing_title", "ds", "cited_year", "cited_title"]].sort_values(
        ["cs", "cited_title"]).to_csv(OUT / "cross_citation_pairs.csv", index=False)

    def pick(a, b):
        return res[(res.citing == a) & (res.cited == b)].iloc[0]

    sc, cs_ = pick("supportive", "critical"), pick("critical", "supportive")
    top_crit = (e[(e.cs == "supportive") & (e.ds == "critical")].groupby("cited_work_id").size()
                .sort_values(ascending=False).head(10))
    top_supp = (e[(e.cs == "critical") & (e.ds == "supportive")].groupby("cited_work_id").size()
                .sort_values(ascending=False).head(10))
    blk = lambda s: "```\n" + s.to_string() + "\n```"
    md = [f"# Cross-citation between stances (NOVA-discussing papers, {len(st)} papers)", "",
          f"**Supportive -> critical:** {int(sc.n_citations)} citations; {sc.pct_papers_with_link}% of supportive papers "
          f"({int(sc.citing_papers_with_link)}/{int(sc.n_papers_citing_group)}) cite at least one critical paper. "
          f"Share of their citations to NOVA papers: {sc.share_of_citations:.1%} (expected {sc.expected_share:.1%}, enrichment {sc.enrichment}x).", "",
          f"**Critical -> supportive:** {int(cs_.n_citations)} citations; {cs_.pct_papers_with_link}% of critical papers "
          f"({int(cs_.citing_papers_with_link)}/{int(cs_.n_papers_citing_group)}) cite at least one supportive paper. "
          f"Share: {cs_.share_of_citations:.1%} (expected {cs_.expected_share:.1%}, enrichment {cs_.enrichment}x).", "",
          "## Full matrix", "", blk(res[["citing", "cited", "n_citations", "share_of_citations", "expected_share", "enrichment", "pct_papers_with_link"]].set_index(["citing", "cited"])),
          "", "## Critical papers most cited by supportive papers", "",
          blk(pd.DataFrame({"cited_by_supportive": top_crit, "year": top_crit.index.map(info.year),
                            "title": top_crit.index.map(info.title).str[:95]}).reset_index(drop=True)),
          "", "## Supportive papers most cited by critical papers", "",
          blk(pd.DataFrame({"cited_by_critical": top_supp, "year": top_supp.index.map(info.year),
                            "title": top_supp.index.map(info.title).str[:95]}).reset_index(drop=True))]
    (OUT / "cross_citation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
