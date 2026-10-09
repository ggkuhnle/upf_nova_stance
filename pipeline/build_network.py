"""Step 3: overlay stance on the bibliography's citation network.

Reads  data/stance.csv and <bibliography_data_dir>/citation_edges_work.csv
Writes output/stance_network_cytoscape.html  interactive Cytoscape.js network (see build_cytoscape.py)
       output/stance_by_year.html      stance mix per year
       output/stance_citation_matrix.csv  citing-stance x cited-stance edge counts
       output/summary.md
Edges are stored (citing, cited) and flipped only for display, as in upf_bibliography.
"""
import pandas as pd
import plotly.graph_objects as go

from common import BIB, COLOURS, OUT, STANCES, DATA


def block(df, **kw):
    return "```\n" + df.to_string(**kw) + "\n```"


def main():
    st = pd.read_csv(DATA / "stance.csv")
    edges = pd.read_csv(BIB / "citation_edges_work.csv")
    info = st.set_index("work_id")
    edges = edges[edges.citing_work_id.isin(info.index) & edges.cited_work_id.isin(info.index)]

    # citing-stance x cited-stance matrix
    e = edges.assign(citing=edges.citing_work_id.map(info.stance), cited=edges.cited_work_id.map(info.stance))
    mat = pd.crosstab(e.citing, e.cited).reindex(index=STANCES, columns=STANCES, fill_value=0)
    mat.to_csv(OUT / "stance_citation_matrix.csv")
    share = mat.div(mat.sum(axis=1).replace(0, 1), axis=0).round(3)

    # stance by year
    yr = pd.crosstab(st.year, st.stance).reindex(columns=STANCES, fill_value=0)
    fig = go.Figure([go.Bar(x=yr.index, y=yr[s], name=s, marker_color=COLOURS[s]) for s in STANCES])
    fig.update_layout(barmode="stack", title="Papers by stance towards NOVA", xaxis_title="Year", yaxis_title="Papers")
    fig.write_html(OUT / "stance_by_year.html")

    import build_cytoscape            # interactive explorer (replaces the old Plotly network)
    build_cytoscape.main()

    top = st[st.stance == "critical"].nlargest(10, "citations")[["year", "citations", "title"]]
    md = ["# NOVA stance summary", "", f"Papers classified: {len(st)}", "",
          block(st.stance.value_counts()), "",
          "## Who cites whom (row = citing stance, share of its citations)", "", block(share), "",
          f"Engages with critique (among NOVA-discussing papers): "
          f"{st[st.stance != 'not_discussed'].engages_critique.mean():.1%}", "",
          "## Most-cited critical papers", "", block(top, index=False)]
    (OUT / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
