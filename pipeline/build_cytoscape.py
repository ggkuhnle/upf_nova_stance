"""Step 3b: interactive Cytoscape.js explorer of the paper-level citation network,
coloured by stance towards NOVA. Layout and look follow upf_bibliography's
citation_network_cytoscape.html (control bar, side panel, ego highlight, influence tracing).

Reads  data/stance.csv + <bibliography_data_dir>/citation_edges_work.csv
Writes output/stance_network_cytoscape.html   (needs internet for the Cytoscape CDN)
Edges are stored (citing, cited) in the data and flipped here: arrows run cited -> citing.
"""
import html
import json
from datetime import date

import networkx as nx
import pandas as pd

from common import BIB, CFG, COLOURS, OUT, STANCES, DATA

CYTOSCAPE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"
ARROW_NOTE = "Arrows point from cited work → citing work (influence direction)."
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692",
           "#B6E880", "#FF97FF", "#FECB52", "#1F77B4", "#FF7F0E"]
SCALE = 900


def build():
    st = pd.read_csv(DATA / "stance.csv")
    for c in ("title", "journal", "study_type", "country", "doi", "basis", "stance_abstract"):
        if c not in st:
            st[c] = ""
        st[c] = st[c].fillna("").astype(str)
    if "basis" in st and not st["basis"].any():
        st["basis"] = "abstract"
    info = st.set_index("work_id")
    edges = pd.read_csv(BIB / "citation_edges_work.csv")
    edges = edges[edges.citing_work_id.isin(info.index) & edges.cited_work_id.isin(info.index)]
    edges = edges[edges.citing_work_id != edges.cited_work_id].drop_duplicates()

    g = nx.DiGraph()
    g.add_edges_from(zip(edges.cited_work_id, edges.citing_work_id))   # display direction
    ids = list(g.nodes)
    # Layout: spring layout of the NOVA-discussing papers only (the default view), so they are
    # not crushed by the thousands of other papers; the rest sit at the centroid of their
    # NOVA-discussing neighbours (jittered), or on an outer ring if they have none.
    import math
    import random
    rnd = random.Random(42)
    core = [n for n in ids if info.loc[n, "stance"] != "not_discussed"]
    ug = g.to_undirected()
    sub = ug.subgraph(core)
    comps = sorted(nx.connected_components(sub), key=len, reverse=True)
    main_c = sub.subgraph(comps[0])
    pos = nx.spring_layout(main_c, seed=42, k=3.0 / max(1, len(main_c)) ** 0.5, iterations=200, scale=1.0)
    # small disconnected groups (spring layouts fling them far away): small local layouts on a spiral
    for i, comp in enumerate(comps[1:]):
        a = i * 2.4  # golden-angle spiral
        r = 1.3 + 0.06 * math.sqrt(i)
        cx, cy0 = r * math.cos(a), r * math.sin(a)
        loc = nx.spring_layout(sub.subgraph(comp), seed=1, scale=0.02 * math.sqrt(len(comp))) if len(comp) > 1 else {next(iter(comp)): (0, 0)}
        for n, (x, y) in loc.items():
            pos[n] = (cx + x, cy0 + y)
    ring = []
    for n in ids:
        if n in pos:
            continue
        nb = [pos[m] for m in ug.neighbors(n) if m in pos]
        if nb:
            cx = sum(p[0] for p in nb) / len(nb)
            cy_ = sum(p[1] for p in nb) / len(nb)
            pos[n] = (cx + rnd.gauss(0, 0.04), cy_ + rnd.gauss(0, 0.04))
        else:
            ring.append(n)
    for i, n in enumerate(ring):
        a = 2 * math.pi * i / max(1, len(ring))
        pos[n] = (1.2 * math.cos(a) * 2.2 + rnd.gauss(0, 0.03), 1.2 * math.sin(a) * 2.2 + rnd.gauss(0, 0.03))
    order = {n: i for i, n in enumerate(sorted(ids, key=lambda n: g.out_degree(n), reverse=True))}
    idx = {n: i for i, n in enumerate(ids)}

    types = [t for t in info.loc[ids, "study_type"].value_counts().index if t]
    tcol = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}
    nodes = []
    for n in ids:
        r = info.loc[n]
        nodes.append(dict(
            id=str(idx[n]), label=f"{r.title[:55]} ({int(r.year) if pd.notna(r.year) else ''})", title=r.title,
            year=int(r.year) if pd.notna(r.year) else 0, journal=r.journal, cites=int(r.citations),
            indeg=g.out_degree(n),            # papers in the corpus that cite this one
            outdeg=g.in_degree(n), stance=r.stance, stance_abs=r.stance_abstract, basis=r.basis,
            doi=r.doi, stype=r.study_type, country=r.country, eng=bool(r.engages_critique),
            rank=order[n] + 1, x=round(float(pos[n][0]) * SCALE, 1), y=round(float(pos[n][1]) * SCALE, 1)))
    cy_nodes = [{"data": d, "position": {"x": d.pop("x"), "y": d.pop("y")}} for d in nodes]
    cy_edges = [{"data": {"id": f"e{i}", "source": str(idx[a]), "target": str(idx[b])}}
                for i, (a, b) in enumerate(g.edges)]
    out_adj = {str(idx[n]): [idx[m] for m in g.successors(n)] for n in ids}     # cited -> citing
    in_adj = {str(idx[n]): [idx[m] for m in g.predecessors(n)] for n in ids}
    counts = pd.Series([d["data"]["stance"] for d in cy_nodes]).value_counts().to_dict()
    return cy_nodes, cy_edges, out_adj, in_adj, counts, tcol


CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;background:#f4f6f8}
#hdr{background:#1a2742;color:#fff;padding:10px 20px;display:flex;align-items:center;gap:16px;flex-shrink:0}
#hdr h1{font-size:1.05rem;font-weight:700}#hdr p{font-size:.72rem;opacity:.65;margin-top:1px}
#ctrl{background:#fff;border-bottom:1px solid #d8dde5;padding:7px 16px;display:flex;flex-wrap:wrap;gap:7px 14px;align-items:center;flex-shrink:0}
.cg{display:flex;flex-direction:column;gap:2px}
.cg>label{font-size:.66rem;font-weight:700;color:#999;text-transform:uppercase;letter-spacing:.04em}
.sep{width:1px;height:28px;background:#dde1e7;margin:0 2px;align-self:center;flex-shrink:0}
select{padding:4px 7px;border:1px solid #c8ced5;border-radius:5px;font-size:.8rem;background:#fff;color:#333}
.btn-grp{display:flex}.tog{background:#eee;border:1px solid #ccc;padding:4px 10px;cursor:pointer;font-size:.78rem;color:#555}
.tog:first-child{border-radius:5px 0 0 5px}.tog:last-child{border-radius:0 5px 5px 0}
.tog.active{background:#1a2742;color:#fff;border-color:#1a2742}
.reset-btn{background:#e74c3c;color:#fff;border:none;border-radius:5px;padding:4px 10px;cursor:pointer;font-size:.78rem;display:none}
.srch-wrap{position:relative}.srch-in{padding:5px 9px;border:1px solid #c8ced5;border-radius:5px;font-size:.82rem;width:220px;outline:none}
.dd{position:absolute;top:100%;left:0;z-index:500;background:#fff;border:1px solid #c8ced5;max-height:240px;overflow-y:auto;width:380px;box-shadow:0 6px 16px rgba(0,0,0,.12);display:none}
.dd-item{padding:5px 9px;cursor:pointer;font-size:.8rem;border-bottom:1px solid #f2f2f2}.dd-item:hover{background:#f0f4ff}
.dd-sub{font-size:.68rem;color:#999}
.chk{display:flex;align-items:center;gap:4px;font-size:.78rem;color:#444;cursor:pointer;white-space:nowrap}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
#workspace{display:flex;flex:1;min-height:0;position:relative}#cy{flex:1;min-width:0;background:#fff}
#legend{position:absolute;left:10px;bottom:10px;background:rgba(255,255,255,.92);border:1px solid #d8dde5;border-radius:6px;padding:6px 10px;font-size:.72rem;max-width:230px}
#legend div{display:flex;align-items:center;gap:6px;margin:2px 0}
#side-panel{width:320px;background:#fff;border-left:1px solid #d8dde5;display:none;flex-direction:column;overflow-y:auto;flex-shrink:0}
#panel-hdr{background:#1a2742;color:#fff;padding:10px 14px;display:flex;justify-content:space-between;gap:8px}
#panel-hdr h3{font-size:.86rem;font-weight:600;line-height:1.3}
#close-panel{background:none;border:none;color:#fff;font-size:1.1rem;cursor:pointer}
#panel-body{padding:10px 14px}.pr{display:flex;justify-content:space-between;gap:10px;padding:3px 0;font-size:.79rem;border-bottom:1px solid #f2f2f2}
.pk{color:#888}.pv{font-weight:600;text-align:right}
#panel-body h4{font-size:.66rem;font-weight:700;color:#888;text-transform:uppercase;margin:10px 0 4px}
.cit-list{list-style:none}.cit-list li{padding:3px 0;font-size:.78rem;cursor:pointer;border-bottom:1px solid #f6f6f6;color:#1a5296}
.cit-list li:hover{text-decoration:underline}.badge{padding:1px 7px;border-radius:9px;color:#fff;font-size:.72rem}
"""

JS = r"""
var STANCES = __STANCES__, SCOL = __SCOL__, TCOL = __TCOL__;
var state = {links:'all', ego:-1, color:'stance', size:'indeg', limit:__TOTAL__, depth:1};
var cy, vis = {};
STANCES.forEach(function(s){ vis[s] = (s !== 'not_discussed'); });
var YMIN = 2009, YMAX = 2026;
function yearColor(y){ var t = Math.max(0, Math.min(1, (y-YMIN)/(YMAX-YMIN))); return 'hsl(' + (220 - 190*t) + ',70%,' + (70-25*t) + '%)'; }
function colorOf(d){
  if (state.color==='stance') return SCOL[d.stance] || '#bbb';
  if (state.color==='year') return d.year ? yearColor(d.year) : '#bbb';
  if (state.color==='stype') return TCOL[d.stype] || '#bbb';
  return d.basis==='fulltext' ? '#2e86ab' : '#c9ced6';
}
function sizeOf(d){ var v = state.size==='indeg' ? d.indeg : d.cites; var p = state.size==='indeg' ? P95a : P95b;
  return 3 + 14*Math.min(1, Math.sqrt(v)/Math.max(1,Math.sqrt(p))); }
var P95a = 1, P95b = 1;
function p95(a){ a = a.slice().sort(function(x,y){return x-y;}); return a[Math.floor(a.length*0.95)] || 1; }

var STYLE = [
 {selector:'node', style:{'background-color':'data(col)','width':'data(sz)','height':'data(sz)','label':'','font-size':12,
   'text-valign':'top','text-wrap':'wrap','text-max-width':160,'color':'#111','text-outline-width':3,
   'text-outline-color':'rgba(255,255,255,0.9)','border-width':0.5,'border-color':'#fff'}},
 {selector:'node.hovered', style:{'label':'data(label)','border-width':2.5,'border-color':'#e74c3c','z-index':9999}},
 {selector:'node.ego', style:{'label':'data(label)','border-width':3,'border-color':'#e74c3c','z-index':100}},
 {selector:'node.neighbor', style:{'border-width':2,'border-color':'rgba(231,76,60,0.5)','z-index':50}},
 {selector:'node.faded', style:{'opacity':0.07}},
 {selector:'node.hidden', style:{'display':'none'}},
 {selector:'node.inf1', style:{'border-width':3,'border-color':'#8e44ad','z-index':110}},
 {selector:'node.inf2', style:{'border-width':2,'border-color':'#bb8fce','z-index':100}},
 {selector:'node.inf3', style:{'border-width':1.5,'border-color':'#d7bde2','z-index':90}},
 {selector:'edge', style:{'width':0.5,'line-color':'rgba(100,100,100,0.14)','target-arrow-shape':'none','curve-style':'straight'}},
 {selector:'edge.faded', style:{'opacity':0.03}},
 {selector:'edge.hidden', style:{'display':'none'}},
 {selector:'edge.x-s2c', style:{'width':1.3,'line-color':'rgba(142,68,173,0.75)','target-arrow-shape':'triangle','target-arrow-color':'rgba(142,68,173,0.9)','arrow-scale':0.7}},
 {selector:'edge.x-c2s', style:{'width':1.3,'line-color':'rgba(230,126,34,0.8)','target-arrow-shape':'triangle','target-arrow-color':'rgba(230,126,34,0.9)','arrow-scale':0.7}},
 {selector:'edge.x-c2c', style:{'width':1.3,'line-color':'rgba(209,73,91,0.8)','target-arrow-shape':'triangle','target-arrow-color':'rgba(209,73,91,0.9)','arrow-scale':0.7}},
 {selector:'edge.hl', style:{'width':1.4,'line-color':'rgba(231,76,60,0.7)','target-arrow-shape':'triangle','target-arrow-color':'rgba(231,76,60,0.8)','arrow-scale':0.8,'z-index':99}}
];

function fsz(){ return 11 / Math.max(0.05, cy.zoom()); }
function init(){
  P95a = p95(NODES.map(function(n){return n.data.indeg;})); P95b = p95(NODES.map(function(n){return n.data.cites;}));
  NODES.forEach(function(n){ n.data.col = colorOf(n.data); n.data.sz = sizeOf(n.data); });
  cy = cytoscape({container:document.getElementById('cy'), elements:{nodes:NODES, edges:EDGES}, style:STYLE,
    layout:{name:'preset'}, minZoom:0.03, maxZoom:8, boxSelectionEnabled:false, pixelRatio:1,
    hideEdgesOnViewport:true, textureOnViewport:true});
  cy.on('tap','node',function(e){ select(e.target); });
  cy.on('tap',function(e){ if(e.target===cy) reset(); });
  cy.on('mouseover','node',function(e){ e.target.style('font-size',fsz()).addClass('hovered'); });
  cy.on('mouseout','node',function(e){ e.target.removeStyle('font-size').removeClass('hovered'); });
  cy.on('zoom',function(){ cy.nodes('.ego').style('font-size',fsz()); });
  buildStanceBoxes(); buildLegend(); applyFilters(true);
}
function restyle(){
  cy.batch(function(){ cy.nodes().forEach(function(n){ var d=n.data(); n.data('col',colorOf(d)); n.data('sz',sizeOf(d)); }); });
  buildLegend();
}
function applyFilters(fit){
  cy.batch(function(){
    cy.nodes().forEach(function(n){ var d=n.data(); if(vis[d.stance] && d.rank<=state.limit) n.removeClass('hidden'); else n.addClass('hidden'); });
    cy.edges().forEach(function(e){ if(e.source().hasClass('hidden')||e.target().hasClass('hidden')) e.addClass('hidden'); else e.removeClass('hidden'); });
  });
  var v = cy.nodes().not('.hidden');
  document.getElementById('shown').textContent = 'Showing ' + v.length + ' papers, ' + cy.edges().not('.hidden').length + ' links';
  if(state.links!=='all') applyLinks(); else document.getElementById('link-badge').textContent='';
  if(fit && v.length) cy.fit(v,40);
}
var LINKMODES = {s2c:['supportive','critical','x-s2c'], c2s:['critical','supportive','x-c2s'], c2c:['critical','critical','x-c2c']};
function applyLinks(){
  var modes = state.links==='both' ? ['s2c','c2s'] : state.links==='all3' ? ['s2c','c2s','c2c'] : [state.links];
  var inv = {}, nE=0, citing={}, cited={};
  cy.batch(function(){
    cy.edges().removeClass('x-s2c x-c2s x-c2c');
    cy.edges().forEach(function(e){
      var cs=e.target().data('stance'), ds=e.source().data('stance'), hit=null;
      modes.forEach(function(m){ var L=LINKMODES[m]; if(cs===L[0] && ds===L[1]) hit=L[2]; });
      if(hit){ e.removeClass('hidden').addClass(hit); inv[e.source().id()]=1; inv[e.target().id()]=1; nE++; citing[e.target().id()]=1; cited[e.source().id()]=1; }
      else e.addClass('hidden');
    });
    cy.nodes().forEach(function(n){
      if(n.id() in inv){ n.removeClass('hidden faded'); }
      else { n.removeClass('faded'); if(!n.hasClass('hidden')) n.addClass('faded'); }
    });
  });
  document.getElementById('link-badge').textContent = nE+' links \u00b7 '+Object.keys(citing).length+' citing papers \u2192 '+Object.keys(cited).length+' cited papers';
}
function setLinks(m){
  state.links=m; buildLegend();
  if(m==='all'){ cy.edges().removeClass('x-s2c x-c2s x-c2c'); cy.nodes().removeClass('faded'); applyFilters(false); }
  else { applyLinks(); var inv=cy.nodes().not('.faded').not('.hidden'); if(inv.length) cy.fit(inv,50); }
}
function buildStanceBoxes(){
  var h=''; STANCES.forEach(function(s){ h += '<label class="chk"><input type="checkbox" '+(vis[s]?'checked':'')+' data-s="'+s+'"> <span class="dot" style="background:'+SCOL[s]+'"></span>'+s.replace('_',' ')+' ('+(COUNTS[s]||0)+')</label>'; });
  var box=document.getElementById('stance-boxes'); box.innerHTML=h;
  box.querySelectorAll('input').forEach(function(i){ i.onchange=function(){ vis[i.dataset.s]=i.checked; applyFilters(false); }; });
}
function buildLegend(){
  var l=document.getElementById('legend'), h='<b>'+({stance:'Stance towards NOVA',year:'Publication year',stype:'Study type',basis:'Evidence used'}[state.color])+'</b>';
  if(state.color==='stance') STANCES.forEach(function(s){ h+='<div><span class="dot" style="background:'+SCOL[s]+'"></span>'+s.replace('_',' ')+'</div>'; });
  else if(state.color==='year') h+='<div><span class="dot" style="background:'+yearColor(YMIN)+'"></span>'+YMIN+' &rarr; <span class="dot" style="background:'+yearColor(YMAX)+'"></span>'+YMAX+'</div>';
  else if(state.color==='stype') Object.keys(TCOL).forEach(function(t){ h+='<div><span class="dot" style="background:'+TCOL[t]+'"></span>'+t+'</div>'; });
  else h+='<div><span class="dot" style="background:#2e86ab"></span>full text</div><div><span class="dot" style="background:#c9ced6"></span>abstract only</div>';
  if(state.links!=='all') h+='<div style="margin-top:4px"><b>Links</b></div><div><span class="dot" style="background:#8e44ad"></span>supportive \u2192 critical</div><div><span class="dot" style="background:#e67e22"></span>critical \u2192 supportive</div><div><span class="dot" style="background:#d1495b"></span>critical \u2192 critical</div>';
  h+='<div style="color:#888;margin-top:4px">Size: '+(state.size==='indeg'?'citations within this corpus':'total citations')+'</div>';
  l.innerHTML=h;
}
function esc(s){ return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];}); }
function select(n){
  reset(true); var id=parseInt(n.id()); state.ego=id;
  var nhd=n.closedNeighborhood(); n.addClass('ego'); nhd.nodes().not(n).addClass('neighbor');
  cy.elements().not(nhd).addClass('faded'); n.connectedEdges().addClass('hl').removeClass('faded');
  n.style('font-size',fsz()); document.getElementById('reset-btn').style.display='inline-block'; openPanel(n);
}
function reset(keep){
  cy.elements().removeClass('ego neighbor faded hl inf1 inf2 inf3'); cy.nodes().removeStyle('font-size');
  if(state.links!=='all') applyLinks();
  if(!keep){ state.ego=-1; document.getElementById('side-panel').style.display='none'; document.getElementById('reset-btn').style.display='none'; document.getElementById('inf-badge').textContent=''; }
}
function listHtml(arr){
  if(!arr.length) return '<ul class="cit-list"><li style="color:#aaa;cursor:default">none in corpus</li></ul>';
  arr = arr.map(function(i){return cy.getElementById(String(i));}).sort(function(a,b){return b.data('cites')-a.data('cites');});
  return '<ul class="cit-list">'+arr.slice(0,15).map(function(m){ return '<li onclick="goto('+m.id()+')">'+esc(m.data('title').slice(0,90))+' <span style="color:#999">('+m.data('year')+', <span style="color:'+(SCOL[m.data('stance')])+'">'+m.data('stance').replace('_',' ')+'</span>)</span></li>'; }).join('')+(arr.length>15?'<li style="color:#999;cursor:default">+ '+(arr.length-15)+' more</li>':'')+'</ul>';
}
function goto(i){ var n=cy.getElementById(String(i)); if(n.hasClass('hidden')) n.removeClass('hidden'); select(n); cy.animate({center:{eles:n},zoom:Math.max(cy.zoom(),0.6)},{duration:300}); }
function openPanel(n){
  var d=n.data(), p=document.getElementById('side-panel');
  document.getElementById('panel-title').textContent=d.title;
  var rows=[['Year',d.year],['Journal',esc(d.journal)],['Stance','<span class="badge" style="background:'+SCOL[d.stance]+'">'+d.stance.replace('_',' ')+'</span>'],
    ['Evidence',d.basis==='fulltext'?'full text':'abstract'],['Abstract-based stance',d.stance_abs.replace('_',' ')],
    ['Engages with critique',d.eng?'yes':'no'],['Study type',esc(d.stype)],['Country',esc(d.country)],
    ['Citations (OpenAlex)',d.cites],['Cited within corpus',d.indeg],['Cites within corpus',d.outdeg]];
  var h=rows.map(function(r){return '<div class="pr"><span class="pk">'+r[0]+'</span><span class="pv">'+r[1]+'</span></div>';}).join('');
  if(d.doi) h+='<div class="pr"><span class="pk">DOI</span><span class="pv"><a href="'+esc(d.doi)+'" target="_blank">open</a></span></div>';
  h+='<h4>Cited by (influence flows to)</h4>'+listHtml(OUT_ADJ[d.id])+'<h4>Cites (influenced by)</h4>'+listHtml(IN_ADJ[d.id]);
  document.getElementById('panel-body').innerHTML=h; p.style.display='flex';
}
function influence(){
  if(state.ego<0){ document.getElementById('inf-badge').textContent='select a paper first'; return; }
  var seen={}; seen[state.ego]=0; var frontier=[state.ego], depth=state.depth, counts=[0,0,0,0];
  for(var d=1; d<=depth; d++){ var nxt=[]; frontier.forEach(function(i){ (OUT_ADJ[i]||[]).forEach(function(j){ if(!(j in seen) && !cy.getElementById(String(j)).hasClass('hidden')){ seen[j]=d; nxt.push(j); counts[d]++; } }); }); frontier=nxt; }
  cy.elements().removeClass('inf1 inf2 inf3 faded hl neighbor');
  cy.elements().addClass('faded');
  cy.getElementById(String(state.ego)).removeClass('faded');
  Object.keys(seen).forEach(function(k){ var n=cy.getElementById(k); n.removeClass('faded'); if(seen[k]>0) n.addClass('inf'+seen[k]); });
  cy.edges().forEach(function(e){ if(!e.hasClass('hidden') && (e.source().id() in seen) && (e.target().id() in seen)) e.removeClass('faded'); });
  document.getElementById('inf-badge').textContent=counts.slice(1,depth+1).map(function(c,i){return 'step '+(i+1)+': '+c;}).join(' · ');
}
function setDepth(d){ state.depth=d; [1,2,3].forEach(function(k){document.getElementById('d'+k).classList.toggle('active',k===d);}); influence(); }

var srch=document.getElementById('srch'), dd=document.getElementById('dd');
srch.oninput=function(){
  var q=srch.value.trim().toLowerCase(); if(q.length<2){ dd.style.display='none'; return; }
  var hits=NODES.filter(function(n){return n.data.title.toLowerCase().indexOf(q)>=0 || n.data.journal.toLowerCase().indexOf(q)>=0;}).sort(function(a,b){return b.data.cites-a.data.cites;}).slice(0,12);
  dd.innerHTML=hits.map(function(n){return '<div class="dd-item" onclick="pick('+n.data.id+')">'+esc(n.data.title.slice(0,100))+'<div class="dd-sub">'+n.data.year+' · '+esc(n.data.journal)+' · '+n.data.stance.replace('_',' ')+'</div></div>';}).join('')||'<div class="dd-item">no match</div>';
  dd.style.display='block';
};
function pick(i){ dd.style.display='none'; var n=cy.getElementById(String(i)); vis[n.data('stance')]=true; buildStanceBoxes(); n.data('rank') > state.limit && (state.limit = __TOTAL__); applyFilters(false); goto(i); }
document.getElementById('zin').onclick=function(){ cy.zoom({level:cy.zoom()*1.4, renderedPosition:{x:cy.width()/2,y:cy.height()/2}}); };
document.getElementById('zout').onclick=function(){ cy.zoom({level:cy.zoom()/1.4, renderedPosition:{x:cy.width()/2,y:cy.height()/2}}); };
document.getElementById('zfit').onclick=function(){ var v=cy.nodes().not('.hidden'); if(!state.links||state.links==='all') cy.fit(v,40); else cy.fit(v.not('.faded'),50); };
document.getElementById('link-mode').onchange=function(){ setLinks(this.value); };
document.getElementById('color-by').onchange=function(){ state.color=this.value; restyle(); };
document.getElementById('size-by').onchange=function(){ state.size=this.value; restyle(); };
var sl=document.getElementById('limit');
sl.oninput=function(){ state.limit=parseInt(sl.value); applyFilters(false); };
document.getElementById('reset-btn').onclick=function(){ reset(); };
document.getElementById('close-panel').onclick=function(){ reset(); };
init();
"""

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - Stance Network</title>
<script src="__CDN__"></script>
<style>__CSS__</style></head><body>
<div id="hdr"><div><h1>__TITLE__ &mdash; Citation Network by NOVA Stance</h1>
<p>__NN__ papers &middot; __NE__ citation links &middot; __ARROW__ &middot; Generated __DATE__</p></div></div>
<div id="ctrl">
 <div class="cg"><label>Find paper / journal</label><div class="srch-wrap"><input id="srch" class="srch-in" placeholder="Type title…" autocomplete="off"><div class="dd" id="dd"></div></div></div>
 <div class="sep"></div>
 <div class="cg"><label>Colour by</label><select id="color-by"><option value="stance">Stance</option><option value="year">Year</option><option value="stype">Study type</option><option value="basis">Evidence used</option></select></div>
 <div class="cg"><label>Size by</label><select id="size-by"><option value="indeg">Citations in corpus</option><option value="cites">Total citations</option></select></div>
 <div class="sep"></div>
 <div class="cg"><label>Show stances</label><div id="stance-boxes" style="display:flex;gap:10px;flex-wrap:wrap"></div></div>
 <div class="sep"></div>
 <div class="cg"><label>Top papers by in-corpus citations</label><input id="limit" type="range" min="10" max="__TOTAL__" value="__TOTAL__" style="width:160px"></div>
 <span id="shown" style="font-size:.74rem;color:#1a5296"></span>
 <div class="sep"></div>
 <div class="cg"><label>Cross-citation links</label><div style="display:flex;gap:6px;align-items:center"><select id="link-mode"><option value="all">All links</option><option value="s2c">Supportive \u2192 critical</option><option value="c2s">Critical \u2192 supportive</option><option value="both">Both directions</option><option value="c2c">Critical \u2192 critical</option><option value="all3">All three</option></select><span id="link-badge" style="font-size:.74rem;color:#8e44ad"></span></div></div>
 <div class="sep"></div>
 <div class="cg"><label>Influence of selected (steps)</label><div style="display:flex;gap:6px;align-items:center"><div class="btn-grp"><button class="tog active" id="d1" onclick="setDepth(1)">1</button><button class="tog" id="d2" onclick="setDepth(2)">2</button><button class="tog" id="d3" onclick="setDepth(3)">3</button></div><span id="inf-badge" style="font-size:.74rem;color:#8e44ad"></span></div></div>
 <button class="reset-btn" id="reset-btn">Reset</button>
</div>
<div id="workspace"><div id="cy"></div><div id="legend"></div>
 <div id="zoom" style="position:absolute;right:10px;top:10px;display:flex;flex-direction:column;gap:4px;z-index:5"><button class="tog" id="zin" style="border-radius:5px">+</button><button class="tog" id="zout" style="border-radius:5px">&minus;</button><button class="tog" id="zfit" style="border-radius:5px;font-size:.7rem">fit</button></div>
 <div style="position:absolute;right:50px;top:12px;font-size:.68rem;color:#999;z-index:5">scroll / pinch to zoom &middot; drag to pan &middot; click a paper</div>
 <div id="side-panel"><div id="panel-hdr"><h3 id="panel-title"></h3><button id="close-panel">&times;</button></div><div id="panel-body"></div></div></div>
<script>
var NODES = __NODES__, EDGES = __EDGES__, OUT_ADJ = __OUTADJ__, IN_ADJ = __INADJ__, COUNTS = __COUNTS__;
</script>
<script>__JS__</script></body></html>
"""


def main():
    nodes, edges, out_adj, in_adj, counts, tcol = build()
    js = (JS.replace("__STANCES__", json.dumps(STANCES)).replace("__SCOL__", json.dumps(COLOURS))
          .replace("__TCOL__", json.dumps(tcol)).replace("__TOTAL__", str(len(nodes))))
    page = (HTML.replace("__JS__", js).replace("__CSS__", CSS).replace("__CDN__", CYTOSCAPE_CDN)
            .replace("__TITLE__", html.escape(CFG["title"])).replace("__ARROW__", ARROW_NOTE)
            .replace("__DATE__", date.today().isoformat()).replace("__NN__", f"{len(nodes):,}")
            .replace("__NE__", f"{len(edges):,}").replace("__TOTAL__", str(len(nodes)))
            .replace("__NODES__", json.dumps(nodes)).replace("__EDGES__", json.dumps(edges))
            .replace("__OUTADJ__", json.dumps(out_adj)).replace("__INADJ__", json.dumps(in_adj))
            .replace("__COUNTS__", json.dumps(counts)))
    out = OUT / "stance_network_cytoscape.html"
    out.write_text(page, encoding="utf-8")
    print(f"{len(nodes)} nodes, {len(edges)} edges -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
