#!/usr/bin/env python3
"""
visualize.py — Generate an interactive HTML knowledge graph from KB
entities, facts, and community assignments.

Usage:
    python scripts/visualize.py --kb-root /path/to/kb
    python scripts/visualize.py --kb-root ~/kb --domain clinical --output my_graph.html
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from kb_root import add_kb_root_arg, resolve_kb_root, load_config

COMMUNITY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


# ── Data fetching ────────────────────────────────────────────────────────────

def _scroll_all(qdrant, collection, filt=None):
    items = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for r in results:
            items.append(r.payload)
        if offset is None:
            break
    return items


def fetch_graph_data(qdrant, cfg, domain=None, max_nodes=5000):
    """Fetch entities and non-superseded facts from Qdrant."""
    entities_col = cfg.get("collections_extra", {}).get("entities", "entities")
    facts_col = cfg.get("collections_extra", {}).get("facts", "facts")

    filt = None
    if domain:
        filt = Filter(must=[FieldCondition(key="domain",
                                           match=MatchValue(value=domain))])

    entities = _scroll_all(qdrant, entities_col, filt)
    facts = _scroll_all(qdrant, facts_col, filt)
    # Filter superseded facts
    facts = [f for f in facts if not f.get("superseded_by")]

    # Trim to max_nodes by keeping most-connected entities
    if len(entities) > max_nodes:
        degree = Counter()
        for f in facts:
            degree[f.get("subject_entity_id", "")] += 1
            degree[f.get("object_entity_id", "")] += 1
        entities.sort(key=lambda e: degree.get(e["entity_id"], 0), reverse=True)
        entities = entities[:max_nodes]
        keep = {e["entity_id"] for e in entities}
        facts = [f for f in facts
                 if f.get("subject_entity_id") in keep
                 and f.get("object_entity_id") in keep]

    return entities, facts


# ── vis.js data building ─────────────────────────────────────────────────────

def build_vis_data(entities, facts):
    """Convert entities and facts to vis.js nodes and edges."""
    # Compute degree per entity
    degree = Counter()
    for f in facts:
        degree[f.get("subject_entity_id", "")] += 1
        degree[f.get("object_entity_id", "")] += 1

    max_deg = max(degree.values(), default=1)

    nodes = []
    for e in entities:
        eid = e["entity_id"]
        comm = e.get("community_id", 0)
        color = COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)]
        deg = degree.get(eid, 0)
        size = 10 + 30 * (deg / max_deg) if max_deg > 0 else 10

        summary = e.get("summary", "")
        tooltip = (f"<b>{_esc(e.get('name', ''))}</b><br>"
                   f"Type: {_esc(e.get('type', ''))}<br>"
                   f"Community: {comm}<br>"
                   f"Connections: {deg}<br>"
                   f"{_esc(summary[:200])}")

        show_label = deg >= max(1, max_deg * 0.15)
        nodes.append({
            "id": eid,
            "label": e.get("name", eid[:8]),
            "title": tooltip,
            "color": {"background": color, "border": color,
                      "highlight": {"background": "#ffffff", "border": color}},
            "size": round(size, 1),
            "font": {"size": 12 if show_label else 0, "color": "#e0e0e0"},
            "community": comm,
            "entityType": e.get("type", ""),
        })

    edges = []
    seen = set()
    for f in facts:
        sid = f.get("subject_entity_id", "")
        oid = f.get("object_entity_id", "")
        if not sid or not oid or sid == oid:
            continue
        edge_key = (min(sid, oid), max(sid, oid))
        if edge_key in seen:
            continue
        seen.add(edge_key)

        conf = f.get("confidence", "")
        is_established = conf == "established"
        fact_text = _esc(f.get("fact", ""))

        edges.append({
            "from": sid,
            "to": oid,
            "label": f.get("relation_type", ""),
            "title": fact_text,
            "dashes": not is_established,
            "width": 2 if is_established else 1,
            "color": {"opacity": 0.7 if is_established else 0.35},
            "arrows": "to",
        })

    return {"nodes": nodes, "edges": edges}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ── HTML rendering ───────────────────────────────────────────────────────────

def render_html(vis_data, entities, title="KB Knowledge Graph"):
    """Generate a self-contained HTML page with vis.js interactive graph."""
    # Build community legend data
    comm_counts = Counter()
    for n in vis_data["nodes"]:
        comm_counts[n["community"]] += 1

    legend_items = []
    for comm_id in sorted(comm_counts):
        color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        legend_items.append({
            "id": comm_id,
            "color": color,
            "count": comm_counts[comm_id],
        })

    nodes_json = json.dumps(vis_data["nodes"])
    edges_json = json.dumps(vis_data["edges"])
    legend_json = json.dumps(legend_items)

    return HTML_TEMPLATE.format(
        title=_esc(title),
        node_count=len(vis_data["nodes"]),
        edge_count=len(vis_data["edges"]),
        community_count=len(comm_counts),
        nodes_json=nodes_json,
        edges_json=edges_json,
        legend_json=legend_json,
    )


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; display: flex; height: 100vh; overflow: hidden; }}
#sidebar {{ width: 300px; background: #1a1a2e; padding: 16px; overflow-y: auto; border-right: 1px solid #2a2a3e; display: flex; flex-direction: column; gap: 16px; }}
#sidebar h2 {{ font-size: 14px; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }}
#search {{ width: 100%; padding: 8px 12px; background: #0f0f1a; border: 1px solid #3a3a5e; border-radius: 4px; color: #e0e0e0; font-size: 14px; outline: none; }}
#search:focus {{ border-color: #4e79a7; }}
#search-results {{ max-height: 200px; overflow-y: auto; }}
.search-item {{ padding: 6px 8px; cursor: pointer; border-radius: 3px; font-size: 13px; }}
.search-item:hover {{ background: #2a2a4e; }}
#info-panel {{ display: none; }}
#info-panel h3 {{ font-size: 16px; color: #ffffff; margin-bottom: 8px; }}
.info-row {{ font-size: 13px; margin: 4px 0; color: #aaaacc; }}
.info-label {{ color: #6666aa; }}
#neighbors {{ max-height: 150px; overflow-y: auto; margin-top: 8px; }}
.neighbor-item {{ font-size: 12px; padding: 3px 0; color: #8888bb; }}
#legend {{ display: flex; flex-direction: column; gap: 4px; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 4px; border-radius: 3px; font-size: 13px; }}
.legend-item:hover {{ background: #2a2a4e; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
.legend-item.hidden {{ opacity: 0.3; }}
#stats {{ font-size: 12px; color: #6666aa; }}
#graph {{ flex: 1; }}
</style>
</head>
<body>

<div id="sidebar">
  <div>
    <h2>Search</h2>
    <input type="text" id="search" placeholder="Search entities...">
    <div id="search-results"></div>
  </div>

  <div id="info-panel">
    <h2>Details</h2>
    <h3 id="info-name"></h3>
    <div class="info-row"><span class="info-label">Type:</span> <span id="info-type"></span></div>
    <div class="info-row"><span class="info-label">Community:</span> <span id="info-comm"></span></div>
    <div class="info-row"><span class="info-label">Connections:</span> <span id="info-deg"></span></div>
    <div id="info-summary" class="info-row" style="margin-top:8px;"></div>
    <h2 style="margin-top:12px;">Neighbors</h2>
    <div id="neighbors"></div>
  </div>

  <div>
    <h2>Communities</h2>
    <div id="legend"></div>
  </div>

  <div id="stats">
    {node_count} nodes &middot; {edge_count} edges &middot; {community_count} communities
  </div>
</div>

<div id="graph"></div>

<script>
const nodesData = {nodes_json};
const edgesData = {edges_json};
const legendData = {legend_json};

const nodes = new vis.DataSet(nodesData);
const edges = new vis.DataSet(edgesData);

const container = document.getElementById('graph');
const data = {{ nodes, edges }};
const options = {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -60,
      centralGravity: 0.005,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.4,
      avoidOverlap: 0.8
    }},
    stabilization: {{ iterations: 200, fit: true }}
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
    navigationButtons: false,
    keyboard: false
  }},
  nodes: {{ shape: 'dot', borderWidth: 1.5 }},
  edges: {{
    smooth: {{ type: 'continuous', roundness: 0.2 }},
    selectionWidth: 3,
    font: {{ size: 0 }}
  }}
}};

const network = new vis.Network(container, data, options);

// Freeze physics after stabilization
network.on('stabilized', function() {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});

// Search
const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');

searchInput.addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  searchResults.innerHTML = '';
  if (q.length < 2) return;

  const matches = nodesData.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  matches.forEach(n => {{
    const div = document.createElement('div');
    div.className = 'search-item';
    div.textContent = n.label;
    div.onclick = () => {{
      network.selectNodes([n.id]);
      network.focus(n.id, {{ scale: 1.5, animation: true }});
      showInfo(n.id);
    }};
    searchResults.appendChild(div);
  }});
}});

// Click to inspect
network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    showInfo(params.nodes[0]);
  }} else {{
    document.getElementById('info-panel').style.display = 'none';
  }}
}});

function showInfo(nodeId) {{
  const node = nodesData.find(n => n.id === nodeId);
  if (!node) return;

  document.getElementById('info-panel').style.display = 'block';
  document.getElementById('info-name').textContent = node.label;
  document.getElementById('info-type').textContent = node.entityType || '—';
  document.getElementById('info-comm').textContent = node.community;

  const connected = network.getConnectedNodes(nodeId);
  document.getElementById('info-deg').textContent = connected.length;

  // Extract summary from tooltip
  const tmp = document.createElement('div');
  tmp.innerHTML = node.title || '';
  const text = tmp.textContent;
  const summaryMatch = text.match(/Connections: \\d+(.+)/);
  document.getElementById('info-summary').textContent = summaryMatch ? summaryMatch[1].trim() : '';

  const neighborsDiv = document.getElementById('neighbors');
  neighborsDiv.innerHTML = '';
  connected.forEach(nid => {{
    const neighbor = nodesData.find(n => n.id === nid);
    if (neighbor) {{
      const div = document.createElement('div');
      div.className = 'neighbor-item';
      div.textContent = neighbor.label;
      div.style.cursor = 'pointer';
      div.onclick = () => {{
        network.selectNodes([nid]);
        network.focus(nid, {{ scale: 1.5, animation: true }});
        showInfo(nid);
      }};
      neighborsDiv.appendChild(div);
    }}
  }});
}}

// Legend toggle
const legendDiv = document.getElementById('legend');
const hiddenCommunities = new Set();

legendData.forEach(item => {{
  const div = document.createElement('div');
  div.className = 'legend-item';
  div.innerHTML = `<span class="legend-dot" style="background:${{item.color}}"></span>Community ${{item.id}} (${{item.count}})`;
  div.onclick = () => {{
    if (hiddenCommunities.has(item.id)) {{
      hiddenCommunities.delete(item.id);
      div.classList.remove('hidden');
    }} else {{
      hiddenCommunities.add(item.id);
      div.classList.add('hidden');
    }}
    applyFilter();
  }};
  legendDiv.appendChild(div);
}});

function applyFilter() {{
  const updates = nodesData.map(n => ({{
    id: n.id,
    hidden: hiddenCommunities.has(n.community)
  }}));
  nodes.update(updates);
}}
</script>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML knowledge graph visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_kb_root_arg(parser)
    parser.add_argument("--domain", help="Filter to a specific domain")
    parser.add_argument("--output", default="graph.html",
                        help="Output filename (default: graph.html)")
    parser.add_argument("--max-nodes", type=int, default=5000,
                        help="Maximum nodes to render (default: 5000)")
    args = parser.parse_args()

    kb_root = resolve_kb_root(args)
    cfg = load_config(kb_root)
    vis_cfg = cfg.get("visualization", {})
    max_nodes = args.max_nodes or vis_cfg.get("max_nodes", 5000)

    qdrant = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

    print("Fetching graph data...")
    entities, facts = fetch_graph_data(qdrant, cfg, args.domain, max_nodes)
    print(f"  {len(entities)} entities, {len(facts)} facts")

    if not entities:
        print("No entities found. Run entity_extract.py first.")
        sys.exit(0)

    print("Building visualization data...")
    vis_data = build_vis_data(entities, facts)

    print("Rendering HTML...")
    output_dir = kb_root / vis_cfg.get("output_dir", "exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    html = render_html(vis_data, entities, title=f"KB Graph — {kb_root.name}")
    output_path.write_text(html, encoding="utf-8")
    print(f"  Written to {output_path}")
    print(f"  Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
