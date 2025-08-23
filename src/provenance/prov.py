# src/provenance/prov.py
"""Minimal PROV-O helpers to append provenance triples to a TTL graph.

This module provides two helpers:
- `add_provenance`: *appending* provenance for a single used/generated path.
- `add_provenance_multi`: *appending* provenance for multiple inputs/outputs.

Both functions:
- *creating* (or *parsing*) an RDF graph in Turtle format,
- *declaring* an activity and an agent (with FOAF name),
- *recording* timestamps when supplied,
- *linking* each input via `prov:used` and each output via `prov:wasGeneratedBy`,
- *serializing* the updated graph back to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph, Namespace, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS, FOAF, XSD

# Namespaces
PROV = Namespace("http://www.w3.org/ns/prov#")


def _uri_for_file(p: str | Path) -> URIRef:
    """Return a stable, readable URIRef for a local path.

    We *avoid* `file://` absolute paths to keep the TTL portable across machines
    and CI runs. Instead, we *encode* the relative POSIX path inside a stable
    HTTP-like namespace.
    """
    p = Path(p)
    # building a stable pseudo-URL using the POSIX path
    return URIRef(f"https://example.org/entity/{p.as_posix()}")


def add_provenance(
    graph_path: Path,
    activity_id: str,
    agent_name: str,
    used_path: str | None,
    generated_path: str | None,
    started: Optional[datetime] = None,
    ended: Optional[datetime] = None,
) -> int:
    """Backward-compatible single-file variant that delegates to `add_provenance_multi`.

    Args:
        graph_path: TTL file to *read*/*write*.
        activity_id: Unique identifier for this run/activity.
        agent_name: Human-readable agent (author) name.
        used_path: Optional input file path.
        generated_path: Optional output file path.
        started: Optional UTC datetime when the activity started.
        ended: Optional UTC datetime when the activity ended.

    Returns:
        The number of triples *added* to the graph.
    """
    used = [used_path] if used_path else []
    gen = [generated_path] if generated_path else []
    return add_provenance_multi(graph_path, activity_id, agent_name, used, gen, started, ended)


def add_provenance_multi(
    graph_path: Path,
    activity_id: str,
    agent_name: str,
    used_paths: Iterable[str | Path] | None,
    generated_paths: Iterable[str | Path] | None,
    started: Optional[datetime] = None,
    ended: Optional[datetime] = None,
) -> int:
    """Append PROV-O triples for one run/activity.

    This function *creates* (or *parses*) a TTL graph, *declares* an activity and
    agent, *records* optional timestamps, *links* each input as `prov:used`,
    and *links* each output as `prov:wasGeneratedBy`.

    Args:
        graph_path: TTL file to *read*/*write*.
        activity_id: Unique identifier for this run/activity.
        agent_name: Human-readable agent (author) name.
        used_paths: Iterable of input file paths.
        generated_paths: Iterable of output file paths.
        started: Optional UTC datetime when the activity started.
        ended: Optional UTC datetime when the activity ended.

    Returns:
        The number of triples *added* to the graph.
    """
    g = Graph()
    if graph_path.exists():
        # parsing existing graph when present
        g.parse(graph_path, format="turtle")

    before = len(g)

    # binding well-known prefixes for readability
    g.bind("prov", PROV)
    g.bind("dct", DCTERMS)
    g.bind("foaf", FOAF)

    # creating core nodes
    activity = URIRef(f"https://example.org/activity/{activity_id}")
    agent = URIRef("https://example.org/agent/author")

    # declaring agent with FOAF name
    g.add((agent, RDF.type, PROV.Agent))
    g.add((agent, FOAF.name, Literal(agent_name)))

    # declaring activity and creator link
    g.add((activity, RDF.type, PROV.Activity))
    g.add((activity, DCTERMS.creator, agent))

    # recording optional timestamps
    if started:
        g.add((activity, PROV.startedAtTime, Literal(started.isoformat(), datatype=XSD.dateTime)))
    if ended:
        g.add((activity, PROV.endedAtTime, Literal(ended.isoformat(), datatype=XSD.dateTime)))

    # linking inputs (prov:used)
    for p in (used_paths or []):
        ent = _uri_for_file(p)
        g.add((ent, RDF.type, PROV.Entity))
        g.add((activity, PROV.used, ent))

    # linking outputs (prov:wasGeneratedBy)
    for p in (generated_paths or []):
        ent = _uri_for_file(p)
        g.add((ent, RDF.type, PROV.Entity))
        g.add((ent, PROV.wasGeneratedBy, activity))

    # serializing the updated graph
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(graph_path, format="turtle")

    return len(g) - before


def main() -> None:
    """CLI entrypoint for appending simple PROV-O provenance to a TTL graph."""
    ap = argparse.ArgumentParser(description="Append simple PROV-O provenance to an RDF TTL graph.")
    ap.add_argument("--graph", required=True, help="Path to the RDF Turtle file to read/write.")
    ap.add_argument("--activity-id", required=True, help="Unique id for this run/activity.")
    ap.add_argument("--agent", required=True, help="Human-readable agent name.")
    ap.add_argument("--used", action="append", help="Path to an input file/entity (repeatable).", default=[])
    ap.add_argument("--generated", action="append", help="Path to an output file/entity (repeatable).", default=[])
    args = ap.parse_args()

    added = add_provenance_multi(
        graph_path=Path(args.graph),
        activity_id=args.activity_id,
        agent_name=args.agent,
        used_paths=args.used,
        generated_paths=args.generated,
        started=datetime.now(timezone.utc),
        ended=datetime.now(timezone.utc),
    )
    print(f"[prov] Added {added} triple(s) to {args.graph}")


if __name__ == "__main__":
    main()