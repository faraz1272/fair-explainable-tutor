# src/provenance/prov.py
from __future__ import annotations
import argparse
from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS, FOAF, XSD

PROV = Namespace("http://www.w3.org/ns/prov#")

def add_provenance(graph_path: Path, activity_id: str, agent_name: str,
                   used_path: str | None, generated_path: str | None) -> int:
    g = Graph()
    if graph_path.exists():
        g.parse(graph_path, format="turtle")

    # Create URIs
    activity = URIRef(f"https://example.org/activity/{activity_id}")
    agent = URIRef("https://example.org/agent/author")  # simple canonical URI
    used_ent = URIRef(f"https://example.org/entity/{Path(used_path).as_posix()}") if used_path else None
    gen_ent  = URIRef(f"https://example.org/entity/{Path(generated_path).as_posix()}") if generated_path else None

    before = len(g)

    # Bind prefixes (nice readable TTL)
    g.bind("prov", PROV)
    g.bind("dct", DCTERMS)
    g.bind("foaf", FOAF)

    # Agent
    g.add((agent, RDF.type, PROV.Agent))
    g.add((agent, FOAF.name, Literal(agent_name)))

    # Activity
    g.add((activity, RDF.type, PROV.Activity))
    g.add((activity, DCTERMS.creator, agent))

    # Used / Generated entities
    if used_ent:
        g.add((used_ent, RDF.type, PROV.Entity))
        g.add((activity, PROV.used, used_ent))

    if gen_ent:
        g.add((gen_ent, RDF.type, PROV.Entity))
        g.add((gen_ent, PROV.wasGeneratedBy, activity))

    # Save back to TTL
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(graph_path, format="turtle")

    return len(g) - before

def main():
    p = argparse.ArgumentParser(description="Append simple PROV-O provenance to an RDF TTL graph.")
    p.add_argument("--graph", required=True, help="Path to the RDF Turtle file to read/write.")
    p.add_argument("--activity-id", required=True, help="Unique id for this run/activity.")
    p.add_argument("--agent", required=True, help="Human-readable agent name.")
    p.add_argument("--used", help="Path to an input file/entity the activity used.")
    p.add_argument("--generated", help="Path to an output file/entity the activity generated.")
    args = p.parse_args()

    added = add_provenance(
        graph_path=Path(args.graph),
        activity_id=args.activity_id,
        agent_name=args.agent,
        used_path=args.used,
        generated_path=args.generated,
    )
    print(f"[prov] Added {added} triple(s) to {args.graph}")

if __name__ == "__main__":
    main()
