from rdflib import Graph

Q = """
PREFIX lom: <http://example.org/lom#>
SELECT ?uri ?title ?difficulty WHERE {
  ?uri a lom:LearningObject ;
       lom:general_title ?title ;
       lom:educational_difficulty ?difficulty ;
       lom:topic ?topic .
  FILTER(?topic = "Nature")
}
LIMIT 10
"""

if __name__ == "__main__":
    g = Graph().parse("rdf/corpus.ttl", format="turtle")
    for row in g.query(Q):
        print(row)
