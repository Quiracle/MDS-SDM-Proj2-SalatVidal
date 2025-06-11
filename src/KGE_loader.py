from rdflib import Graph, URIRef, Literal
import csv

# Load the RDF graph from Turtle file
g = Graph()
g.parse("./publication_abox.ttl", format="turtle")  # Replace with your actual .ttl file name

# Lists to hold different types of triples
kge_triples = []
exploitation_triples = []

# Define predicates to exclude from KGE (these involve literals)
literal_predicates = {
    "http://example.org/publication/hasName",
    "http://example.org/publication/hasAbstract",
    "http://example.org/publication/hasReviewText",
    "http://example.org/publication/hasStartDate",
    "http://example.org/publication/hasEndDate",
    "http://example.org/publication/hasYear",
    "http://example.org/publication/hasPublicationDate",
    "http://example.org/publication/hasVolumeNumber",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment"
}

# Iterate through triples
for s, p, o in g:
    s_str, p_str = str(s), str(p)
    
    if isinstance(o, URIRef) and p_str not in literal_predicates:
        o_str = str(o)
        kge_triples.append((s_str, p_str, o_str))
    elif isinstance(o, Literal):
        o_str = str(o)
        exploitation_triples.append((s_str, p_str, o_str))

# Write KGE triples
with open("./triples_kge.tsv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(kge_triples)

# Write exploitation triples
with open("./triples_exploitation.tsv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(exploitation_triples)

print(f"Exported {len(kge_triples)} KGE triples to triples_kge.tsv")
print(f"Exported {len(exploitation_triples)} literal triples to triples_exploitation.tsv")
