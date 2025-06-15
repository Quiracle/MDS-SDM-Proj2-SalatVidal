from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL

# Create a new graph
g = Graph()

# Define namespaces
PUB = Namespace("http://example.org/publication/")
g.bind("pub", PUB)
g.bind("owl", OWL)

# Define classes
classes = {
    "Paper": "A research paper",
    "Author": "A person who writes papers",
    "Event": "An academic event where research is presented",
    "Conference": "A well-established research forum",
    "Workshop": "A forum for exploring new trends",
    "Edition": "A specific instance of a conference or workshop",
    "Journal": "A periodical publication",
    "Volume": "A collection of papers in a journal",
    "Proceedings": "A collection of papers from a conference/workshop edition",
    "Review": "An evaluation of a paper",
    "Reviewer": "A scientist who reviews papers",
    "Topic": "A subject area of a paper",
    "Venue": "A location where an edition takes place"
}

# Define properties
properties = {
    # Paper properties
    "hasAbstract": "The abstract of a paper",
    "hasKeyword": "A keyword describing the paper's topic",
    "cites": "A paper that is cited by another paper",
    "isPublishedIn": "The venue where a paper is published (either proceedings or volume)",
    "hasAuthor": "An author of the paper",
    "hasCorrespondingAuthor": "The corresponding author of the paper",
    "hasReview": "A review of the paper",
    
    # Author properties
    "writes": "A paper written by an author",
    "hasName": "The name of the author",
    
    # Conference/Workshop properties
    "hasEdition": "An edition of a conference or workshop",
    
    # Edition properties
    "hasVenue": "The venue where the edition takes place",
    "hasStartDate": "The start date of the edition",
    "hasEndDate": "The end date of the edition",
    "hasYear": "The year of the edition",
    "hasProceedings": "The proceedings of this edition",
    
    # Journal properties
    "hasVolume": "A volume of the journal",
    
    # Volume properties
    "hasPublicationDate": "The publication date of the volume",
    "hasVolumeNumber": "The number of the volume",
    
    # Review properties
    "isAssignedBy": "The person who assigned the review",
    "hasReviewText": "The text content of the review"
}

# Add classes to the graph
for class_name, description in classes.items():
    class_uri = PUB[class_name]
    g.add((class_uri, RDF.type, RDFS.Class))
    g.add((class_uri, RDFS.label, Literal(class_name)))
    g.add((class_uri, RDFS.comment, Literal(description)))

# Add properties to the graph
for prop_name, description in properties.items():
    prop_uri = PUB[prop_name]
    g.add((prop_uri, RDF.type, RDF.Property))
    g.add((prop_uri, RDFS.label, Literal(prop_name)))
    g.add((prop_uri, RDFS.comment, Literal(description)))

# Add domain and range constraints
# Paper properties
g.add((PUB.hasAbstract, RDFS.domain, PUB.Paper))
g.add((PUB.hasAbstract, RDFS.range, XSD.string))

g.add((PUB.hasKeyword, RDFS.domain, PUB.Paper))
g.add((PUB.hasKeyword, RDFS.range, PUB.Topic))

g.add((PUB.cites, RDFS.domain, PUB.Paper))
g.add((PUB.cites, RDFS.range, PUB.Paper))

g.add((PUB.isPublishedIn, RDFS.domain, PUB.Paper))
g.add((PUB.isPublishedIn, RDFS.range, PUB.Proceedings))
g.add((PUB.isPublishedIn, RDFS.range, PUB.Volume))

g.add((PUB.hasAuthor, RDFS.domain, PUB.Paper))
g.add((PUB.hasAuthor, RDFS.range, PUB.Author))

g.add((PUB.hasCorrespondingAuthor, RDFS.domain, PUB.Paper))
g.add((PUB.hasCorrespondingAuthor, RDFS.range, PUB.Author))

# Author properties
g.add((PUB.hasName, RDFS.domain, PUB.Author))
g.add((PUB.hasName, RDFS.range, XSD.string))

# Conference/Workshop properties
g.add((PUB.hasEdition, RDFS.domain, PUB.Event))
g.add((PUB.hasEdition, RDFS.range, PUB.Edition))

# Edition properties
g.add((PUB.hasVenue, RDFS.domain, PUB.Edition))
g.add((PUB.hasVenue, RDFS.range, PUB.Venue))

g.add((PUB.hasStartDate, RDFS.domain, PUB.Edition))
g.add((PUB.hasStartDate, RDFS.range, XSD.date))

g.add((PUB.hasEndDate, RDFS.domain, PUB.Edition))
g.add((PUB.hasEndDate, RDFS.range, XSD.date))

g.add((PUB.hasYear, RDFS.domain, PUB.Edition))
g.add((PUB.hasYear, RDFS.range, XSD.gYear))

g.add((PUB.hasProceedings, RDFS.domain, PUB.Edition))
g.add((PUB.hasProceedings, RDFS.range, PUB.Proceedings))

# Journal properties
g.add((PUB.hasVolume, RDFS.domain, PUB.Journal))
g.add((PUB.hasVolume, RDFS.range, PUB.Volume))

# Volume properties
g.add((PUB.hasPublicationDate, RDFS.domain, PUB.Volume))
g.add((PUB.hasPublicationDate, RDFS.range, XSD.date))

g.add((PUB.hasVolumeNumber, RDFS.domain, PUB.Volume))
g.add((PUB.hasVolumeNumber, RDFS.range, XSD.integer))

# Review properties
g.add((PUB.isAssignedBy, RDFS.domain, PUB.Review))
g.add((PUB.isAssignedBy, RDFS.range, PUB.Reviewer))

g.add((PUB.hasReviewText, RDFS.domain, PUB.Review))
g.add((PUB.hasReviewText, RDFS.range, XSD.string))

# Add inverse properties
g.add((PUB.isPublishedIn, OWL.inverseOf, PUB.containsPaper))
g.add((PUB.hasEdition, OWL.inverseOf, PUB.isEditionOf))
g.add((PUB.hasVolume, OWL.inverseOf, PUB.isVolumeOf))

# Add subclasses
g.add((PUB.Conference, RDFS.subClassOf, PUB.Event))
g.add((PUB.Workshop, RDFS.subClassOf, PUB.Event))
g.add((PUB.Reviewer, RDFS.subClassOf, PUB.Author))

# Add subproperties
g.add((PUB.hasCorrespondingAuthor, RDFS.subPropertyOf, PUB.hasAuthor))

# Save the TBOX
g.serialize(destination="publication_tbox.rdf", format="xml")