import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import hashlib
from datetime import datetime

# Create namespace
PUB = Namespace("http://example.org/publication/")

def create_uri(base, identifier):
    """Create a URI by hashing the identifier"""
    if pd.isna(identifier):
        raise ValueError(f"Cannot create URI for NaN identifier in {base}")
    if isinstance(identifier, float):
        identifier = str(int(identifier))
    hash_object = hashlib.md5(str(identifier).encode())
    return URIRef(f"{base}{hash_object.hexdigest()}")

def create_abox():
    g = Graph()
    g.bind("pub", PUB)
    
    # Load CSV files
    papers_df = pd.read_csv('data/paper.csv')
    authors_df = pd.read_csv('data/researcher.csv')
    paper_authors_df = pd.read_csv('data/paper_authors.csv')
    citations_df = pd.read_csv('data/citations.csv')
    reviews_df = pd.read_csv('data/review.csv')
    editions_df = pd.read_csv('data/edition.csv')
    conferences_df = pd.read_csv('data/conference.csv')
    workshops_df = pd.read_csv('data/workshop.csv')
    journals_df = pd.read_csv('data/journal.csv')
    volumes_df = pd.read_csv('data/volume.csv')
    
    # Create Topic instances from unique keywords
    all_keywords = set()
    for keywords in papers_df['keywords'].dropna():
        all_keywords.update(k.strip() for k in keywords.split(','))
    
    topics = {}
    for keyword in all_keywords:
        topic_uri = create_uri(PUB.Topic, keyword)
        topics[keyword] = topic_uri
        g.add((topic_uri, RDFS.label, Literal(keyword)))
    
    # Create Venue instances
    venues = {}
    for venue in editions_df['venue'].dropna().unique():
        venue_uri = create_uri(PUB.Venue, venue)
        venues[venue] = venue_uri
        g.add((venue_uri, RDFS.label, Literal(venue)))
    
    # Create Author instances
    authors = {}
    for _, row in authors_df.dropna(subset=['author_id']).iterrows():
        author_uri = create_uri(PUB.Author, row['author_id'])
        authors[row['author_id']] = author_uri
        g.add((author_uri, PUB.hasName, Literal(row['name'])))
        if pd.notna(row['email']):
            g.add((author_uri, PUB.hasEmail, Literal(row['email'])))
    
    # Create Conference and Workshop instances
    events = {}
    for _, row in conferences_df.dropna(subset=['name']).iterrows():
        event_uri = create_uri(PUB.Conference, row['name'])
        events[row['name']] = event_uri
        g.add((event_uri, RDFS.label, Literal(row['name'])))
    
    for _, row in workshops_df.dropna(subset=['name']).iterrows():
        event_uri = create_uri(PUB.Workshop, row['name'])
        events[row['name']] = event_uri
        g.add((event_uri, RDFS.label, Literal(row['name'])))
    
    # Create Edition instances and their Proceedings
    editions = {}
    proceedings = {}
    for _, row in editions_df.dropna(subset=['edition_id', 'venue']).iterrows():
        edition_uri = create_uri(PUB.Edition, row['edition_id'])
        editions[row['edition_id']] = edition_uri
        if pd.notna(row['year']):
            g.add((edition_uri, PUB.hasYear, Literal(int(row['year']), datatype=XSD.gYear)))
        if row['venue'] in venues:
            g.add((edition_uri, PUB.hasVenue, venues[row['venue']]))
        if pd.notna(row['number']):
            g.add((edition_uri, PUB.hasNumber, Literal(int(row['number']), datatype=XSD.integer)))
        # Create a Proceeding for each Edition
        proceeding_uri = create_uri(PUB.Proceeding, row['edition_id'])
        proceedings[row['edition_id']] = proceeding_uri
        g.add((edition_uri, PUB.hasProceedings, proceeding_uri))
    
    # Create Journal instances
    journals = {}
    for _, row in journals_df.dropna(subset=['name']).iterrows():
        journal_uri = create_uri(PUB.Journal, row['name'])
        journals[row['name']] = journal_uri
        g.add((journal_uri, RDFS.label, Literal(row['name'])))
    
    # Create Volume instances
    volumes = {}
    for _, row in volumes_df.dropna(subset=['journal_name', 'year', 'number']).iterrows():
        volume_uri = create_uri(PUB.Volume, f"{row['journal_name']}_{row['year']}_{row['number']}")
        volumes[f"{row['journal_name']}_{row['year']}_{row['number']}"] = volume_uri
        g.add((volume_uri, PUB.hasYear, Literal(int(row['year']), datatype=XSD.gYear)))
        g.add((volume_uri, PUB.hasVolumeNumber, Literal(int(row['number']), datatype=XSD.integer)))
        if row['journal_name'] in journals:
            g.add((volume_uri, PUB.isVolumeOf, journals[row['journal_name']]))
            g.add((journals[row['journal_name']], PUB.hasVolume, volume_uri))
    
    # Create Paper instances and their relationships
    for _, row in papers_df.dropna(subset=['paper_id']).iterrows():
        paper_uri = create_uri(PUB.Paper, row['paper_id'])
        g.add((paper_uri, RDFS.label, Literal(row['title'])))
        if pd.notna(row['abstract']):
            g.add((paper_uri, PUB.hasAbstract, Literal(row['abstract'])))
        if pd.notna(row['year']):
            g.add((paper_uri, PUB.hasYear, Literal(int(row['year']), datatype=XSD.gYear)))
        
        # Add keywords as topics
        if pd.notna(row['keywords']):
            for keyword in row['keywords'].split(','):
                keyword = keyword.strip()
                if keyword in topics:
                    g.add((paper_uri, PUB.hasKeyword, topics[keyword]))
        
        # Add publication venue
        if pd.notna(row['venue']) and pd.notna(row['venue_type']):
            if row['venue_type'] == 'conference' or row['venue_type'] == 'workshop':
                # Find the corresponding edition
                matching_editions = editions_df[editions_df['venue'] == row['venue']]
                if not matching_editions.empty:
                    edition = matching_editions.iloc[0]
                    if edition['edition_id'] in proceedings:
                        proceeding_uri = proceedings[edition['edition_id']]
                        g.add((paper_uri, PUB.isPublishedIn, proceeding_uri))
            elif row['venue_type'] == 'journal':
                # Find the corresponding volume
                matching_volumes = volumes_df[volumes_df['journal_name'] == row['venue']]
                if not matching_volumes.empty:
                    volume = matching_volumes.iloc[0]
                    volume_key = f"{row['venue']}_{volume['year']}_{volume['number']}"
                    if volume_key in volumes:
                        volume_uri = volumes[volume_key]
                        g.add((paper_uri, PUB.isPublishedIn, volume_uri))
    
    # Add author relationships
    for _, row in paper_authors_df.dropna(subset=['paper_id', 'author_id']).iterrows():
        if row['paper_id'] in papers_df['paper_id'].values and row['author_id'] in authors:
            paper_uri = create_uri(PUB.Paper, row['paper_id'])
            author_uri = authors[row['author_id']]
            g.add((paper_uri, PUB.hasAuthor, author_uri))
            if row['corresponding']:
                g.add((paper_uri, PUB.hasCorrespondingAuthor, author_uri))
    
    # Add citation relationships
    for _, row in citations_df.dropna(subset=['citing_paper_id', 'cited_paper_id']).iterrows():
        if (row['citing_paper_id'] in papers_df['paper_id'].values and 
            row['cited_paper_id'] in papers_df['paper_id'].values):
            citing_paper = create_uri(PUB.Paper, row['citing_paper_id'])
            cited_paper = create_uri(PUB.Paper, row['cited_paper_id'])
            g.add((citing_paper, PUB.cites, cited_paper))
    
    # Add review relationships
    for _, row in reviews_df.dropna(subset=['paper_id', 'reviewer_id']).iterrows():
        if row['paper_id'] in papers_df['paper_id'].values and row['reviewer_id'] in authors:
            paper_uri = create_uri(PUB.Paper, row['paper_id'])
            reviewer_uri = authors[row['reviewer_id']]
            review_uri = create_uri(PUB.Review, f"{row['paper_id']}_{row['reviewer_id']}")
            # g.add((review_uri, RDF.type, PUB.Review))
            g.add((review_uri, PUB.isAssignedBy, reviewer_uri))
            if pd.notna(row['review_text']):
                g.add((review_uri, PUB.hasReviewText, Literal(row['review_text'])))
            g.add((paper_uri, PUB.hasReview, review_uri))
    
    # Relate Conferences and Workshops to their Editions by matching names
    for _, row in editions_df.dropna(subset=['edition_id', 'venue']).iterrows():
        edition_uri = editions[row['edition_id']]
        venue_name = row['venue']
        # Conference
        if venue_name in events:
            g.add((events[venue_name], PUB.hasEdition, edition_uri))
    
    return g

if __name__ == '__main__':
    g = create_abox()
    g.serialize('publication_abox.ttl', format='turtle')