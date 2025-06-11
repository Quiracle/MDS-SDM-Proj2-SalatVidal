import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdflib import Graph
import os

def load_data():
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
    
    return {
        'papers': papers_df,
        'authors': authors_df,
        'paper_authors': paper_authors_df,
        'citations': citations_df,
        'reviews': reviews_df,
        'editions': editions_df,
        'conferences': conferences_df,
        'workshops': workshops_df,
        'journals': journals_df,
        'volumes': volumes_df
    }

def create_entity_distribution_plot(data):
    plt.figure(figsize=(10, 6))
    entities = {
        'Papers': len(data['papers']),
        'Authors': len(data['authors']),
        'Conferences': len(data['conferences']),
        'Workshops': len(data['workshops']),
        'Journals': len(data['journals']),
        'Editions': len(data['editions']),
        'Volumes': len(data['volumes']),
        'Reviews': len(data['reviews'])
    }
    
    plt.bar(entities.keys(), entities.values())
    plt.title('Distribution of Entities in the Knowledge Graph')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('doc/img/entity_distribution.pdf')
    plt.close()
    
    return entities

def create_relationship_distribution_plot(data):
    plt.figure(figsize=(10, 6))
    relationships = {
        'Paper-Author': len(data['paper_authors']),
        'Citations': len(data['citations']),
        'Reviews': len(data['reviews'])
    }
    
    plt.bar(relationships.keys(), relationships.values())
    plt.title('Distribution of Relationships in the Knowledge Graph')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('doc/img/relationship_distribution.pdf')
    plt.close()
    
    return relationships

def create_venue_distribution_plot(data):
    plt.figure(figsize=(10, 6))
    venue_types = data['papers']['venue_type'].value_counts()
    
    plt.pie(venue_types.values, labels=venue_types.index, autopct='%1.1f%%')
    plt.title('Distribution of Publication Venues')
    plt.tight_layout()
    plt.savefig('doc/img/venue_distribution.pdf')
    plt.close()
    
    return venue_types

def create_year_distribution_plot(data):
    plt.figure(figsize=(12, 6))
    years = data['papers']['year'].value_counts().sort_index()
    
    plt.plot(years.index, years.values, marker='o')
    plt.title('Distribution of Papers by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('doc/img/year_distribution.pdf')
    plt.close()
    
    return years

def create_keyword_distribution_plot(data):
    # Count keywords
    all_keywords = []
    for keywords in data['papers']['keywords'].dropna():
        all_keywords.extend(k.strip() for k in keywords.split(','))
    
    keyword_counts = pd.Series(all_keywords).value_counts().head(20)
    
    plt.figure(figsize=(12, 6))
    plt.barh(keyword_counts.index, keyword_counts.values)
    plt.title('Top 20 Keywords')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('doc/img/keyword_distribution.pdf')
    plt.close()
    
    return keyword_counts

def main():
    # Create img directory if it doesn't exist
    os.makedirs('doc/img', exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Generate statistics and plots
    entity_stats = create_entity_distribution_plot(data)
    relationship_stats = create_relationship_distribution_plot(data)
    venue_stats = create_venue_distribution_plot(data)
    year_stats = create_year_distribution_plot(data)
    keyword_stats = create_keyword_distribution_plot(data)
    
    # Save statistics to a text file
    with open('doc/img/abox_statistics.txt', 'w') as f:
        f.write("ABOX Statistics\n")
        f.write("===============\n\n")
        
        f.write("Entity Counts:\n")
        for entity, count in entity_stats.items():
            f.write(f"{entity}: {count}\n")
        
        f.write("\nRelationship Counts:\n")
        for rel, count in relationship_stats.items():
            f.write(f"{rel}: {count}\n")
        
        f.write("\nVenue Distribution:\n")
        for venue, count in venue_stats.items():
            f.write(f"{venue}: {count}\n")
        
        f.write("\nYear Distribution:\n")
        for year, count in year_stats.items():
            f.write(f"{year}: {count}\n")
        
        f.write("\nTop 20 Keywords:\n")
        for keyword, count in keyword_stats.items():
            f.write(f"{keyword}: {count}\n")

if __name__ == '__main__':
    main() 