import torch
import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.evaluation import RankBasedEvaluator
import os
from utils import load_model, get_device, train_transe_model

def find_most_likely_cited_paper(model, entity_to_id, relation_to_id, paper_id):
    """
    Find the most likely paper to be cited by the given paper.
    
    Args:
        model: Trained TransE model
        entity_to_id (dict): Mapping from entity names to IDs
        relation_to_id (dict): Mapping from relation names to IDs
        paper_id (str): ID of the paper to find citations for
    
    Returns:
        tuple: (predicted_paper_id, embedding_vector)
    """
    # Get device
    device = next(model.parameters()).device
    
    # Get the embedding for the input paper
    paper_idx = entity_to_id[paper_id]
    paper_embedding = model.entity_representations[0](indices=torch.tensor([paper_idx], device=device))
    
    # Get the citation relation embedding
    cites_idx = relation_to_id['http://example.org/publication/cites']
    cites_embedding = model.relation_representations[0](indices=torch.tensor([cites_idx], device=device))
    
    # Calculate the expected embedding of the cited paper
    # In TransE: head + relation ≈ tail
    expected_cited_embedding = paper_embedding + cites_embedding
    
    # Get all entity embeddings
    all_embeddings = model.entity_representations[0](indices=None)
    
    # Calculate distances to all entities
    distances = torch.norm(all_embeddings - expected_cited_embedding, dim=1)
    
    # Exclude the input paper
    distances[paper_idx] = float('inf')
    
    # Filter out non-paper entities
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    for idx, entity_id in id_to_entity.items():
        if not entity_id.startswith('http://example.org/publication/Paper'):
            distances[idx] = float('inf')
    
    # Get the index of the closest entity
    closest_idx = torch.argmin(distances).item()
    predicted_entity_id = id_to_entity[closest_idx]
    
    return predicted_entity_id, expected_cited_embedding

def find_most_likely_author(model, entity_to_id, relation_to_id, paper_embedding):
    """
    Find the most likely author of a paper based on its embedding.
    
    Args:
        model: Trained TransE model
        entity_to_id (dict): Mapping from entity names to IDs
        relation_to_id (dict): Mapping from relation names to IDs
        paper_embedding (torch.Tensor): Embedding vector of the paper
    
    Returns:
        str: ID of the most likely entity
    """
    # Get device
    device = next(model.parameters()).device
    
    # Get the author relation embedding
    author_idx = relation_to_id['http://example.org/publication/hasAuthor']
    author_embedding = model.relation_representations[0](indices=torch.tensor([author_idx], device=device))
    
    # Calculate the expected author embedding
    # In TransE: paper - relation ≈ author
    expected_author_embedding = paper_embedding - author_embedding
    
    # Get all entity embeddings
    all_embeddings = model.entity_representations[0](indices=None)
    
    # Calculate distances to all entities
    distances = torch.norm(all_embeddings - expected_author_embedding, dim=1)
    
    # Filter out non-author entities
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    for idx, entity_id in id_to_entity.items():
        if not entity_id.startswith('http://example.org/publication/Author'):
            distances[idx] = float('inf')
    
    # Get the index of the closest entity
    closest_idx = torch.argmin(distances).item()
    predicted_entity_id = id_to_entity[closest_idx]
    
    return predicted_entity_id

def main():
    # Set paths
    triples_path = "triples_kge.tsv"
    model_dir = "models/transe_model"
    
    # Train model with train/test split
    model, entity_to_id, relation_to_id, test_triples = train_transe_model(
        triples_path, 
        model_dir,
        test_size=0.2,  # Use 20% of data for testing
        random_state=42
    )
    
    print(f"Model saved to {model_dir}")
    
    # Get a paper from the test set
    test_paper = test_triples.triples[0][0]  # Get the first head entity from test triples
    print(f"\nUsing test paper: {test_paper}")
    
    # Find most likely cited paper
    cited_paper, paper_embedding = find_most_likely_cited_paper(
        model, entity_to_id, relation_to_id, test_paper
    )
    print(f"Most likely cited paper: {cited_paper}")
    
    # Find most likely author
    author = find_most_likely_author(model, entity_to_id, relation_to_id, paper_embedding)
    print(f"Most likely author: {author}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, os.path.join(model_dir, "trained_model.pkl"))
    
    # Demonstrate loading the model
    loaded_model, loaded_entity_to_id, loaded_relation_to_id = load_model(model_dir)
    print("\nModel loaded successfully from disk")

if __name__ == "__main__":
    main() 