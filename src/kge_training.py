import os
import json
import pandas as pd
from utils import train_model, cross_validate_model, get_default_hyperparams
import itertools
import torch

def train_and_evaluate_models(triples_path, output_dir, model_names=None):
    """
    Train and evaluate multiple KGE models with hyperparameter optimization.
    
    Args:
        triples_path (str): Path to the TSV file containing triples
        output_dir (str): Directory to save models and results
        model_names (list): List of model names to train (default: all models)
    """
    if model_names is None:
        model_names = ['TransE', 'TransH', 'RotatE', 'DistMult']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define hyperparameter grid
    param_grid = {
        'embedding_dim': [50, 100, 200],
        'num_negatives': [1, 3, 5],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [64, 128, 256],
        'early_stopping': [True],
        'patience': [5],
        'relative_delta': [0.002]
    }
    
    # Store results
    results = {}
    
    # Train and evaluate each model
    for model_name in model_names:
        print(f"\nTraining {model_name}...")
        
        # Get default hyperparameters for this model
        default_params = get_default_hyperparams(model_name)
        
        # Perform grid search with cross-validation
        best_score = float('-inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for params in param_combinations:
            # Update with default parameters
            params.update(default_params)
            
            print(f"Testing parameters: {params}")
            
            # Perform cross-validation
            metrics = cross_validate_model(
                triples_path=triples_path,
                model_name=model_name,
                hyperparams=params,
                n_splits=3  # Use 3-fold CV for faster training
            )
            
            # Use MRR as the optimization metric
            score = metrics.get('mean_reciprocal_rank', 0)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        # Train final model with best parameters
        print(f"\nTraining final {model_name} model with best parameters...")
        model, entity_to_id, relation_to_id, test_triples, final_metrics = train_model(
            triples_path=triples_path,
            model_name=model_name,
            hyperparams=best_params
        )
        
        # Save model and results
        model_dir = os.path.join(output_dir, f"{model_name.lower()}_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and mappings
        torch.save(model, os.path.join(model_dir, "trained_model.pkl"))
        
        # Save entity mappings
        entity_df = pd.DataFrame({
            'id': list(entity_to_id.values()),
            'entity': list(entity_to_id.keys())
        })
        entity_df.to_csv(os.path.join(model_dir, "entity_to_id.tsv.gz"), 
                        sep='\t', index=False, compression='gzip')
        
        # Save relation mappings
        relation_df = pd.DataFrame({
            'id': list(relation_to_id.values()),
            'relation': list(relation_to_id.keys())
        })
        relation_df.to_csv(os.path.join(model_dir, "relation_to_id.tsv.gz"), 
                          sep='\t', index=False, compression='gzip')
        
        # Store results
        results[model_name] = {
            'best_parameters': best_params,
            'metrics': final_metrics
        }
        
        # Save results to JSON
        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump({
                'best_parameters': best_params,
                'metrics': final_metrics
            }, f, indent=2)
    
    # Create summary table
    summary_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        summary_data.append({
            'Model': model_name,
            'MRR': metrics.get('mean_reciprocal_rank', 0),
            'Hits@1': metrics.get('hits_at_1', 0),
            'Hits@3': metrics.get('hits_at_3', 0),
            'Hits@10': metrics.get('hits_at_10', 0)
        })
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Print summary
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))

def main():
    # Set paths
    triples_path = "triples_kge.tsv"
    output_dir = "models"
    
    # Train and evaluate models
    train_and_evaluate_models(triples_path, output_dir)

if __name__ == "__main__":
    main() 