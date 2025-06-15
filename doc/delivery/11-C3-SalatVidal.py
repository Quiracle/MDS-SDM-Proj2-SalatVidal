import os
import json
import pandas as pd
import torch
import itertools

from utils import (
    train_model,
    cross_validate_model,
    get_default_hyperparams
)

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
    
    os.makedirs(output_dir, exist_ok=True)
    
    param_grid = {
        'embedding_dim': [50, 100, 200],
        'num_negatives': [1, 3, 5],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [64, 128, 256],
        'early_stopping': [True],
        'patience': [5],
        'relative_delta': [0.002],
        'num_epochs': [100]
    }
    
    results = {}
    
    for model_name in model_names:
        print(f"\n🚀 Training {model_name}...")
        
        default_params = get_default_hyperparams(model_name)
        best_score = float('-inf')
        best_params = None
        
        param_combinations = [dict(zip(param_grid.keys(), v))
                              for v in itertools.product(*param_grid.values())]

        for i, params in enumerate(param_combinations, 1):
            params.update(default_params)
            print(f"\n🔁 [{i}/{len(param_combinations)}] Testing parameters: {params}")
            
            metrics = cross_validate_model(
                triples_path=triples_path,
                model_name=model_name,
                hyperparams=params,
                n_splits=3
            )

            print("📊 Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            score = metrics.get('mean_reciprocal_rank', 0)
            if score > best_score:
                best_score = score
                best_params = params

        print(f"\n🏁 Training final {model_name} model with best parameters...")
        model, entity_to_id, relation_to_id, test_triples, final_metrics = train_model(
            triples_path=triples_path,
            model_name=model_name,
            hyperparams=best_params
        )
        
        model_dir = os.path.join(output_dir, f"{model_name.lower()}_model")
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(model, os.path.join(model_dir, "trained_model.pkl"))
        
        entity_df = pd.DataFrame({
            'id': list(entity_to_id.values()),
            'entity': list(entity_to_id.keys())
        })
        entity_df.to_csv(os.path.join(model_dir, "entity_to_id.tsv.gz"), sep='\t', index=False, compression='gzip')
        
        relation_df = pd.DataFrame({
            'id': list(relation_to_id.values()),
            'relation': list(relation_to_id.keys())
        })
        relation_df.to_csv(os.path.join(model_dir, "relation_to_id.tsv.gz"), sep='\t', index=False, compression='gzip')
        
        results[model_name] = {
            'best_parameters': best_params,
            'metrics': final_metrics
        }

        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump(results[model_name], f, indent=2)

    summary_data = []
    for model_name, result in results.items():
        m = result['metrics']
        summary_data.append({
            'Model': model_name,
            'MRR': m.get('mean_reciprocal_rank', 0),
            'Hits@1': m.get('hits_at_1', 0),
            'Hits@3': m.get('hits_at_3', 0),
            'Hits@10': m.get('hits_at_10', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    print("\n📊 Model Comparison Summary:")
    print(summary_df.to_string(index=False))

def main():
    triples_path = "triples_kge.tsv"
    output_dir = "models"
    train_and_evaluate_models(triples_path, output_dir)

if __name__ == "__main__":
    main()
