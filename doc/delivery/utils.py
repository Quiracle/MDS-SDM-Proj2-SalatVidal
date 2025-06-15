import torch
import pandas as pd
import os
import logging
import warnings
import numpy as np
from sklearn.model_selection import KFold
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss
import hashlib

# Suppress PyKEEN and torch logs
warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_dir):
    """Load saved model and its mappings."""
    model = torch.load(os.path.join(model_dir, "trained_model.pkl"), weights_only=False)
    entity_df = pd.read_csv(os.path.join(model_dir, "entity_to_id.tsv.gz"), sep='\t', compression='gzip')
    relation_df = pd.read_csv(os.path.join(model_dir, "relation_to_id.tsv.gz"), sep='\t', compression='gzip')
    entity_to_id = dict(zip(entity_df['entity'], entity_df['id']))
    relation_to_id = dict(zip(relation_df['relation'], relation_df['id']))
    return model, entity_to_id, relation_to_id

def train_model(triples_path, model_name, hyperparams, random_state=42):
    """Train KGE model with best hyperparameters and return cleaned metrics."""

    def extract_final_metrics(metrics):
        realistic = metrics.get('both', {}).get('realistic', {})
        return {
            'mean_reciprocal_rank': realistic.get('inverse_harmonic_mean_rank', 0.0),
            'hits_at_1': realistic.get('hits_at_1', 0.0),
            'hits_at_3': realistic.get('hits_at_3', 0.0),
            'hits_at_10': realistic.get('hits_at_10', 0.0)
        }

    triples = pd.read_csv(triples_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    tf = TriplesFactory.from_labeled_triples(triples.values, create_inverse_triples=True)
    training_tf, testing_tf = tf.split()

    model_kwargs = {'loss': MarginRankingLoss(margin=hyperparams.get('margin', 1.0))}
    if hyperparams.get('model_kwargs'):
        model_kwargs.update(hyperparams['model_kwargs'])

    training_kwargs = {
        'num_epochs': hyperparams.get('num_epochs', 100),
        'batch_size': hyperparams.get('batch_size', 128)
    }

    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        model=model_name,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        optimizer='Adam',
        optimizer_kwargs=dict(lr=hyperparams.get('learning_rate', 0.001)),
        negative_sampler='basic',
        negative_sampler_kwargs=dict(num_negs_per_pos=hyperparams.get('num_negatives', 1)),
        device=get_device(),
        random_seed=random_state,
        use_tqdm=False,
        evaluator='rankbased',
        evaluator_kwargs=dict(filtered=True, metrics=['mean_reciprocal_rank', 'hits_at_k']),
        evaluation_kwargs=dict(use_tqdm=False)
    )

    return result.model, tf.entity_to_id, tf.relation_to_id, testing_tf, extract_final_metrics(result.metric_results.to_dict())

def cross_validate_model(triples_path, model_name, hyperparams, n_splits=3, random_state=42):
    """Cross-validate a KGE model and return averaged clean metrics."""

    def extract_fold_metrics(metrics):
        realistic = metrics.get('both', {}).get('realistic', {})
        return {
            'mean_reciprocal_rank': realistic.get('inverse_harmonic_mean_rank', 0.0),
            'hits_at_1': realistic.get('hits_at_1', 0.0),
            'hits_at_3': realistic.get('hits_at_3', 0.0),
            'hits_at_10': realistic.get('hits_at_10', 0.0)
        }

    triples = pd.read_csv(triples_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    tf = TriplesFactory.from_labeled_triples(triples.values, create_inverse_triples=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_mrr, all_h1, all_h3, all_h10 = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(tf.triples)):
        print(f"\n🔁 Fold {fold + 1}/{n_splits}")

        training_tf = TriplesFactory.from_labeled_triples(
            triples=tf.triples[train_idx],
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
            create_inverse_triples=True
        )
        testing_tf = TriplesFactory.from_labeled_triples(
            triples=tf.triples[test_idx],
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
            create_inverse_triples=True
        )

        result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model=model_name,
            model_kwargs=hyperparams.get('model_kwargs', {}),
            training_kwargs=hyperparams.get('training_kwargs', {}),
            optimizer='Adam',
            optimizer_kwargs=dict(lr=hyperparams.get('learning_rate', 0.001)),
            device=get_device(),
            random_seed=random_state + fold,
            evaluator='rankbased',
            evaluator_kwargs=dict(filtered=True, metrics=['mean_reciprocal_rank', 'hits_at_k']),
            evaluation_kwargs=dict(use_tqdm=False),
            use_tqdm=False
        )

        metrics = extract_fold_metrics(result.metric_results.to_dict())

        print("📈 Fold Summary:")
        print(f"  MRR:     {metrics['mean_reciprocal_rank']:.4f}")
        print(f"  Hits@1:  {metrics['hits_at_1']:.4f}")
        print(f"  Hits@3:  {metrics['hits_at_3']:.4f}")
        print(f"  Hits@10: {metrics['hits_at_10']:.4f}")

        all_mrr.append(metrics['mean_reciprocal_rank'])
        all_h1.append(metrics['hits_at_1'])
        all_h3.append(metrics['hits_at_3'])
        all_h10.append(metrics['hits_at_10'])

    return {
        'mean_reciprocal_rank': float(np.mean(all_mrr)),
        'hits_at_1': float(np.mean(all_h1)),
        'hits_at_3': float(np.mean(all_h3)),
        'hits_at_10': float(np.mean(all_h10))
    }

def get_default_hyperparams(model_name):
    """Return default hyperparameters for a given KGE model."""
    base_params = {
        'embedding_dim': 100,
        'num_epochs': 100,
        'batch_size': 256,
        'num_negatives': 1,
        'learning_rate': 0.001,
        'early_stopping': True,
        'patience': 5,
        'relative_delta': 0.002
    }

    if model_name == 'TransE':
        return base_params
    elif model_name == 'TransH':
        params = base_params.copy()
        params['scoring_fct_norm'] = 2
        return params
    elif model_name == 'RotatE':
        params = base_params.copy()
        params['scoring_fct_norm'] = 2
        return params
    elif model_name == 'DistMult':
        params = base_params.copy()
        params['regularizer_weight'] = 0.1
        return params
    elif model_name == 'ComplEx':
        params = base_params.copy()
        params['regularizer_weight'] = 0.1
        return params
    else:
        raise ValueError(f"Unknown model name: {model_name}")


import hashlib

def hash_id(identifier):
    """
    Hashes an ID using MD5 after safely converting to a clean string.

    Handles common float representations like '123.0' and ensures consistent output.
    """
    try:
        # Convert float-like input (e.g., 145642373.0) to clean int string ('145642373')
        clean_str = str(int(float(identifier)))
    except Exception:
        # Fallback to safe string strip
        clean_str = str(identifier).strip()

    return hashlib.md5(clean_str.encode()).hexdigest()