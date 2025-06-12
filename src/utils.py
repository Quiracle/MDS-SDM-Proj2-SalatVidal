import torch
import pandas as pd
import os
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.losses import MarginRankingLoss, SoftplusLoss
from pykeen.regularizers import LpRegularizer
import numpy as np
from sklearn.model_selection import KFold

def get_device():
    """
    Get the appropriate device (CUDA if available, CPU otherwise).
    
    Returns:
        str: 'cuda' or 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_dir):
    """
    Load a saved model and its mappings.
    
    Args:
        model_dir (str): Directory containing the saved model
    
    Returns:
        tuple: (model, entity_to_id, relation_to_id)
    """
    # Load the model with weights_only=False to handle all model components
    model = torch.load(os.path.join(model_dir, "trained_model.pkl"), weights_only=False)
    
    # Load entity mappings
    entity_df = pd.read_csv(os.path.join(model_dir, "training_triples/entity_to_id.tsv.gz"), 
                           sep='\t', compression='gzip', header=0)
    entity_to_id = dict(zip(entity_df.iloc[:, 1], entity_df.iloc[:, 0]))
    
    # Load relation mappings
    relation_df = pd.read_csv(os.path.join(model_dir, "training_triples/relation_to_id.tsv.gz"), 
                             sep='\t', compression='gzip', header=0)
    relation_to_id = dict(zip(relation_df.iloc[:, 1], relation_df.iloc[:, 0]))
    
    return model, entity_to_id, relation_to_id

def train_model(triples_path, model_name, hyperparams, random_state=42):
    """
    Train a KGE model with the given hyperparameters.
    
    Args:
        triples_path (str): Path to the TSV file containing triples
        model_name (str): Name of the model
        hyperparams (dict): Dictionary of hyperparameters
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (trained_model, entity_to_id, relation_to_id, test_triples, metrics)
    """
    # Load triples
    triples = pd.read_csv(triples_path, sep='\t', header=None, 
                         names=['head', 'relation', 'tail'])
    
    # Create triples factory
    tf = TriplesFactory.from_labeled_triples(
        triples=triples[['head', 'relation', 'tail']].values,
        create_inverse_triples=False
    )
    
    # Split into training and testing sets
    training_tf, testing_tf = tf.split()
    
    # Configure model parameters
    model_kwargs = {
        'loss': MarginRankingLoss(margin=hyperparams.get('margin', 1.0))
    }
    
    # Add model-specific parameters
    if hyperparams.get('model_kwargs'):
        model_kwargs.update(hyperparams['model_kwargs'])
    
    # Configure training parameters
    training_kwargs = {
        'num_epochs': hyperparams.get('num_epochs', 100),
        'batch_size': hyperparams.get('batch_size', 128)
    }
    
    # Train the model
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
        use_tqdm=True,
        # Configure evaluation with basic metrics
        evaluator='rankbased',
        evaluator_kwargs=dict(
            filtered=True,
            metrics=['mrr', 'hitsat']
        )
    )
    
    return result.model, tf.entity_to_id, tf.relation_to_id, testing_tf, result.metric_results.to_dict()

def _extract_numeric_value(metric_value):
    """
    Helper function to extract numeric value from a metric, handling nested dictionaries.
    
    Args:
        metric_value: The metric value, which can be a number, string, or dictionary
    
    Returns:
        float: The extracted numeric value
    """
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    elif isinstance(metric_value, str):
        return float(metric_value)
    elif isinstance(metric_value, dict):
        # Try to get arithmetic_mean first
        if 'arithmetic_mean' in metric_value:
            return _extract_numeric_value(metric_value['arithmetic_mean'])
        # Otherwise, recursively process the first value
        return _extract_numeric_value(next(iter(metric_value.values())))
    else:
        raise ValueError(f"Unexpected metric value type: {type(metric_value)}")

def cross_validate_model(triples_path, model_name, hyperparams, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation for a KGE model.
    
    Args:
        triples_path (str): Path to the TSV file containing triples
        model_name (str): Name of the model
        hyperparams (dict): Dictionary of hyperparameters
        n_splits (int): Number of folds for cross-validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing average metrics across folds
    """
    # Load triples
    triples = pd.read_csv(triples_path, sep='\t', header=None, 
                         names=['head', 'relation', 'tail'])
    
    # Create triples factory
    tf = TriplesFactory.from_labeled_triples(
        triples=triples[['head', 'relation', 'tail']].values,
        create_inverse_triples=False
    )
    
    # Initialize metrics storage
    metrics = []
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(kf.split(tf.triples)):
        # Create training and testing triples factories
        training_tf = TriplesFactory.from_labeled_triples(
            triples=tf.triples[train_idx],
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
            create_inverse_triples=False
        )
        testing_tf = TriplesFactory.from_labeled_triples(
            triples=tf.triples[test_idx],
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
            create_inverse_triples=False
        )
        
        # Train model for this fold
        result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model=model_name,
            model_kwargs=hyperparams.get('model_kwargs', {}),
            training_kwargs=hyperparams.get('training_kwargs', {}),
            optimizer='Adam',
            optimizer_kwargs=dict(lr=hyperparams.get('learning_rate', 0.001)),
            device=get_device(),
            random_seed=random_state + fold
        )
        
        # Store metrics
        metrics.append(result.metric_results.to_dict())
    
    # Calculate average metrics
    avg_metrics = {}
    # Get all metric names from the first fold
    metric_names = metrics[0].keys()
    
    for metric_name in metric_names:
        # Extract the numeric values for this metric from each fold
        values = []
        for fold_metrics in metrics:
            try:
                value = _extract_numeric_value(fold_metrics[metric_name])
                values.append(value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not extract numeric value for metric {metric_name}: {e}")
                continue
        
        if values:  # Only calculate average if we have valid values
            avg_metrics[metric_name] = float(np.mean(values))
    
    return avg_metrics

def get_default_hyperparams(model_name):
    """
    Get default hyperparameters for a specific model.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary of default hyperparameters
    """
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
    else:
        raise ValueError(f"Unknown model name: {model_name}") 

def train_transe_model(triples_path, model_dir, test_size=0.2, random_state=42):
    """
    Train a TransE model on the knowledge graph.
    
    Args:
        triples_path (str): Path to the TSV file containing triples
        model_dir (str): Directory to save the model
        test_size (float): Proportion of triples to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (model, entity_to_id, relation_to_id, test_triples)
    """
    # Load triples
    triples = pd.read_csv(triples_path, sep='\t', header=None, 
                         names=['head', 'relation', 'tail'])
    
    # Create triples factory
    tf = TriplesFactory.from_labeled_triples(
        triples=triples[['head', 'relation', 'tail']].values,
        create_inverse_triples=False
    )
    
    # Split into training and testing
    training_tf, testing_tf = tf.split([1.0 - test_size, test_size], random_state=random_state)
    
    # Get device
    device = get_device()
    
    # Train model
    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        model='TransE',
        model_kwargs=dict(embedding_dim=50),
        training_kwargs=dict(
            num_epochs=20,
            batch_size=256
        ),
        optimizer='Adam',
        optimizer_kwargs=dict(lr=0.01),
        device=device,
        random_seed=random_state
    )
    
    # Save model and mappings
    os.makedirs(model_dir, exist_ok=True)
    result.save_to_directory(model_dir)
    
    # Get entity and relation mappings
    entity_to_id = training_tf.entity_to_id
    relation_to_id = training_tf.relation_to_id
    
    return result.model, entity_to_id, relation_to_id, testing_tf 