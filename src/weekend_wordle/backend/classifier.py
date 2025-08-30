import os
import numpy as np
import spacy
from wordfreq import word_frequency
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from typing import Callable

DEFAULT_CONFIG = {
    'use_vectors': True,
    'spy_rate': 0.15,
    'max_iterations': 1000,
    'convergence_tolerance': 1e-2,
    'random_seed': None,
    'evaluation_threshold': 0.07,
    'explicit_features': {
        'frequency': 1.0, 
        'is_regular_plural': 1.0, 
        'is_irregular_plural': 1.0,
        'is_past_tense': 1.0, 
        'is_adjective': 1.0,
        'is_proper_noun': 1.0, 
        'is_gerund': 1.0, 
        'vowel_count': 1.0, 
        'has_double_letter': 1.0
    }
}

def load_spacy_model(model_name="en_core_web_lg") -> spacy.language.Language:
    """Loads a spaCy model, prompting the user to download it if not found."""
    try:
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError:
        print(f"spaCy model '{model_name}' not found. Attempting to download.")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            return nlp
        except SystemExit:
            print(f"Failed to automatically download model.")
            print("Please download it manually using.")
            print(f"uv add https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl")
            exit()

def compute_word_features(words: np.ndarray[str], nlp):
    """Efficiently gets a combined feature DataFrame for a list of words."""
    word_series = pd.Series(words)
    docs = list(nlp.pipe(words.tolist()))
    
    vectors = [doc[0].vector for doc in docs]
    tags = [doc[0].tag_ for doc in docs]
    tags_series = pd.Series(tags, index=word_series.index)

    features = pd.DataFrame(index=word_series.index)
    features['is_regular_plural'] = ((tags_series == 'NNS') & (word_series.str.endswith('s'))).astype(int)
    features['is_irregular_plural'] = ((tags_series == 'NNS') & (~word_series.str.endswith('s'))).astype(int)
    features['frequency'] = [np.log1p(word_frequency(word, 'en')) for word in words]
    features['is_past_tense'] = (tags_series == 'VBD').astype(int)
    features['is_adjective'] = (tags_series == 'JJ').astype(int)
    features['is_proper_noun'] = (tags_series == 'NNP').astype(int)
    features['is_gerund'] = (tags_series == 'VBG').astype(int)
    features['vowel_count'] = word_series.str.count('[aeiou]').astype(int)
    features['has_double_letter'] = (word_series.str.findall(r'(.)\1').str.len() > 0).astype(int)
    features['vector'] = vectors
    
    return features

def get_word_features(
    all_words: np.ndarray, 
    save_file: str = 'data/word_features.pkl', 
    recompute: bool = False, 
    save: bool = True,
    model_name: str = "en_core_web_lg"
    ) -> pd.DataFrame:
    """Loads pre-computed word features from a file or recomputes them if needed."""
    if not recompute and os.path.exists(save_file):
        print(f"Loading pre-computed features from '{save_file}'...")
        return pd.read_pickle(save_file)
    
    print("Recomputing features for all words...")
    nlp = load_spacy_model(model_name)
    features_df = compute_word_features(all_words, nlp)
    features_df['word'] = all_words # Add word column for easy merging
    
    if save and save_file:
        print(f"Saving new features to '{save_file}'...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        features_df.to_pickle(save_file)
        
    return features_df

def evaluate_training_performance(
    probabilities: np.ndarray[float],
    all_words: np.ndarray[str],
    positive_words: np.ndarray[str],
    threshold: float = 0.5
    ):
    """
    Evaluates the final model's performance on the entire dataset using the prediction function.
    """
    print("\n--- Evaluating Model Performance on All Known Data ---")
    
    results_df = pd.DataFrame({'word': all_words, 'probability': probabilities})
    
    predicted_positives = set(results_df[results_df['probability'] >= threshold]['word'])
    known_positives = set(positive_words)
    
    true_positives = predicted_positives.intersection(known_positives)
    false_negatives = known_positives.difference(predicted_positives)
    
    recall = len(true_positives) / len(known_positives) if len(known_positives) > 0 else 0
    known_answer_density = len(true_positives) / len(predicted_positives) if len(predicted_positives) > 0 else 0
    
    print(f"\nEvaluation with probability threshold >= {threshold:.2f}")
    print("-" * 45)
    print(f"{'Total Postives Predicted:':<30} {len(predicted_positives):>5}")
    print(f"{'Total Known Positives:':<30} {len(known_positives):>5}")
    print(f"{'Identified Positives (TPs):':<30} {len(true_positives):>5}")
    print(f"{'False Negatives:':<30} {len(false_negatives):>5}")
    print("-" * 45)
    print(f"{'Recall on Positives:':<30} {recall:>5.2%}")
    print(f"{'Known Answer Density:':<30} {known_answer_density:>5.2%}")
    print("-" * 45)
    
    print("\n--- False Negatives (Words the Model Missed) ---")
    fn_df = results_df[results_df['word'].isin(false_negatives)]
    formatted_fns = [f"{row.word} ({row.probability:.1%})" for row in fn_df.sort_values('word').itertuples()]
    
    if not formatted_fns:
        print("None")
    else:
        words_per_row = 5
        for i in range(0, len(formatted_fns), words_per_row):
            chunk = formatted_fns[i:i + words_per_row]
            print(", ".join(chunk))

def train_classifier(
    feature_df: pd.DataFrame, 
    positive_words: np.ndarray,
    all_words: np.ndarray, 
    config: dict = DEFAULT_CONFIG, 
    ) -> dict:
    """
    Trains the PU classifier from scratch and evaluates its performance.
    Returns a dictionary containing the trained model and scaler.
    """
    print("\n--- Starting New Model Training ---")
    
    # Prepare DataFrame for training
    train_df = feature_df.copy()
    train_df['type'] = np.where(train_df['word'].isin(positive_words), 'P', 'U')
    
    # --- Spy Selection ---
    p_df = train_df[train_df['type'] == 'P']
    u_df = train_df[train_df['type'] == 'U']
    spies_df = p_df.sample(frac=config['spy_rate'], random_state=config['random_seed'])
    reliable_positives_df = p_df.drop(spies_df.index)

    enabled_features = list(config['explicit_features'].keys())

    # --- Prepare feature matrices ---
    def get_matrix(df: pd.DataFrame) -> np.ndarray[np.float64]:
        explicit = df[enabled_features].values.astype(np.float64)
        if config['use_vectors']:
            embedding = np.array(df['vector'].tolist())
            return np.concatenate([embedding, explicit], axis=1)
        return explicit

    X_reliable_positives = get_matrix(reliable_positives_df)
    X_unlabeled = get_matrix(u_df)
    X_spies = get_matrix(spies_df)

    # --- Iterative Spy EM Process ---
    unlabeled_weights = np.full(len(X_unlabeled), 0.5)
    model, scaler = None, None
    
    feature_slice = slice(-len(enabled_features), None) if config['use_vectors'] else slice(None)
    weights_vector = np.array([config['explicit_features'][feat] for feat in enabled_features])

    for i in range(config['max_iterations']):
        print(f"--- Training Iteration {i+1}/{config['max_iterations']} ---")
        X_train = np.concatenate([X_reliable_positives, X_unlabeled])
        y_train = np.array([1] * len(X_reliable_positives) + [0] * len(X_unlabeled))
        sample_weights = np.concatenate([np.ones(len(X_reliable_positives)), unlabeled_weights])

        scaler = StandardScaler().fit(X_train[:, feature_slice])
        X_train_scaled = X_train.copy()
        X_train_scaled[:, feature_slice] = scaler.transform(X_train[:, feature_slice])
        X_train_scaled[:, feature_slice] *= weights_vector

        model = LogisticRegression(solver='liblinear', random_state=config['random_seed'])
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # E-Step
        X_spies_scaled = X_spies.copy()
        X_spies_scaled[:, feature_slice] = scaler.transform(X_spies[:, feature_slice])
        X_spies_scaled[:, feature_slice] *= weights_vector
        spy_probs = model.predict_proba(X_spies_scaled)[:, 1]
        c = np.mean(spy_probs)
        print(f"Average spy probability (c): {c:.4f}")

        X_unlabeled_scaled = X_unlabeled.copy()
        X_unlabeled_scaled[:, feature_slice] = scaler.transform(X_unlabeled[:, feature_slice])
        X_unlabeled_scaled[:, feature_slice] *= weights_vector
        unlabeled_probs = model.predict_proba(X_unlabeled_scaled)[:, 1]
        
        new_unlabeled_weights = np.clip(unlabeled_probs / c, 0, 1)
        weight_change = np.sum(np.abs(new_unlabeled_weights - unlabeled_weights))
        print(f"Total change in weights: {weight_change:.4f}")
        
        if weight_change < config['convergence_tolerance']:
            print("Convergence reached.")
            break
        unlabeled_weights = new_unlabeled_weights
    
    # --- Post-Training Evaluation ---
    if model and scaler:
        eval_df = train_df.copy()
        X_inf = get_matrix(eval_df)
        X_inf[:, feature_slice] = scaler.transform(X_inf[:, feature_slice])
        X_inf[:, feature_slice] *= weights_vector
        probabilities =  model.predict_proba(X_inf)[:, 1]
        
        evaluate_training_performance(
            probabilities=probabilities,
            all_words=all_words,
            positive_words=positive_words,
            threshold=config.get('evaluation_threshold', 0.07)
        )

    return {'model': model, 'scaler': scaler}

def load_classifier(
    feature_df: pd.DataFrame, 
    save_file: str = 'data/wordle_classifier.pkl',
    retrain: bool = False,
    save: bool = True,
    positive_words: np.ndarray = None,
    all_words: np.ndarray = None,
    config: dict = DEFAULT_CONFIG
    ) -> Callable:
    """
    Loads a pre-trained classifier or retrains one if needed.
    Returns a self-contained prediction function.
    """
    if not retrain and os.path.exists(save_file):
        print(f"Loading pre-trained model artifacts from '{save_file}'...")
        with open(save_file, 'rb') as f:
            artifacts = pickle.load(f)
    else:
        if config is None:
            raise ValueError("A configuration dictionary must be provided for retraining.")
        print("Training new model...")
        trained_components = train_classifier(feature_df, positive_words, all_words, config)
        artifacts = {**trained_components, 'config': config}
        
        if save and save_file:
            print(f"Saving new model artifacts to '{save_file}'...")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'wb') as f:
                pickle.dump(artifacts, f)

    print("\nPre-computing probabilities for all words for fast lookup...")
    model: LogisticRegression = artifacts['model']
    scaler: StandardScaler = artifacts['scaler']
    model_config = artifacts['config']
    
    # Use the full feature_df to build the matrix for all words
    indexed_feature_df = feature_df.set_index('word')

    enabled_features = list(model_config['explicit_features'].keys())

    def get_matrix(df: pd.DataFrame):
        explicit = df[enabled_features].values.astype(np.float64)
        if model_config['use_vectors']:
            embedding = np.array(df['vector'].tolist())
            return np.concatenate([embedding, explicit], axis=1)
        return explicit

    X_all = get_matrix(indexed_feature_df)
    
    feature_slice = slice(-len(enabled_features), None) if model_config['use_vectors'] else slice(None)
    weights_vector = np.array([model_config['explicit_features'][feat] for feat in enabled_features])
    
    X_all[:, feature_slice] = scaler.transform(X_all[:, feature_slice])
    X_all[:, feature_slice] *= weights_vector
    
    all_probabilities = model.predict_proba(X_all)[:, 1]
    
    # Create the fast lookup Series
    probability_lookup = pd.Series(all_probabilities, index=indexed_feature_df.index)

    def predict_word_probabilities(words: str|list|np.ndarray) -> float|np.ndarray:
        """
        Takes a single word (str) or a list/array of words and returns their probabilities
        using a pre-computed lookup table for maximum speed.
        """
        is_single_word = isinstance(words, str)
        input_words = [words] if is_single_word else words
        
        # Use .get() for safe lookup with a default value of 0.0 for unknown words
        probs = [probability_lookup.get(word, 0.0) for word in input_words]
        
        # Return a single float or a numpy array based on the input type
        return probs[0] if is_single_word else np.array(probs)

    return predict_word_probabilities

def filter_words_by_probability(
    prediction_function: Callable,
    words_to_filter: np.ndarray,
    threshold: float = 0.07
    ) -> np.ndarray:
    """
    Uses a prediction function to filter a list of words based on a probability threshold.
    """
    probabilities = prediction_function(words_to_filter)
    return words_to_filter[probabilities >= threshold]