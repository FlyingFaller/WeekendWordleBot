import os
import numpy as np
import spacy
from wordfreq import word_frequency
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from typing import Callable

from weekend_wordle.backend.messenger import UIMessenger, ConsoleMessenger
from weekend_wordle.backend.helpers import get_messenger

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

def load_spacy_model(model_name="en_core_web_lg",
                     messenger: UIMessenger = None) -> spacy.language.Language:
    """Loads a spaCy model, prompting the user to download it if not found."""
    messenger = get_messenger(messenger)
    with messenger.task(f"Loading spaCy model '{model_name}'"):
        try:
            nlp = spacy.load(model_name)
            messenger.task_log("Model loaded successfully.", level="INFO")
            return nlp
        except OSError:
            messenger.task_log("Model not found. Attempting to download...", level="INFO")
            try:
                spacy.cli.download(model_name)
                # Reload after download
                nlp = spacy.load(model_name)
                messenger.task_log("Model downloaded and loaded successfully.", level="INFO")
                return nlp
            except SystemExit:
                messenger.task_log("Automatic download failed.", level="ERROR")
                messenger.task_log("Please download it manually, for example:", level="INFO")
                messenger.task_log("uv add https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl", level="NONE")
                exit() # Exit cleanly after logging message

def compute_word_features(words: np.ndarray[str], nlp, messenger: UIMessenger):
    """Efficiently gets a combined feature DataFrame for a list of words."""
    with messenger.task("Computing word features"):
        word_series = pd.Series(words)
        messenger.task_log("Running NLP pipeline...", level="INFO")
        docs = list(nlp.pipe(words.tolist()))
        
        messenger.task_log("Extracting vectors and tags...", level="INFO")
        vectors = [doc[0].vector for doc in docs]
        tags = [doc[0].tag_ for doc in docs]
        tags_series = pd.Series(tags, index=word_series.index)

        messenger.task_log("Assembling feature DataFrame...", level="INFO")
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
    savefile: str = 'data/word_features.pkl', 
    recompute: bool = False, 
    save: bool = True,
    model_name: str = "en_core_web_lg",
    messenger: UIMessenger = None
    ) -> pd.DataFrame:
    """Loads pre-computed word features from a file or recomputes them if needed."""
    messenger = get_messenger(messenger)
    with messenger.task(f"Acquiring word features"):
        if not recompute and os.path.exists(savefile):
            messenger.task_log(f"Loading pre-computed features from '{savefile}'...", level="INFO")
            return pd.read_pickle(savefile)

        if recompute:
            messenger.task_log("Recompute requested. Starting new computation...", level="INFO")
        else:
            messenger.task_log("Local file not found. Starting new computation...", level="INFO")

        nlp = load_spacy_model(model_name, messenger)
        features_df = compute_word_features(all_words, nlp, messenger)
        features_df['word'] = all_words
        
        if save:
            messenger.task_log(f"Saving new features to '{savefile}'...", level="INFO")
            features_df.to_pickle(savefile)
            messenger.task_log("Save complete.", level="INFO")
            
        return features_df

def evaluate_training_performance(
    probabilities: np.ndarray[float],
    all_words: np.ndarray[str],
    positive_words: np.ndarray[str],
    threshold: float = 0.5,
    messenger: UIMessenger = None
    ):
    """Evaluates and logs the final model's performance."""
    messenger = get_messenger(messenger)

    with messenger.task(f"Evaluating model performance (threshold >= {threshold:.2%})"):
        results_df = pd.DataFrame({'word': all_words, 'probability': probabilities})
        predicted_positives = set(results_df[results_df['probability'] >= threshold]['word'])
        known_positives = set(positive_words)
        
        true_positives = predicted_positives.intersection(known_positives)
        false_negatives = known_positives.difference(predicted_positives)
        
        recall = len(true_positives) / len(known_positives) if len(known_positives) > 0 else 0
        known_answer_density = len(true_positives) / len(predicted_positives) if len(predicted_positives) > 0 else 0
        
        messenger.task_log(f"{'Total Postives Predicted:':<30} {len(predicted_positives):>5}", level="INFO")
        messenger.task_log(f"{'Total Known Positives:':<30} {len(known_positives):>5}", level="INFO")
        messenger.task_log(f"{'Identified Positives (TPs):':<30} {len(true_positives):>5}", level="INFO")
        messenger.task_log(f"{'False Negatives:':<30} {len(false_negatives):>5}", level="INFO")
        messenger.task_log("", level="NONE")
        messenger.task_log(f"{'Recall on Positives:':<30} {recall:>5.2%}", level="INFO")
        messenger.task_log(f"{'Known Answer Density:':<30} {known_answer_density:>5.2%}", level="INFO")
        messenger.task_log("", level="NONE")
        
        messenger.task_log("False Negatives (Words the Model Missed)", level="STEP")
        fn_df = results_df[results_df['word'].isin(false_negatives)]
        formatted_fns = [f"{row.word} ({row.probability:.1%})" for row in fn_df.sort_values('word').itertuples()]
        
        if not formatted_fns:
            messenger.task_log("None", level="INFO")
        else:
            words_per_row = 5
            for i in range(0, len(formatted_fns), words_per_row):
                chunk = formatted_fns[i:i + words_per_row]
                messenger.task_log(", ".join(chunk), level="INFO")

def train_classifier(
    feature_df: pd.DataFrame, 
    positive_words: np.ndarray,
    all_words: np.ndarray, 
    config: dict = DEFAULT_CONFIG,
    messenger: UIMessenger = None 
    ) -> dict:
    """Trains the PU classifier from scratch and evaluates its performance."""
    messenger = get_messenger(messenger)

    with messenger.task("Training new classifier model"):
        train_df = feature_df.copy()
        train_df['type'] = np.where(train_df['word'].isin(positive_words), 'P', 'U')
        
        messenger.task_log("Performing spy selection...", level="INFO")
        p_df = train_df[train_df['type'] == 'P']
        u_df = train_df[train_df['type'] == 'U']
        spies_df = p_df.sample(frac=config['spy_rate'], random_state=config['random_seed'])
        reliable_positives_df = p_df.drop(spies_df.index)

        enabled_features = list(config['explicit_features'].keys())

        def get_matrix(df: pd.DataFrame) -> np.ndarray[np.float64]:
            explicit = df[enabled_features].values.astype(np.float64)
            if config['use_vectors']:
                embedding = np.array(df['vector'].tolist())
                return np.concatenate([embedding, explicit], axis=1)
            return explicit

        X_reliable_positives = get_matrix(reliable_positives_df)
        X_unlabeled = get_matrix(u_df)
        X_spies = get_matrix(spies_df)

        unlabeled_weights = np.full(len(X_unlabeled), 0.5)
        model, scaler = None, None
        feature_slice = slice(-len(enabled_features), None) if config['use_vectors'] else slice(None)
        weights_vector = np.array([config['explicit_features'][feat] for feat in enabled_features])

        # --- New progress bar logic ---
        p_bar_started = False
        last_weight = None
        epsilon = 1e-9

        messenger.task_log("Starting iterative Spy EM process...", level="STEP")
        for i in range(config['max_iterations']):
            with messenger.task(f"Iteration {i+1}/{config['max_iterations']} (max)"):
                X_train = np.concatenate([X_reliable_positives, X_unlabeled])
                y_train = np.array([1] * len(X_reliable_positives) + [0] * len(X_unlabeled))
                sample_weights = np.concatenate([np.ones(len(X_reliable_positives)), unlabeled_weights])

                scaler = StandardScaler().fit(X_train[:, feature_slice])
                X_train_scaled = X_train.copy()
                X_train_scaled[:, feature_slice] = scaler.transform(X_train[:, feature_slice]) * weights_vector

                model = LogisticRegression(solver='liblinear', random_state=config['random_seed'])
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

                X_spies_scaled = X_spies.copy()
                X_spies_scaled[:, feature_slice] = scaler.transform(X_spies[:, feature_slice]) * weights_vector
                spy_probs = model.predict_proba(X_spies_scaled)[:, 1]
                c = np.mean(spy_probs)
                messenger.task_log(f"Average spy probability (c): {c:.4f}", level="INFO")

                X_unlabeled_scaled = X_unlabeled.copy()
                X_unlabeled_scaled[:, feature_slice] = scaler.transform(X_unlabeled[:, feature_slice]) * weights_vector
                unlabeled_probs = model.predict_proba(X_unlabeled_scaled)[:, 1]
                
                new_unlabeled_weights = np.clip(unlabeled_probs / c, 0, 1)
                weight_change = np.sum(np.abs(new_unlabeled_weights - unlabeled_weights))
                messenger.task_log(f"Total change in weights: {weight_change:.4f}", level="INFO")
                
                # --- Update convergence progress bar ---
                if not p_bar_started:
                    log_start = np.log(max(weight_change, epsilon))
                    log_target = np.log(config['convergence_tolerance'])
                    if log_start > log_target:
                        messenger.start_progress(total = log_start-log_target, desc="Training Convergence")
                        p_bar_started = True
                
                elif last_weight is not None:
                    log_last = np.log(max(last_weight, epsilon))
                    log_new = np.log(max(weight_change, epsilon))
                    messenger.update_progress(advance = log_last-log_new)
                
                last_weight = weight_change

                if weight_change < config['convergence_tolerance']:
                    messenger.task_log("Convergence reached.", level="INFO")
                    break
                unlabeled_weights = new_unlabeled_weights

        if p_bar_started:
            messenger.stop_progress()

        if model and scaler:
            eval_df = train_df.copy()
            X_inf = get_matrix(eval_df)
            X_inf[:, feature_slice] = scaler.transform(X_inf[:, feature_slice]) * weights_vector
            probabilities =  model.predict_proba(X_inf)[:, 1]
            
            evaluate_training_performance(
                probabilities=probabilities,
                all_words=all_words,
                positive_words=positive_words,
                threshold=config.get('evaluation_threshold', 0.07),
                messenger=messenger
            )

        return {'model': model, 'scaler': scaler}

def load_classifier(
    feature_df: pd.DataFrame, 
    savefile: str = 'data/wordle_classifier.pkl',
    recompute: bool = False,
    save: bool = True,
    positive_words: np.ndarray = None,
    all_words: np.ndarray = None,
    config: dict = DEFAULT_CONFIG,
    messenger: UIMessenger = None
    ) -> Callable:
    """Loads a pre-trained classifier or retrains one if needed."""
    messenger = get_messenger(messenger)
    
    with messenger.task("Acquiring classifier"):
        if not recompute and os.path.exists(savefile):
            messenger.task_log(f"Loading pre-trained model artifacts from '{savefile}'...", level="INFO")
            with open(savefile, 'rb') as f:
                artifacts = pickle.load(f)
        else:
            if config is None:
                raise ValueError("A configuration dictionary must be provided for retraining.")
            
            messenger.task_log("No local file found or recompute requested.", level="INFO")
            trained_components = train_classifier(feature_df, positive_words, all_words, config, messenger)
            artifacts = {**trained_components, 'config': config}
            
            if save and savefile:
                messenger.task_log(f"Saving new model artifacts to '{savefile}'...", level="INFO")
                parent_dir = os.path.dirname(savefile)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                with open(savefile, 'wb') as f:
                    pickle.dump(artifacts, f)

        messenger.task_log("Pre-computing probabilities for fast lookup...", level="INFO")

        model: LogisticRegression = artifacts['model']
        scaler: StandardScaler = artifacts['scaler']
        model_config = artifacts['config']
        
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
        
        X_all[:, feature_slice] = scaler.transform(X_all[:, feature_slice]) * weights_vector
        all_probabilities = model.predict_proba(X_all)[:, 1]
        
        probability_lookup = pd.Series(all_probabilities, index=indexed_feature_df.index)

        def predict_word_probabilities(words: str|list|np.ndarray) -> float|np.ndarray:
            """Uses a pre-computed lookup table for maximum speed."""
            is_single_word = isinstance(words, str)
            input_words = [words] if is_single_word else words
            probs = [probability_lookup.get(word, 0.0) for word in input_words]
            return probs[0] if is_single_word else np.array(probs)

        return predict_word_probabilities

def filter_words_by_probability(
    prediction_function: Callable,
    words_to_filter: np.ndarray,
    threshold: float = 0.07,
    messenger: UIMessenger = None
    ) -> np.ndarray:
    """
    Uses a prediction function to filter a list of words based on a probability threshold.
    """
    messenger = get_messenger(messenger)
    initial_count = len(words_to_filter)

    with messenger.task(f"Filtering by classifier probability (>= {threshold:.2%})"):
        messenger.task_log("Getting predictions from classifier...", level="INFO")
        probabilities = prediction_function(words_to_filter)
        
        filtered_words = words_to_filter[probabilities >= threshold]
        final_count = len(filtered_words)
        messenger.task_log(f"Filtered from {initial_count} ⟶  {final_count} words.", level="INFO")
        return filtered_words