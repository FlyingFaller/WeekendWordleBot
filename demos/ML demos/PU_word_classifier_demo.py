import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import time
from functools import wraps
from sklearn.neural_network import MLPClassifier

# --- REAL IMPLEMENTATION LIBRARIES ---
# Make sure you have these installed: pip install numpy pandas scikit-learn wordfreq spacy
# You will also need to download the spaCy model.
import spacy
from wordfreq import word_frequency
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# --- 0. TIMING WRAPPER ---
def timer(func):
    """
    A decorator that prints the execution time of the decorated function.
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"\n'{func.__name__}' completed in {run_time:.4f} seconds.\n")
        return value
    return wrapper_timer

# --- 1. DATA LOADING & SETUP ---
@timer
def load_spacy_model(model_name="en_core_web_lg"):
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

@timer
def load_word_list(filename):
    """Loads a list of words from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file '{filename}' was not found. Please make sure it's in the correct path.")
    with open(filename, 'r') as f:
        words = {line.strip().lower() for line in f if line.strip()}
    return words

# --- 2. FEATURE ENGINEERING WITH SPACY ---
@timer
def get_features_for_words_vectorized(words, nlp):
    """
    Efficiently gets a combined feature DataFrame for a list of words using
    spaCy's nlp.pipe for batch processing and pandas for vectorized operations.
    This function returns raw, unweighted features.
    """
    word_series = pd.Series(words)
    docs = list(nlp.pipe(words))
    
    vectors = [doc[0].vector for doc in docs]
    tags = [doc[0].tag_ for doc in docs]
    tags_series = pd.Series(tags, index=word_series.index)

    features = pd.DataFrame(index=word_series.index)
    
    # A regular plural is tagged as a plural noun (NNS) AND ends in 's'.
    features['is_regular_plural'] = ((tags_series == 'NNS') & (word_series.str.endswith('s'))).astype(int)
    # An irregular plural is tagged as a plural noun (NNS) but does NOT end in 's'.
    features['is_irregular_plural'] = ((tags_series == 'NNS') & (~word_series.str.endswith('s'))).astype(int)

    features['frequency'] = [word_frequency(word, 'en') for word in words]
    features['is_past_tense'] = (tags_series == 'VBD').astype(int)
    features['is_adjective'] = (tags_series == 'JJ').astype(int)
    features['is_proper_noun'] = (tags_series == 'NNP').astype(int)
    features['is_gerund'] = (tags_series == 'VBG').astype(int)
    features['vowel_count'] = word_series.str.count('[aeiou]').astype(int)
    # features['has_double_letter'] = word_series.str.contains(r'(.)\1', regex=True).astype(int)
    features['has_double_letter'] = (word_series.str.findall(r'(.)\1').str.len() > 0).astype(int)
    features['vector'] = vectors
    
    return features

@timer
def get_feature_matrix(df, use_vectors, explicit_features_list):
    """Builds the feature matrix, conditionally including word vectors."""
    explicit = df[explicit_features_list].values.astype(np.float64)
    if use_vectors:
        embedding = np.array(df['vector'].tolist())
        return np.concatenate([embedding, explicit], axis=1)
    else:
        return explicit

# --- 3. GET PREDICTIONS FROM MODEL ---
@timer
def get_predictions_for_word_list(words, model, nlp, scaler, use_vectors, explicit_features, feature_slice, feature_weights_vector):
    """Efficiently gets predictions for a list of words, respecting feature configuration."""
    print(f"Processing {len(words)} words for inference...")

    # 1. Get raw features
    inference_df = get_features_for_words_vectorized(words, nlp)
    
    # 2. Apply log transform to frequency if used (before scaling)
    if 'frequency' in explicit_features:
        inference_df['frequency'] = np.log1p(inference_df['frequency'])
        
    # 3. Build the feature matrix
    X_inf = get_feature_matrix(inference_df, use_vectors, explicit_features)
    
    # 4. Scale the feature matrix
    X_inf[:, feature_slice] = scaler.transform(X_inf[:, feature_slice])
    
    # 5. Apply the feature weights to the scaled features
    X_inf[:, feature_slice] *= feature_weights_vector

    # 6. Get probabilities
    probabilities = model.predict_proba(X_inf)[:, 1]
    return probabilities

@timer
def evaluate_model(model, scaler, nlp, all_valid_words: set, known_positives: set, config):
    """
    Evaluates the final model's performance on the entire dataset.
    """
    print("\n--- Final Model Evaluation ---")
    
    # 1. Get predictions for all possible words
    all_words_list = sorted(list(all_valid_words))
    probabilities = get_predictions_for_word_list(
        all_words_list, model, nlp, scaler,
        use_vectors=config['use_vectors'],
        explicit_features=config['explicit_features'],
        feature_slice=config['feature_slice'],
        feature_weights_vector=config['feature_weights_vector']
    )
    
    # 2. Create a DataFrame with results
    results_df = pd.DataFrame({'word': all_words_list, 'probability': probabilities})
    
    # 3. Identify predicted positives based on the threshold
    threshold = config['prediction_threshold']
    predicted_positives = set(results_df[results_df['probability'] >= threshold]['word'])
    
    # 4. Calculate metrics
    true_positives = predicted_positives.intersection(known_positives) # all predicted positives in known positives
    false_negatives = known_positives.difference(predicted_positives) # all known positives not in predicition 
    
    recall = len(true_positives) / len(known_positives) if len(known_positives) > 0 else 0
    precision = len(true_positives) / len(predicted_positives) if len(predicted_positives) > 0 else 0
    
    # 5. Print summary
    print(f"\nEvaluation with probability threshold >= {threshold:.2%}")
    print("-" * 40)
    print(f"{'Total Predictions Made:': <39}{len(predicted_positives): >5}")
    print(f"{'Total Known Positives:': <39}{len(known_positives): >5}")
    print(f"{'Identified Positives (True Positives):': <39}{len(true_positives): >5}")
    print(f"{'Missed Positives (False Negatives):': <39}{len(false_negatives): >5}")
    print("-" * 40)
    print(f"{'Recall on Known Positives:': <48}{f'{recall:.2%}': >7}")
    print(f"{'Positive Predictive Value (PPV) on Labeled Set:': <48}{f'{precision:.2%}': >7}")
    print("(Known Answer Density)")
    print("-" * 40)
    
    print("\n--- False Negatives (Words the Model Missed) ---")
    fn_df = results_df[results_df['word'].isin(false_negatives)]
    formatted_fns = [f"{row.word} ({row.probability:.1%})" for row in fn_df.sort_values('word').itertuples()]
    words_per_row = 5
    for i in range(0, len(formatted_fns), words_per_row):
        chunk = formatted_fns[i:i + words_per_row]
        print(", ".join(chunk))
    if len(formatted_fns) <= 0:
        print('NONE')

    return results_df

# --- 4. MAIN SCRIPT LOGIC ---
@timer
def main():
    """Main function to run the data preparation, true Spy EM PU training, and evaluation."""
    # --- Configuration ---
    PAST_ANSWERS_FILE = "data/past_answers.txt"
    PRE_NYT_ANSWERS_FILE = "data/original_answers.txt"
    VALID_GUESSES_FILE = "data/valid_guesses.txt"
    SPACY_MODEL_NAME = "en_core_web_lg"
    SPY_RATE = 0.15
    MAX_ITERATIONS = 100
    CONVERGENCE_TOLERANCE = 1e-2
    
    USE_WORD_VECTORS = True

    EXPLICIT_FEATURE_WEIGHTS = {
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
    EXPLICIT_FEATURES = list(EXPLICIT_FEATURE_WEIGHTS.keys())
    
    # This array will be used to multiply the feature columns after scaling.
    feature_weights_vector = np.array([EXPLICIT_FEATURE_WEIGHTS[feature] for feature in EXPLICIT_FEATURES])
    
    n_explicit_features = len(EXPLICIT_FEATURES)

    print(f"\n--- Model Configuration ---")
    print(f"Using Word Vectors: {USE_WORD_VECTORS}")
    print(f"Using Explicit Features: {EXPLICIT_FEATURES}")
    print(f"Feature Weights: {EXPLICIT_FEATURE_WEIGHTS}")
    
    nlp = load_spacy_model(SPACY_MODEL_NAME)

    # --- Data Loading (P and U sets) ---
    print("\n--- Loading and Preparing Data for PU Learning ---")
    try:
        positive_examples = load_word_list(PAST_ANSWERS_FILE).union(load_word_list(PRE_NYT_ANSWERS_FILE))
        all_guesses = load_word_list(VALID_GUESSES_FILE)
    except FileNotFoundError as e:
        print(e); return

    unlabeled_examples = all_guesses - positive_examples
    
    # --- Feature Engineering for all data once ---
    print("\n--- Performing Feature Engineering for All Words ---")
    positive_df = pd.DataFrame(sorted(list(positive_examples)), columns=['word'])
    unlabeled_df = pd.DataFrame(sorted(list(unlabeled_examples)), columns=['word'])
    
    all_words_df = pd.concat([positive_df.assign(type='P'), unlabeled_df.assign(type='U')], ignore_index=True)
    
    # 1. Get raw features
    feature_df = get_features_for_words_vectorized(all_words_df['word'].tolist(), nlp)
    all_words_df = pd.concat([all_words_df, feature_df], axis=1)
    
    # 2. Apply log transform before scaling
    if 'frequency' in EXPLICIT_FEATURES:
        all_words_df['frequency'] = np.log1p(all_words_df['frequency'])

    # --- Spy Selection ---
    print(f"\n--- Selecting {SPY_RATE:.0%} of positives as spies ---")
    p_df = all_words_df[all_words_df['type'] == 'P']
    u_df = all_words_df[all_words_df['type'] == 'U']
    
    spies_df = p_df.sample(frac=SPY_RATE, random_state=42)
    reliable_positives_df = p_df.drop(spies_df.index)

    # --- Prepare feature matrices once, as they don't change ---
    X_reliable_positives = get_feature_matrix(reliable_positives_df, USE_WORD_VECTORS, EXPLICIT_FEATURES)
    X_unlabeled = get_feature_matrix(u_df, USE_WORD_VECTORS, EXPLICIT_FEATURES)
    X_spies = get_feature_matrix(spies_df, USE_WORD_VECTORS, EXPLICIT_FEATURES)

    # --- Iterative Spy EM Process ---
    print("\n--- Starting Iterative Spy EM Process ---")
    unlabeled_weights = np.full(len(X_unlabeled), 0.5)
    final_model = None
    final_scaler = None

    explicit_feature_slice = slice(-n_explicit_features, None) if USE_WORD_VECTORS else slice(None)

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")

        X_train_iter = np.concatenate([X_reliable_positives, X_unlabeled])
        y_train_iter = np.array([1] * len(X_reliable_positives) + [0] * len(X_unlabeled))
        current_weights = np.concatenate([np.ones(len(X_reliable_positives)), unlabeled_weights])

        # Step 1: Scale features
        scaler_iter = StandardScaler().fit(X_train_iter[:, explicit_feature_slice])
        X_train_iter_scaled = X_train_iter.copy()
        X_train_iter_scaled[:, explicit_feature_slice] = scaler_iter.transform(X_train_iter[:, explicit_feature_slice])

        # Step 2: Apply weights to scaled features
        X_train_iter_scaled[:, explicit_feature_slice] *= feature_weights_vector

        ### THIS IS THE MODEL SELECTION HERE ###
        iter_model = LogisticRegression(solver='liblinear', random_state=42, class_weight=None)
        # iter_model = MLPClassifier(
        #     hidden_layer_sizes=(2,),  # Two hidden layers with two neurons each
        #     activation='relu',          # Use the ReLU activation function for hidden layers
        #     solver='adam',              # A standard, robust optimizer
        #     max_iter=500,               # Increase iterations as NNs can take longer to converge
        #     random_state=42
        # )

        ### TRAIN THE MODEL ###
        iter_model.fit(X_train_iter_scaled, y_train_iter, sample_weight=current_weights)

        # Apply same scaling and weighting to spy and unlabeled sets for prediction
        X_spies_scaled = X_spies.copy()
        X_spies_scaled[:, explicit_feature_slice] = scaler_iter.transform(X_spies[:, explicit_feature_slice])
        X_spies_scaled[:, explicit_feature_slice] *= feature_weights_vector

        X_unlabeled_scaled = X_unlabeled.copy()
        X_unlabeled_scaled[:, explicit_feature_slice] = scaler_iter.transform(X_unlabeled[:, explicit_feature_slice])
        X_unlabeled_scaled[:, explicit_feature_slice] *= feature_weights_vector

        spy_probs = iter_model.predict_proba(X_spies_scaled)[:, 1]
        c = np.mean(spy_probs)
        print(f"Average spy probability (c): {c:.4f}")
        
        unlabeled_probs = iter_model.predict_proba(X_unlabeled_scaled)[:, 1]
        new_unlabeled_weights = unlabeled_probs / c
        new_unlabeled_weights = np.clip(new_unlabeled_weights, 0, 1)

        weight_change = np.sum(np.abs(new_unlabeled_weights - unlabeled_weights))
        print(f"Total change in weights: {weight_change:.4f}")
        if weight_change < CONVERGENCE_TOLERANCE:
            print("Convergence reached: Weights have stabilized.")
            final_model = iter_model
            final_scaler = scaler_iter
            break
        
        unlabeled_weights = new_unlabeled_weights
        final_model = iter_model
        final_scaler = scaler_iter
    else:
        print("Reached max iterations. Using the model from the final iteration.")

    # --- Evaluation & Inference ---
    print("\n--- Predicting Likeliness of New Words using Final PU Model ---")
    test_words = ['words', 'abaci', 'xylyl', 'slate', 'crept', 'audio', 'gofer', 
                  'wordy', 'brass', 'board', 'tizzy', 'nervy', 'atria', 'taupe',
                  'omega', 'assay', 'frill', 'banjo', 'daunt', 'lumpy', 'rigid',
                  'stork', 'groan', 'coral', 'imbue', 'nasal', 'minty', 'south']
    if final_model and final_scaler:
        # Test on some specific words
        test_probs = get_predictions_for_word_list(test_words, final_model, nlp, final_scaler, 
                                                   use_vectors=USE_WORD_VECTORS, 
                                                   explicit_features=EXPLICIT_FEATURES,
                                                   feature_slice=explicit_feature_slice,
                                                   feature_weights_vector=feature_weights_vector)

        for word, prob in zip(test_words, test_probs):
            print(f"The word '{word}' has a {f'{prob:.2%}': >6} probability of being a Wordle answer.")

        # Evaluation of model
        eval_config = {
            'use_vectors': USE_WORD_VECTORS,
            'explicit_features': EXPLICIT_FEATURES,
            'feature_slice': explicit_feature_slice,
            'feature_weights_vector': feature_weights_vector,
            'prediction_threshold': 0.07
        }
        positive_examples.update({'slate', 'crept', 'audio', 'gofer', 'wordy', 
                                  'brass', 'board', 'tizzy', 'nervy', 'atria', 
                                  'taupe', 'omega', 'assay', 'frill', 'banjo', 
                                  'daunt', 'lumpy', 'rigid', 'stork', 'groan', 
                                  'coral', 'imbue', 'nasal', 'minty', 'south'})
        evaluate_model(final_model, final_scaler, nlp, all_guesses, positive_examples, eval_config)
    else:
        print("Could not generate a final model. Aborting inference.")


if __name__ == '__main__':
    main()
