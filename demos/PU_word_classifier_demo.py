import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import time

# --- REAL IMPLEMENTATION LIBRARIES ---
# Make sure you have these installed: pip install numpy pandas scikit-learn wordfreq spacy
# You will also need to download the spaCy model.
import spacy
from wordfreq import word_frequency
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# --- 1. DATA LOADING & SETUP ---

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
            print(f"pip install https://github.com/explosion/spacy-models/releases/download/{model_name}-3.4.0/{model_name}-3.4.0.tar.gz")
            exit()

def load_word_list(filename):
    """Loads a list of words from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file '{filename}' was not found. Please make sure it's in the correct path.")
    with open(filename, 'r') as f:
        words = {line.strip().lower() for line in f if line.strip()}
    return words

# --- 2. FEATURE ENGINEERING WITH SPACY ---

def get_features_for_word(word, nlp):
    """
    Gets a combined feature dictionary for a word using spaCy and wordfreq.
    This function remains unchanged as requested.
    """
    doc = nlp(word)
    token = doc[0]
    vector = token.vector
    freq = word_frequency(word, 'en')
    tag = token.tag_
    is_plural = 1 if tag == 'NNS' else 0
    is_past_tense = 1 if tag == 'VBD' else 0
    is_adjective = 1 if tag == 'JJ' else 0
    is_proper_noun = 1 if tag == 'NNP' else 0
    is_gerund = 1 if tag == 'VBG' else 0
    vowels = "aeiou"
    vowel_count = sum(1 for char in word if char in vowels)
    has_double_letter = 1 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0
    return {
        'vector': vector, 'frequency': freq, 'is_plural': is_plural,
        'is_past_tense': is_past_tense, 'is_adjective': is_adjective,
        'is_proper_noun': is_proper_noun, 'is_gerund': is_gerund,
        'vowel_count': vowel_count, 'has_double_letter': has_double_letter
    }

# --- 3. MAIN SCRIPT LOGIC ---

def main():
    """Main function to run the data preparation, true Spy EM PU training, and evaluation."""
    # --- Configuration ---
    PAST_ANSWERS_FILE = "data/past_answers.txt"
    PRE_NYT_ANSWERS_FILE = "data/original_answers.txt"
    VALID_GUESSES_FILE = "data/valid_guesses.txt"
    SPACY_MODEL_NAME = "en_core_web_lg"
    SPY_RATE = 0.15
    MAX_ITERATIONS = 250 # Maximum number of EM iterations
    CONVERGENCE_TOLERANCE = 1e-02 # Tolerance for convergence check

    ACTIVE_EXPLICIT_FEATURES = [
        'frequency', 
        'is_plural', 
        'is_past_tense', 
        'is_adjective',
        'is_proper_noun', 
        'is_gerund', 
        'vowel_count', 
        'has_double_letter'
    ]
    
    n_explicit_features = len(ACTIVE_EXPLICIT_FEATURES)

    print(f"\n--- Using the following explicit features: {ACTIVE_EXPLICIT_FEATURES} ---")
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
    feature_data = all_words_df['word'].apply(lambda word: get_features_for_word(word, nlp))
    feature_df = pd.json_normalize(feature_data)
    all_words_df = pd.concat([all_words_df, feature_df], axis=1)
    
    if 'frequency' in ACTIVE_EXPLICIT_FEATURES:
        all_words_df['frequency'] = np.log1p(all_words_df['frequency'])

    # --- Spy Selection ---
    print(f"\n--- Selecting {SPY_RATE:.0%} of positives as spies ---")
    p_df = all_words_df[all_words_df['type'] == 'P']
    u_df = all_words_df[all_words_df['type'] == 'U']
    
    spies_df = p_df.sample(frac=SPY_RATE, random_state=42)
    reliable_positives_df = p_df.drop(spies_df.index)

    # --- Prepare feature matrices once, as they don't change ---
    def get_feature_matrix(df):
        explicit = df[ACTIVE_EXPLICIT_FEATURES].values
        embedding = np.array(df['vector'].tolist())
        return np.concatenate([embedding, explicit], axis=1)

    X_reliable_positives = get_feature_matrix(reliable_positives_df)
    X_unlabeled = get_feature_matrix(u_df)
    X_spies = get_feature_matrix(spies_df)

    # <<< CHANGE START: The entire iterative process is replaced >>>
    
    print("\n--- Starting Iterative Spy EM Process ---")

    # Initialize weights for unlabeled data as 0.5
    unlabeled_weights = np.full(len(X_unlabeled), 0.5)
    final_model = None # Ensure we have a final model
    final_scaler = None

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")

        # --- M-Step: Train model with current weights ---
        # Combine reliable positives and the entire unlabeled set for training
        X_train_iter = np.concatenate([X_reliable_positives, X_unlabeled])
        y_train_iter = np.array([1] * len(X_reliable_positives) + [0] * len(X_unlabeled))

        # Weights: 1.0 for reliable positives, calculated weights for unlabeled
        current_weights = np.concatenate([np.ones(len(X_reliable_positives)), unlabeled_weights])

        # Fit scaler ONLY on the current training data's explicit features
        scaler_iter = StandardScaler().fit(X_train_iter[:, -n_explicit_features:])
        
        # Scale a copy of the training data
        X_train_iter_scaled = X_train_iter.copy()
        X_train_iter_scaled[:, -n_explicit_features:] = scaler_iter.transform(X_train_iter[:, -n_explicit_features:])

        # Train with sample weights
        # iter_model = SVC(kernel='linear', probability=True, random_state=42, class_weight=None) # Use sample_weight instead of class_weight

        # OPTION 1: Logistic Regression (More stable linear model)
        iter_model = LogisticRegression(solver='liblinear', random_state=42, class_weight=None)

        # OPTION 2: LightGBM (Powerful non-linear model)
        # iter_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=0)

        iter_model.fit(X_train_iter_scaled, y_train_iter, sample_weight=current_weights)

        # --- E-Step: Re-evaluate probabilities and weights ---
        # Scale the spy and unlabeled features using the new scaler
        X_spies_scaled = X_spies.copy()
        X_spies_scaled[:, -n_explicit_features:] = scaler_iter.transform(X_spies[:, -n_explicit_features:])
        
        X_unlabeled_scaled = X_unlabeled.copy()
        X_unlabeled_scaled[:, -n_explicit_features:] = scaler_iter.transform(X_unlabeled[:, -n_explicit_features:])
        
        # Calculate `c`, the average probability of a true positive (a spy)
        spy_probs = iter_model.predict_proba(X_spies_scaled)[:, 1]
        c = np.mean(spy_probs)
        print(f"Average spy probability (c): {c:.4f}")
        
        # Update weights for all unlabeled data
        unlabeled_probs = iter_model.predict_proba(X_unlabeled_scaled)[:, 1]
        new_unlabeled_weights = unlabeled_probs / c
        new_unlabeled_weights = np.clip(new_unlabeled_weights, 0, 1) # Ensure weights are between 0 and 1

        # Check for convergence
        weight_change = np.sum(np.abs(new_unlabeled_weights - unlabeled_weights))
        print(f"Total change in weights: {weight_change:.4f}")
        if weight_change < CONVERGENCE_TOLERANCE:
            print("Convergence reached: Weights have stabilized.")
            final_model = iter_model
            final_scaler = scaler_iter
            break
        
        unlabeled_weights = new_unlabeled_weights
        final_model = iter_model # Keep the last trained model
        final_scaler = scaler_iter

    else: # This 'else' belongs to the 'for' loop, runs if the loop finishes without 'break'
        print("Reached max iterations. Using the model from the final iteration.")

    # <<< CHANGE END >>>

    # --- Evaluation & Inference (This part remains the same) ---
    print("\n--- Evaluating Final Model ---")
    print("\n--- Predicting Likeliness of New Words using Final PU Model ---")

    def get_predictions_for_word_list(words, model, nlp, scaler, active_features):
        """Efficiently gets predictions for a list of words."""
        # This function is unchanged but now uses the final_model and final_scaler from the EM loop
        print(f"Processing {len(words)} words...")
        start_time = time.time()
        
        feature_list = [get_features_for_word(word, nlp) for word in words]
        inference_df = pd.DataFrame(feature_list)
        if 'frequency' in active_features:
            inference_df['frequency'] = np.log1p(inference_df['frequency'])
        
        explicit_features_inf = inference_df[active_features].values
        embedding_features_inf = np.array(inference_df['vector'].tolist())
        X_inf = np.concatenate([embedding_features_inf, explicit_features_inf], axis=1)
        X_inf[:, -len(active_features):] = scaler.transform(X_inf[:, -len(active_features):])
        
        probabilities = model.predict_proba(X_inf)[:, 1]
        end_time = time.time()
        print(f"Batch processing finished in {end_time - start_time:.2f} seconds.")
        return probabilities

    test_words = ['slate', 'crept', 'xylyl', 'audio', 'words', 'abaci', 'gofer', 'wordy', 'brass', 'board', 'tizzy', 'nervy', 'atria', 'taupe']
    if final_model and final_scaler:
        test_probs = get_predictions_for_word_list(test_words, final_model, nlp, final_scaler, ACTIVE_EXPLICIT_FEATURES)
        for word, prob in zip(test_words, test_probs):
            print(f"The word '{word}' has a {prob:.2%} probability of being a Wordle answer.")
    else:
        print("Could not generate a final model. Aborting inference.")


if __name__ == '__main__':
    main()