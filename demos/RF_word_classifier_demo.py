import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import time

# --- REAL IMPLEMENTATION LIBRARIES ---
# Make sure you have these installed: pip install numpy pandas scikit-learn wordfreq spacy
# You will also need to download the spaCy model.
import spacy
from wordfreq import word_frequency
from xgboost import XGBClassifier

# --- 1. DATA LOADING & SETUP ---

def load_spacy_model(model_name="en_core_web_lg"):
    """Loads a spaCy model, prompting the user to download it if not found."""
    try:
        # Load the specified spaCy model
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError:
        print(f"spaCy model '{model_name}' not found. Attempting to download.")
        try:     
            spacy.cli.download(model_name) # This line throws error that
            nlp = spacy.load(model_name)
            return nlp
        except SystemExit:
            print(f"Failed to automatically download model.")
            print("Please download it manually using.")
            print(f"Install '{model_name}' however you install packages with pip")
            # Exit the script gracefully if the model is missing
            exit()

def load_word_list(filename):
    """Loads a list of words from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file '{filename}' was not found. Please make sure it's in the correct path.")
    with open(filename, 'r') as f:
        # Read lines, strip whitespace, and convert to lowercase
        words = {line.strip().lower() for line in f if line.strip()}
    return words

# --- 2. FEATURE ENGINEERING WITH SPACY ---

def get_features_for_word(word, nlp):
    """
    Gets a combined feature dictionary for a word using spaCy and wordfreq.
    """
    # Process the word with the spaCy pipeline
    doc = nlp(word)
    token = doc[0] # We process one word at a time, so we take the first token

    # 1. Get the word vector from spaCy
    vector = token.vector

    # 2. Get the word frequency
    freq = word_frequency(word, 'en')

    # 3. Get specific, relevant syntactic features
    tag = token.tag_
    is_plural = 1 if tag == 'NNS' else 0
    is_past_tense = 1 if tag == 'VBD' else 0
    is_adjective = 1 if tag == 'JJ' else 0
    is_proper_noun = 1 if tag == 'NNP' else 0
    is_gerund = 1 if tag == 'VBG' else 0 # Words ending in "-ing"

    # 4. *** ADDED BACK *** Get orthographic/structural features
    vowels = "aeiou"
    vowel_count = sum(1 for char in word if char in vowels)
    has_double_letter = 1 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0

    return {
        'vector': vector,
        'frequency': freq,
        'is_plural': is_plural,
        'is_past_tense': is_past_tense,
        'is_adjective': is_adjective,
        'is_proper_noun': is_proper_noun,
        'is_gerund': is_gerund,
        'vowel_count': vowel_count,
        'has_double_letter': has_double_letter
    }


# --- 3. MAIN SCRIPT LOGIC ---

def main():
    """Main function to run the data preparation, training, and evaluation."""
    # --- Configuration ---
    # !!! IMPORTANT: UPDATE THESE PATHS IF NEEDED !!!
    PAST_ANSWERS_FILE = "data/past_answers.txt"
    PRE_NYT_ANSWERS_FILE = "data/original_answers.txt"
    VALID_GUESSES_FILE = "data/valid_guesses.txt"
    SPACY_MODEL_NAME = "en_core_web_lg" # The large spaCy model with vectors

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

    print(f"\n--- Using the following explicit features: {ACTIVE_EXPLICIT_FEATURES} ---")
    # --- Setup ---
    nlp = load_spacy_model(SPACY_MODEL_NAME)

    # --- Data Loading ---
    print("\n--- Loading and Preparing Data ---")
    positive_examples = load_word_list(PAST_ANSWERS_FILE).union(load_word_list(PRE_NYT_ANSWERS_FILE))
    valid_guesses = load_word_list(VALID_GUESSES_FILE)
    negative_examples = valid_guesses - positive_examples

    positive_list = sorted(list(positive_examples))
    negative_list = sorted(list(negative_examples))

    positive_df = pd.DataFrame(positive_list, columns=['word']); positive_df['label'] = 1
    negative_df = pd.DataFrame(negative_list, columns=['word']); negative_df['label'] = 0
    df = pd.concat([positive_df, negative_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Feature Engineering ---
    print("\n--- Performing Feature Engineering using spaCy ---")
    
    # Apply the feature extraction function to each word in the DataFrame
    # This is more efficient than calling nlp() on each word individually in a loop
    feature_data = df['word'].apply(lambda word: get_features_for_word(word, nlp))
    feature_df = pd.json_normalize(feature_data) # Expands the dict into columns

    # Combine the new feature columns with the original DataFrame
    df = pd.concat([df, feature_df], axis=1)

    if 'frequency' in ACTIVE_EXPLICIT_FEATURES:
        print("\nApplying log transformation to frequency feature...")
        df['frequency'] = np.log1p(df['frequency'])

    # --- Model Training ---
    print("\n--- Preparing Data and Training Model ---")
    # Use the active features list to select columns for the model
    explicit_features = df[ACTIVE_EXPLICIT_FEATURES].values
    embedding_features = np.array(df['vector'].tolist())
    
    X = np.concatenate([embedding_features, explicit_features], axis=1)
    y = df['label'].values

    # 1. First, split into training+validation (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Then, split the 80% into training (64%) and validation (16%)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)


    # --- Feature Scaling (Normalization) ---
    scaler = StandardScaler()
    n_explicit_features = len(ACTIVE_EXPLICIT_FEATURES)
    X_train[:, -n_explicit_features:] = scaler.fit_transform(X_train[:, -n_explicit_features:])
    # Apply the same transformation to validation and test sets
    X_test[:, -n_explicit_features:] = scaler.transform(X_test[:, -n_explicit_features:])
    print("\nExplicit features have been scaled.")


    # --- Training RandomForest ---
    print("\nTraining RandomForest model...")
    # RandomForestClassifier uses `class_weight='balanced'` to handle imbalance
    # instead of `scale_pos_weight`.
    model = RandomForestClassifier(
        n_estimators=200,       # Number of trees in the forest
        max_depth=10,           # Limit tree depth to prevent overfitting
        class_weight='balanced',# Handle class imbalance
        random_state=42,        # For reproducibility
        n_jobs=-1               # Use all available CPU cores
    )

    # The .fit() call is simpler as RandomForest doesn't use early stopping
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Evaluation ---
    print("\n--- Evaluating Model Performance on Unseen Test Data ---")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Unlikely (0)', 'Likely (1)']))

    # --- Inference on New Words ---
    print("\n--- Predicting Likeliness of New Words ---")

    def get_predictions_for_word_list(words, model, nlp, scaler, active_features):
        """
        Efficiently gets predictions for a large list of words using batch processing.
        """
        print(f"Processing {len(words)} words in a batch...")
        start_time = time.time()
        
        # Use nlp.pipe for efficient processing
        docs = nlp.pipe(words)
        
        # Extract features for all words in a vectorized way
        feature_list = []
        for doc in docs:
            token = doc[0]
            tag = token.tag_
            freq = word_frequency(token.text, 'en')
            vowels = "aeiou"
            
            features = {
                'vector': token.vector,
                'frequency': freq,
                'is_plural': 1 if tag == 'NNS' else 0,
                'is_past_tense': 1 if tag == 'VBD' else 0,
                'is_adjective': 1 if tag == 'JJ' else 0,
                'is_proper_noun': 1 if tag == 'NNP' else 0,
                'is_gerund': 1 if tag == 'VBG' else 0,
                'vowel_count': sum(1 for char in token.text if char in vowels),
                'has_double_letter': 1 if any(token.text[i] == token.text[i+1] for i in range(len(token.text)-1)) else 0
            }
            feature_list.append(features)
        
        # Create a DataFrame from the extracted features
        inference_df = pd.DataFrame(feature_list)

        # Apply log transform if frequency is an active feature
        if 'frequency' in active_features:
            inference_df['frequency'] = np.log1p(inference_df['frequency'])
            
        # Build the full feature matrix for prediction
        explicit_features_inf = inference_df[active_features].values
        embedding_features_inf = np.array(inference_df['vector'].tolist())
        
        X_inf = np.concatenate([embedding_features_inf, explicit_features_inf], axis=1)
        
        # Scale the explicit features using the already-fitted scaler
        num_explicit = len(active_features)
        X_inf[:, -num_explicit:] = scaler.transform(X_inf[:, -num_explicit:])
        
        # Get all probabilities at once
        probabilities = model.predict_proba(X_inf)[:, 1]
        
        end_time = time.time()
        print(f"Batch processing finished in {end_time - start_time:.2f} seconds.")
        
        return probabilities
    
    test_words = ['slate', 'crept', 'xylyl', 'audio', 'words', 'abaci', 'gofer', 'wordy', 'brass', 'board', 'tizzy', 'nervy', 'atria', 'taupe']
    test_probs = get_predictions_for_word_list(test_words, model, nlp, scaler, ACTIVE_EXPLICIT_FEATURES)
    for word, prob in zip(test_words, test_probs):
        print(f"The word '{word}' has a {prob:.2%} probability of being a Wordle answer.")

    # 2. Run on the full valid_guesses list to create the reduced answer list
    print("\n--- Creating Reduced Answer List ---")
    threshold = 0.1 # Example threshold
    
    # Use the efficient batch function on the full list
    all_words_sorted = sorted(list(valid_guesses))
    all_probs = get_predictions_for_word_list(all_words_sorted, model, nlp, scaler, ACTIVE_EXPLICIT_FEATURES)
    
    # Filter the list based on the threshold
    reduced_answer_list = [word for word, prob in zip(all_words_sorted, all_probs) if prob > threshold]
    
    print(f"\nGuess list reduced from {len(valid_guesses):,} to {len(reduced_answer_list):,} ({len(reduced_answer_list) - len(valid_guesses):,})")

if __name__ == '__main__':
    main()
