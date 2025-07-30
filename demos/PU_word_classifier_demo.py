import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import os
import time

# --- REAL IMPLEMENTATION LIBRARIES ---
# Make sure you have these installed: pip install numpy pandas scikit-learn==1.2.2 wordfreq spacy xgboost pulearn
# You will also need to download the spaCy model.
import spacy
from wordfreq import word_frequency
from xgboost import XGBClassifier
# PU LEARNING CHANGE: Import the PU learning classifier wrapper
from pulearn import ElkanotoPuClassifier


# --- PU LEARNING PATCH CLASS ---
# PU LEARNING FIX: The original ElkanotoPuClassifier from pulearn passes labels
# with `-1` directly to the XGBoost estimator, which causes a ValueError because
# XGBoost expects labels to be 0 and 1. This patched class fixes that.
class PatchedElkanotoPuClassifier(ElkanotoPuClassifier):
    """
    A patched version of the ElkanotoPuClassifier that correctly handles
    estimators like XGBoost which require binary (0, 1) labels.
    """
    def fit(self, X, y):
        """
        Fit the PU classifier.
        Args:
            X (np.array): Feature data.
            y (np.array): Labels, where 1 is positive and -1 is unlabeled.
        """
        # Find the indices of the positive examples from the original y
        positive_indices = np.where(y == 1)[0]
        if len(positive_indices) == 0:
            raise ValueError("The training data must contain at least one positive example.")

        # Convert labels from {-1, 1} to {0, 1} for the estimator
        y_binary_for_estimator = np.where(y == 1, 1, 0)

        # Fit the underlying estimator with the corrected binary labels
        self.estimator.fit(X, y_binary_for_estimator)

        # Estimate c = E[P(s=1|x)] for positive examples using the original positive set.
        positive_probs = self.estimator.predict_proba(X[positive_indices])[:, 1]
        self.c = np.mean(positive_probs)

        if self.c == 0:
            print("Warning: The classifier assigned a probability of 0 to all positive examples. Predictions may be unreliable.")
            # Set c to a very small number to avoid division by zero
            self.c = 1e-9
        return self

    # PU LEARNING FIX 2: Add predict and predict_proba methods to the patched
    # class to make it fully self-contained and avoid issues with the parent
    # class's implementation which was causing the NotFittedError.
    def predict_proba(self, X):
        """
        Predict posterior probabilities P(y=1|x).
        """
        if self.c is None:
            raise NotFittedError(
                "The estimator must be fitted before calling predict_proba(...)."
            )
        
        # Get the standard probabilities P(s=1|x) from the base estimator
        base_probs = self.estimator.predict_proba(X)

        # Adjust probabilities using c: P(y=1|x) = P(s=1|x) / c
        positive_posterior_probs = base_probs[:, 1] / self.c

        # Ensure probabilities are valid (between 0 and 1)
        positive_posterior_probs = np.clip(positive_posterior_probs, 0, 1)

        # Create the final probability array [P(y=0|x), P(y=1|x)]
        final_probs = np.vstack((1 - positive_posterior_probs, positive_posterior_probs)).T
        return final_probs

    def predict(self, X, threshold=0.5):
        """
        Predict class labels based on the adjusted probabilities.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)


# --- 1. DATA LOADING & SETUP ---

def load_spacy_model(model_name="en_core_web_lg"):
    """Loads a spaCy model, prompting the user to download it if not found."""
    try:
        # Load the specified spaCy model
        nlp = spacy.load(model_name)
        print(f"spaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError:
        print(f"spaCy model '{model_name}' not found. Please run 'python -m spacy download {model_name}' to install it.")
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

    # 4. Get orthographic/structural features
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
    # For this demo, you'll need to create a 'data' folder with these files.
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
    print("\n--- Loading and Preparing Data for PU Learning ---")
    positive_examples = load_word_list(PAST_ANSWERS_FILE).union(load_word_list(PRE_NYT_ANSWERS_FILE))
    valid_guesses = load_word_list(VALID_GUESSES_FILE)
    
    # The "negative" examples are now the "unlabeled" set.
    unlabeled_examples = valid_guesses - positive_examples

    positive_list = sorted(list(positive_examples))
    unlabeled_list = sorted(list(unlabeled_examples))

    # Create a DataFrame with positive (1) and unlabeled (-1) labels.
    positive_df = pd.DataFrame(positive_list, columns=['word']); positive_df['label'] = 1
    unlabeled_df = pd.DataFrame(unlabeled_list, columns=['word']); unlabeled_df['label'] = -1
    df = pd.concat([positive_df, unlabeled_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(positive_df)} positive and {len(unlabeled_df)} unlabeled examples.")

    # --- Feature Engineering ---
    print("\n--- Performing Feature Engineering using spaCy ---")
    
    # This part remains the same, as we need features for all words.
    feature_data = df['word'].apply(lambda word: get_features_for_word(word, nlp))
    feature_df = pd.json_normalize(feature_data)
    df = pd.concat([df, feature_df], axis=1)

    if 'frequency' in ACTIVE_EXPLICIT_FEATURES:
        print("\nApplying log transformation to frequency feature...")
        df['frequency'] = np.log1p(df['frequency'])

    # --- Model Training ---
    print("\n--- Preparing Data and Training PU Learning Model ---")
    explicit_features = df[ACTIVE_EXPLICIT_FEATURES].values
    embedding_features = np.array(df['vector'].tolist())
    
    X = np.concatenate([embedding_features, explicit_features], axis=1)
    # y now contains 1s for positive and -1s for unlabeled.
    y = df['label'].values

    # We still split the data to have a final hold-out test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Feature Scaling (Normalization) ---
    scaler = StandardScaler()
    n_explicit_features = len(ACTIVE_EXPLICIT_FEATURES)
    X_train[:, -n_explicit_features:] = scaler.fit_transform(X_train[:, -n_explicit_features:])
    X_test[:, -n_explicit_features:] = scaler.transform(X_test[:, -n_explicit_features:])
    print("\nExplicit features have been scaled.")
    
    # --- Training PU Classifier with XGBoost as estimator ---
    print("\nTraining PU Classifier with XGBoost estimator...")
    
    # 1. Define the base scikit-learn estimator
    estimator = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42,
        max_depth=7,
        use_label_encoder=False # Suppress a common XGBoost warning
    )
    
    # 2. PU LEARNING FIX: Use our new, patched wrapper class
    model = PatchedElkanotoPuClassifier(estimator=estimator)
    
    # 3. Fit the model. It expects the labels to be 1 (positive) and -1 (unlabeled).
    model.fit(X_train, y_train)
    print("Model training complete.")


    # --- Evaluation ---
    print("\n--- Evaluating Model Performance on Unseen Test Data ---")
    # The test set still has -1 labels. For evaluation,
    # we need to treat them as the "negative" class (0).
    y_test_binary = np.where(y_test == 1, 1, 0)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test_binary, y_pred))
    print("\nClassification Report:\n", classification_report(y_test_binary, y_pred, target_names=['Unlikely (0)', 'Likely (1)']))


    # --- Inference on New Words ---
    print("\n--- Predicting Likeliness of New Words ---")

    def get_predictions_for_word_list(words, model, nlp, scaler, active_features):
        """
        Efficiently gets predictions for a large list of words using batch processing.
        """
        print(f"Processing {len(words)} words in a batch...")
        start_time = time.time()
        
        # This inference logic remains largely the same.
        docs = nlp.pipe(words)
        
        feature_list = []
        for doc in docs:
            token = doc[0]
            # This logic is duplicated from get_features_for_word for batch efficiency
            tag = token.tag_
            freq = word_frequency(token.text, 'en')
            vowels = "aeiou"
            features = {
                'vector': token.vector, 'frequency': freq, 'is_plural': 1 if tag == 'NNS' else 0,
                'is_past_tense': 1 if tag == 'VBD' else 0, 'is_adjective': 1 if tag == 'JJ' else 0,
                'is_proper_noun': 1 if tag == 'NNP' else 0, 'is_gerund': 1 if tag == 'VBG' else 0,
                'vowel_count': sum(1 for char in token.text if char in vowels),
                'has_double_letter': 1 if any(token.text[i] == token.text[i+1] for i in range(len(token.text)-1)) else 0
            }
            feature_list.append(features)
        
        inference_df = pd.DataFrame(feature_list)

        if 'frequency' in active_features:
            inference_df['frequency'] = np.log1p(inference_df['frequency'])
            
        explicit_features_inf = inference_df[active_features].values
        embedding_features_inf = np.array(inference_df['vector'].tolist())
        
        X_inf = np.concatenate([embedding_features_inf, explicit_features_inf], axis=1)
        
        num_explicit = len(active_features)
        X_inf[:, -num_explicit:] = scaler.transform(X_inf[:, -num_explicit:])
        
        # Use the predict_proba method from our PU classifier wrapper.
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
    threshold = 0.1 # You can tune this threshold based on desired recall/precision
    
    all_words_sorted = sorted(list(valid_guesses))
    all_probs = get_predictions_for_word_list(all_words_sorted, model, nlp, scaler, ACTIVE_EXPLICIT_FEATURES)
    
    reduced_answer_list = [word for word, prob in zip(all_words_sorted, all_probs) if prob > threshold]
    
    print(f"\nGuess list reduced from {len(valid_guesses):,} to {len(reduced_answer_list):,} words using a {threshold:.0%} threshold.")

if __name__ == '__main__':
    # You will need to create a 'data' directory and place the word list files there.
    # For example: data/past_answers.txt, data/original_answers.txt, data/valid_guesses.txt
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory. Please add your word list .txt files there.")
    else:
        main()
