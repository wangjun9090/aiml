import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from collections import Counter
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)

# File paths
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.1.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
TRANSFORMER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/power_transformer.pkl'

# Persona constants (unchanged)
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']
PERSONA_OVERSAMPLING_RATIO = {
    'drug': 4.5, 'dental': 3.5, 'doctor': 4.8, 'dsnp': 4.0, 'vision': 2.5, 'csnp': 4.5
}
PERSONA_CLASS_WEIGHT = {
    'drug': 5.0, 'dental': 4.5, 'doctor': 5.5, 'dsnp': 4.8, 'vision': 3.0, 'csnp': 5.2
}
PERSONA_THRESHOLD = {
    'drug': 0.28, 'dental': 0.25, 'doctor': 0.22, 'dsnp': 0.25, 'vision': 0.30, 'csnp': 0.24
}
HIGH_PRIORITY_PERSONAS = ['drug', 'dsnp', 'dental', 'doctor', 'csnp']
SUPER_PRIORITY_PERSONAS = ['drug', 'dsnp', 'doctor', 'csnp']
PERSONA_FEATURES = {
    'dental': ['query_dental', 'filter_dental', 'time_dental_pages', 'accordion_dental', 'ma_dental_benefit'],
    'doctor': ['query_provider', 'filter_provider', 'click_provider', 'ma_provider_network'],
    'dsnp': ['query_dsnp', 'filter_dsnp', 'time_dsnp_pages', 'accordion_dsnp', 'dsnp'],
    'drug': ['query_drug', 'filter_drug', 'time_drug_pages', 'accordion_drug', 'click_drug', 'ma_drug_benefit'],
    'vision': ['query_vision', 'filter_vision', 'time_vision_pages', 'accordion_vision', 'ma_vision'],
    'csnp': ['query_csnp', 'filter_csnp', 'time_csnp_pages', 'accordion_csnp', 'csnp']
}
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp', 'time_col': 'time_csnp_pages', 'accordion_col': 'accordion_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental', 'time_col': 'time_dental_pages', 'accordion_col': 'accordion_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp', 'time_col': 'time_dsnp_pages', 'accordion_col': 'accordion_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug', 'time_col': 'time_drug_pages', 'accordion_col': 'accordion_drug'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision', 'time_col': 'time_vision_pages', 'accordion_col': 'accordion_vision'}
}

# Helper functions (unchanged, abbreviated for brevity)
def generate_synthetic_persona_examples(X, feature_columns, persona, num_samples=1000):
    # [Same as previous, generates synthetic data for each persona]
    synthetic_examples = []
    # ... (omitted for brevity)
    return pd.DataFrame(synthetic_examples)

def safe_bool_to_int(boolean_value, df):
    # [Same as previous, converts boolean to int]
    if isinstance(boolean_value, pd.Series):
        return boolean_value.astype(int)
    return pd.Series([int(boolean_value)] * len(df), index=df.index)

def get_feature_as_series(df, col_name, default=0):
    # [Same as previous, ensures feature as Series]
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default] * len(df), index=df.index)

def normalize_persona(df):
    # [Same as previous, normalizes persona column]
    valid_personas = PERSONAS
    new_rows = []
    # ... (omitted for brevity)
    return pd.DataFrame(new_rows).reset_index(drop=True)

def calculate_persona_weight(row, persona_info, persona):
    # [Same as previous, computes persona weight]
    query_col = persona_info['query_col']
    # ... (omitted for brevity)
    return min(max(weight, 0), 1.0)

# HIGHLIGHT: Load_data function where persona column is loaded for ground truth
def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw behavioral columns: {list(behavioral_df.columns)}")
        
        # HIGHLIGHT: Load and process 'persona' column for ground truth labels during evaluation
        # Note: 'persona' column is used only for ground truth, not as input for prediction
        if 'persona' in behavioral_df.columns:
            logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
            logger.info(f"Persona value counts:\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        else:
            logger.warning("Persona column missing in behavioral data")
        
        # HIGHLIGHT: Clean and normalize 'persona' column to ensure valid ground truth labels
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].fillna('dental')
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        behavioral_df['zip'] = behavioral_df['zip'].fillna('unknown')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].fillna('unknown')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        logger.info(f"Behavioral_df after cleaning: {len(behavioral_df)} rows")
        
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan_df columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# HIGHLIGHT: Prepare_features function where persona is separated as ground truth (y) and features (X) are prepared for prediction
def prepare_features(behavioral_df, plan_df):
    try:
        # HIGHLIGHT: Normalize 'persona' column to ensure valid ground truth labels
        behavioral_df = normalize_persona(behavioral_df)
        
        if behavioral_df.empty:
            logger.warning("Behavioral_df is empty after normalization. Using plan_df only.")
            training_df = plan_df.copy()
            training_df['persona'] = 'dental'
        else:
            training_df = behavioral_df.merge(
                plan_df.rename(columns={'StateCode': 'state'}),
                how='left', on=['zip', 'plan_id']
            ).reset_index(drop=True)
            logger.info(f"Rows after merge: {len(training_df)}")
            logger.info(f"training_df columns: {list(training_df.columns)}")
        
        # Define features (excluding persona)
        plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network']
        for col in plan_features:
            if col not in training_df.columns:
                training_df[col] = 0
            else:
                training_df[col] = training_df[col].fillna(0)
        
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_provider', 'accordion_vision', 'accordion_dsnp'
        ]
        
        imputer = SimpleImputer(strategy='median')
        for col in behavioral_features:
            if col in training_df.columns:
                training_df[col] = imputer.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = 0

        # [Feature engineering: temporal features, clustering, embeddings, etc. - omitted for brevity]
        if 'start_time' in training_df.columns:
            training_df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(training_df['start_time'])).dt.days.fillna(30)
            training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
            training_df['time_of_day'] = pd.to_datetime(training_df['start_time']).dt.hour.fillna(12) // 6
        else:
            training_df['recency'] = 30
            training_df['visit_frequency'] = 1
            training_df['time_of_day'] = 2
        
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        if all(col in training_df.columns for col in cluster_features):
            kmeans = KMeans(n_clusters=5, random_state=42)
            training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
        else:
            training_df['user_cluster'] = 0
        
        training_df['dental_time_ratio'] = training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5)
        training_df['click_ratio'] = training_df.get('num_clicks', 0) / (training_df.get('num_pages_viewed', 1) + 1e-5)
        
        if 'plan_id' in training_df.columns:
            plan_sentences = training_df.groupby('userid')['plan_id'].apply(list).tolist()
            w2v_model = Word2Vec(sentences=plan_sentences, vector_size=10, window=5, min_count=1, workers=4)
            plan_embeddings = training_df['plan_id'].apply(
                lambda x: w2v_model.wv[x] if x in w2v_model.wv else np.zeros(10)
            )
            embedding_cols = [f'plan_emb_{i}' for i in range(10)]
            training_df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=training_df.index)
        else:
            embedding_cols = [f'plan_emb_{i}' for i in range(10)]
            training_df[embedding_cols] = 0
        
        query_cols = [c for c in behavioral_features if c.startswith('query_') and c in training_df.columns]
        filter_cols = [c for c in behavioral_features if c.startswith('filter_') and c in training_df.columns]
        training_df['query_count'] = training_df[query_cols].sum(axis=1) if query_cols else pd.Series(0, index=training_df.index)
        training_df['filter_count'] = training_df[filter_cols].sum(axis=1) if filter_cols else pd.Series(0, index=training_df.index)
        
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        # [Additional feature engineering for personas - abbreviated]
        additional_features = []
        for persona in PERSONAS:
            persona_info = PERSONA_INFO.get(persona, {})
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'))
            # ... (omitted for brevity, includes persona-specific features like dental_engagement_score, doctor_interaction_score, etc.)
        
        # HIGHLIGHT: Define feature columns for prediction (excluding persona)
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster', 
            'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        
        # HIGHLIGHT: Separate features (X) and ground truth (y)
        # Note: 'persona' column is used as ground truth (y), not included in features (X)
        X = training_df[feature_columns].fillna(0)
        variances = X.var()
        valid_features = variances[variances > 1e-5].index.tolist()
        X = X[valid_features]
        logger.info(f"Selected features after variance filtering: {valid_features}")
        
        # HIGHLIGHT: Extract 'persona' column as ground truth (y)
        y = training_df['persona']
        training_df = training_df[training_df['persona'].notna()].reset_index(drop=True)
        logger.info(f"Rows after filtering: {len(training_df)}")
        logger.info(f"Pre-SMOTE persona distribution:\n{training_df['persona'].value_counts(dropna=False).to_string()}")
        
        # [Synthetic data generation and SMOTE - abbreviated]
        for persona in PERSONAS:
            num_samples = 2000 if persona in SUPER_PRIORITY_PERSONAS else (
                1500 if persona == 'dental' else 800)
            synthetic_examples = generate_synthetic_persona_examples(X, valid_features, persona, num_samples=num_samples)
            X = pd.concat([X, synthetic_examples], ignore_index=True)
            y = pd.concat([y, pd.Series([persona] * len(synthetic_examples))], ignore_index=True)
            logger.info(f"After adding synthetic {persona} examples: {Counter(y)}")
        
        class_counts = pd.Series(y).value_counts()
        sampling_strategy = {
            persona: int(count * PERSONA_OVERSAMPLING_RATIO.get(persona, 2.0))
            for persona, count in class_counts.items()
        }
        logger.info(f"Balanced SMOTE sampling strategy: {sampling_strategy}")
        
        smote = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
        X, y = smote.fit_resample(X, y)
        logger.info(f"Rows after balanced SMOTE: {len(X)}")
        logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(y).value_counts().to_string()}")
        
        return X, y, None  # Transformer loaded later
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    # [Same as previous, computes per-persona accuracy]
    per_persona_accuracy = {}
    # ... (omitted for brevity)
    return per_persona_accuracy

# HIGHLIGHT: Main function where evaluation uses ground truth (y_test) for metrics
def main():
    logger.info("Starting model evaluation...")
    
    # HIGHLIGHT: Load data, including 'persona' column for ground truth
    behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    
    # HIGHLIGHT: Prepare features, separating X (features) and y (ground truth)
    X, y, _ = prepare_features(behavioral_df, plan_df)
    
    # HIGHLIGHT: Split data, keeping 20% as test set with ground truth labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # HIGHLIGHT: Load label encoder and transform ground truth labels
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        le = pickle.load(f)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # HIGHLIGHT: Apply transformer to features (X), not ground truth
    with open(TRANSFORMER_FILE, 'rb') as f:
        transformer = pickle.load(f)
    X_train = pd.DataFrame(transformer.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(transformer.transform(X_test), columns=X_test.columns)
    
    # HIGHLIGHT: Load model and predict using features (X_test) only
    with open(MODEL_FILE, 'rb') as f:
        main_model = pickle.load(f)
    
    # Load binary classifiers
    binary_classifiers = {}
    for persona in PERSONAS:
        binary_model_path = MODEL_FILE.replace('.pkl', f'_{persona}_binary.pkl')
        try:
            with open(binary_model_path, 'rb') as f:
                binary_classifiers[persona] = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Binary classifier for {persona} not found at {binary_model_path}")
            raise
    
    # HIGHLIGHT: Generate predictions using features (X_test)
    y_pred_probas_multi = main_model.predict_proba(X_test)
    
    # Get binary classifier probabilities
    binary_probas = {}
    for persona, classifier in binary_classifiers.items():
        binary_probas[persona] = classifier.predict_proba(X_test)[:,1]
    
    # [Blending probabilities and overrides - abbreviated]
    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            if persona in ['doctor', 'csnp']:
                blend_ratio = 0.35
            elif persona in ['drug', 'dsnp']:
                blend_ratio = 0.4
            elif persona in ['dental']:
                blend_ratio = 0.5
            else:
                blend_ratio = 0.6
            y_pred_probas_multi[:, i] = blend_ratio * y_pred_probas_multi[:, i] + (1-blend_ratio) * binary_probas[persona]
    
    y_pred = np.argmax(y_pred_probas_multi, axis=1)
    
    # [Override logic - omitted for brevity]
    
    # HIGHLIGHT: Evaluate predictions against ground truth (y_test_encoded)
    overall_accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    logger.info(f"Overall Accuracy on Test Set: {overall_accuracy * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
    # HIGHLIGHT: Compute per-persona metrics using ground truth
    per_persona_accuracy = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    test_sample_counts = pd.Series(y_test).value_counts()
    
    logger.info("\nPer-Persona Test Metrics:")
    logger.info(f"{'Persona':<12} {'Test Samples':<12} {'Accuracy (%)':<12}")
    logger.info("-" * 36)
    for persona in PERSONAS:
        sample_count = test_sample_counts.get(persona, 0)
        accuracy = per_persona_accuracy.get(persona, 0.0)
        logger.info(f"{persona:<12} {sample_count:<12} {accuracy:.2f}")
    
    logger.info("Classification Report:\n" + classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test_encoded, y_pred)))
    
    # HIGHLIGHT: Report feature importances (based on features, not persona)
    logger.info(f"Feature importances: {dict(zip(X.columns, main_model.get_feature_importance()))}")

if __name__ == "__main__":
    main()
