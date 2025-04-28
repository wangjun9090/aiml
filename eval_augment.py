import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
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
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.1.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
TRANSFORMER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/power_transformer.pkl'

# Persona constants
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']
PERSONA_CLASS_WEIGHT = {
    'drug': 2.5,    # Reduced to balance with others
    'dental': 8.0,  # Increased to boost dental
    'doctor': 7.0,  # Increased to boost doctor
    'dsnp': 4.0,
    'vision': 8.0,  # Increased for rare class
    'csnp': 3.5
}
PERSONA_THRESHOLD = {
    'drug': 0.25,
    'dental': 0.15,  # Lowered to classify more dental
    'doctor': 0.15,  # Lowered to classify more doctor
    'dsnp': 0.20,
    'vision': 0.15,  # Lowered for rare class
    'csnp': 0.20
}
PERSONA_FEATURES = {
    'dental': ['query_dental', 'filter_dental', 'time_dental_pages', 'ma_dental_benefit'],
    'doctor': ['query_provider', 'filter_provider', 'click_provider', 'ma_provider_network'],
    'dsnp': ['query_dsnp', 'filter_dsnp', 'time_dsnp_pages', 'dsnp'],
    'drug': ['query_drug', 'filter_drug', 'time_drug_pages', 'click_drug', 'ma_drug_benefit'],
    'vision': ['query_vision', 'filter_vision', 'time_vision_pages', 'ma_vision'],
    'csnp': ['query_csnp', 'filter_csnp', 'time_csnp_pages', 'csnp']
}
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp', 'time_col': 'time_csnp_pages'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental', 'time_col': 'time_dental_pages'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp', 'time_col': 'time_dsnp_pages'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug', 'time_col': 'time_drug_pages'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision', 'time_col': 'time_vision_pages'}
}

# Helper functions
def safe_bool_to_int(boolean_value, df):
    if isinstance(boolean_value, pd.Series):
        return boolean_value.astype(int)
    return pd.Series([int(boolean_value)] * len(df), index=df.index)

def get_feature_as_series(df, col_name, default=0):
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default] * len(df), index=df.index)

def normalize_persona(df):
    valid_personas = PERSONAS
    new_rows = []
    invalid_personas = set()
    
    for _, row in df.iterrows():
        persona = row['persona']
        if pd.isna(persona) or not persona:
            continue
        
        personas = [p.strip().lower() for p in str(persona).split(',')]
        valid_found = [p for p in personas if p in valid_personas]
        
        if not valid_found:
            invalid_personas.update(personas)
            continue
        
        row_copy = row.copy()
        row_copy['persona'] = valid_found[0]
        new_rows.append(row_copy)
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(f"Rows after persona normalization: {len(result)}")
    if invalid_personas:
        logger.info(f"Invalid personas found: {invalid_personas}")
    if result.empty:
        logger.error(f"No valid personas found. Valid personas: {valid_personas}")
        raise ValueError("No valid personas found")
    return result

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw behavioral columns: {list(behavioral_df.columns)}")
        
        if 'persona' not in behavioral_df.columns:
            logger.error("Persona column missing in behavioral data")
            raise ValueError("Persona column required for evaluation ground truth")
        
        logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
        logger.info(f"Persona value counts:\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        behavioral_df['zip'] = behavioral_df['zip'].fillna('unknown')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].fillna('unknown')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        logger.info(f"Behavioral_df after cleaning: {len(behavioral_df)} rows")
        
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan_df columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).strip()
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def prepare_features(behavioral_df, plan_df, expected_features=None):
    try:
        behavioral_df = normalize_persona(behavioral_df)
        
        if behavioral_df.empty:
            logger.error("Behavioral_df is empty after normalization")
            raise ValueError("No valid data after persona normalization")
        
        training_df = behavioral_df.merge(
            plan_df.rename(columns={'StateCode': 'state'}),
            how='left', on=['zip', 'plan_id']
        ).reset_index(drop=True)
        logger.info(f"Rows after merge: {len(training_df)}")
        logger.info(f"training_df columns: {list(training_df.columns)}")
        
        if 'persona' not in training_df.columns:
            logger.error("Persona column missing in training_df after merge")
            raise ValueError("Persona column required in training_df")
        
        plan_features = ['ma_dental_benefit', 'ma_vision', 'dsnp', 'ma_drug_benefit', 'ma_provider_network', 'csnp']
        for col in plan_features:
            if col not in training_df.columns:
                training_df[col] = 0
            else:
                training_df[col] = training_df[col].fillna(0)
        
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'time_dsnp_pages'
        ]
        
        imputer = SimpleImputer(strategy='median')
        for col in behavioral_features:
            if col in training_df.columns:
                training_df[col] = imputer.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = 0

        sparsity_cols = ['query_dsnp', 'time_dsnp_pages', 'query_drug', 'time_drug_pages', 'query_dental', 'query_provider']
        sparsity_stats = training_df[sparsity_cols].describe().to_dict()
        logger.info(f"Feature sparsity stats:\n{sparsity_stats}")
        
        if 'start_time' in training_df.columns:
            training_df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(training_df['start_time'])).dt.days.fillna(30)
            if 'userid' in training_df.columns:
                training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
            else:
                logger.warning("userid column missing, setting visit_frequency to default value 1")
                training_df['visit_frequency'] = 1
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
        
        if 'plan_id' in training_df.columns:
            plan_sentences = training_df.groupby('userid')['plan_id'].apply(list).tolist() if 'userid' in training_df.columns else training_df['plan_id'].apply(lambda x: [x]).tolist()
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
        
        additional_features = []
        for persona in PERSONAS:
            persona_info = PERSONA_INFO.get(persona, {})
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'))
            filter_col = get_feature_as_series(training_df, persona_info.get('filter_col'))
            click_col = get_feature_as_series(training_df, persona_info.get('click_col', 'dummy_col'))
            time_col = get_feature_as_series(training_df, persona_info.get('time_col', 'dummy_col'))
            plan_col = get_feature_as_series(training_df, persona_info.get('plan_col'))
            
            signal_weights = 3.5 if persona == 'drug' else 3.0
            training_df[f'{persona}_signal'] = (
                query_col * 2.0 +
                filter_col * 2.0 +
                time_col.clip(upper=5) * 1.5 +
                click_col * 2.0
            ) * signal_weights
            additional_features.append(f'{persona}_signal')
            
            has_interaction = (
                (query_col > 0) | 
                (filter_col > 0) | 
                (click_col > 0)
            )
            training_df[f'{persona}_interaction'] = safe_bool_to_int(has_interaction, training_df) * 3.0
            additional_features.append(f'{persona}_interaction')
            
            training_df[f'{persona}_primary'] = (
                safe_bool_to_int(query_col > 0, training_df) * 2.0 +
                safe_bool_to_int(filter_col > 0, training_df) * 2.0 +
                safe_bool_to_int(click_col > 0, training_df) * 2.0 +
                safe_bool_to_int(time_col > 2, training_df) * 1.5
            ) * 2.0
            additional_features.append(f'{persona}_primary')
        
        dental_query = get_feature_as_series(training_df, 'query_dental')
        dental_filter = get_feature_as_series(training_df, 'filter_dental')
        dental_time = get_feature_as_series(training_df, 'time_dental_pages')
        dental_benefit = get_feature_as_series(training_df, 'ma_dental_benefit')
        
        training_df['dental_time_intensity'] = (
            (training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 5.0
        additional_features.append('dental_time_intensity')
        
        training_df['dental_engagement_score'] = (
            dental_query * 3.0 +
            dental_filter * 3.0 +
            dental_time.clip(upper=5) * 2.0 +
            dental_benefit * 4.0
        ) * 3.0
        additional_features.append('dental_engagement_score')
        
        training_df['dental_benefit_multiplier'] = (
            (dental_query + dental_filter) * 
            (dental_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('dental_benefit_multiplier')
        
        provider_query = get_feature_as_series(training_df, 'query_provider')
        provider_filter = get_feature_as_series(training_df, 'filter_provider')
        provider_click = get_feature_as_series(training_df, 'click_provider')
        provider_network = get_feature_as_series(training_df, 'ma_provider_network')
        
        training_df['doctor_interaction_score'] = (
            provider_query * 3.0 +
            provider_filter * 3.0 +
            provider_click * 5.0 +
            provider_network * 4.0
        ) * 4.0
        additional_features.append('doctor_interaction_score')
        
        training_df['doctor_specificity'] = (
            provider_query * 3.0 - 
            (training_df.get('query_dental', 0) + training_df.get('query_vision', 0)) * 0.7
        ).clip(lower=0) * 4.0
        additional_features.append('doctor_specificity')
        
        training_df['doctor_network_boost'] = (
            (provider_query + provider_filter + provider_click) *
            (provider_network + 0.5) * 6.0
        ).clip(lower=0, upper=25)
        additional_features.append('doctor_network_boost')
        
        training_df['doctor_page_depth'] = (
            (provider_click / (provider_query + 0.1)) * 10.0
        ).clip(0, 20)
        additional_features.append('doctor_page_depth')
        
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        dsnp_filter = get_feature_as_series(training_df, 'filter_dsnp')
        dsnp_time = get_feature_as_series(training_df, 'time_dsnp_pages')
        dsnp_plan = get_feature_as_series(training_df, 'dsnp')
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        
        training_df['dsnp_csnp_ratio'] = (
            (dsnp_query + 0.8) / (csnp_query + dsnp_query + 1e-5)
        ).clip(0, 1) * 5.0
        additional_features.append('dsnp_csnp_ratio')
        
        training_df['dsnp_engagement_score'] = (
            dsnp_query * 3.0 +
            dsnp_filter * 3.0 +
            dsnp_time.clip(upper=5) * 2.0 +
            dsnp_plan * 5.0
        ) * 3.0
        additional_features.append('dsnp_engagement_score')
        
        training_df['dsnp_plan_multiplier'] = (
            (dsnp_query + dsnp_filter) *
            (dsnp_plan + 0.5) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('dsnp_plan_multiplier')
        
        drug_query = get_feature_as_series(training_df, 'query_drug')
        drug_filter = get_feature_as_series(training_df, 'filter_drug')
        drug_time = get_feature_as_series(training_df, 'time_drug_pages')
        drug_click = get_feature_as_series(training_df, 'click_drug')
        drug_benefit = get_feature_as_series(training_df, 'ma_drug_benefit')
        
        training_df['drug_engagement_score'] = (
            drug_query * 3.0 +
            drug_filter * 3.0 +
            drug_time.clip(upper=5) * 2.0 +
            drug_click * 4.0 +
            drug_benefit * 4.0
        ) * 3.5
        additional_features.append('drug_engagement_score')
        
        training_df['drug_interest_ratio'] = (
            (drug_query + drug_filter) /
            (training_df.get('query_count', 1) + training_df.get('filter_count', 1) + 1e-5)
        ).clip(upper=0.9) * 10.0
        additional_features.append('drug_interest_ratio')
        
        training_df['drug_benefit_boost'] = (
            (drug_query + drug_filter + drug_click) *
            (drug_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=25)
        additional_features.append('drug_benefit_boost')
        
        training_df['drug_time_intensity'] = (
            (drug_time / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 6.0
        additional_features.append('drug_time_intensity')
        
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        csnp_filter = get_feature_as_series(training_df, 'filter_csnp')
        csnp_time = get_feature_as_series(training_df, 'time_csnp_pages')
        csnp_plan = get_feature_as_series(training_df, 'csnp')
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        
        training_df['csnp_dsnp_ratio'] = (
            (csnp_query + 0.8) / (dsnp_query + csnp_query + 1e-5)
        ).clip(0, 1) * 5.0
        additional_features.append('csnp_dsnp_ratio')
        
        training_df['csnp_specificity'] = (
            csnp_query * 3.0 - 
            (training_df.get('query_dental', 0) + 
             training_df.get('query_vision', 0) + 
             training_df.get('query_drug', 0)) * 0.8
        ).clip(lower=0) * 4.0
        additional_features.append('csnp_specificity')
        
        training_df['csnp_engagement_score'] = (
            csnp_query * 3.0 +
            csnp_filter * 3.0 +
            csnp_time.clip(upper=5) * 2.0 +
            csnp_plan * 5.0
        ) * 4.0
        additional_features.append('csnp_engagement_score')
        
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster'
        ] + embedding_cols
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        logger.info(f"Generated feature columns: {list(X.columns)}")
        logger.info(f"Test set size: {len(X)} samples")
        logger.info(f"Test persona distribution:\n{y.value_counts(dropna=False).to_string()}")
        
        # Filter features to match expected_features
        if expected_features is not None:
            missing_features = [f for f in expected_features if f not in X.columns]
            extra_features = [f for f in X.columns if f not in expected_features]
            
            logger.info(f"Missing features (added as zeros): {missing_features}")
            logger.info(f"Extra features (removed): {extra_features}")
            
            for f in missing_features:
                X[f] = 0
            X = X[expected_features]
        
        logger.info(f"Final feature columns: {list(X.columns)}")
        
        return X, y
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    return per_persona_accuracy

def main():
    logger.info("Starting model evaluation...")
    
    # Load model, label encoder, and transformer
    try:
        with open(MODEL_FILE, 'rb') as f:
            main_model = pickle.load(f)
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            le = pickle.load(f)
        with open(TRANSFORMER_FILE, 'rb') as f:
            transformer = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model or files: {e}")
        raise
    
    expected_features = getattr(main_model, 'feature_names_', None)
    if expected_features is None:
        logger.warning("Model feature_names_ not available, using generated features")
        expected_features = None
    else:
        logger.info(f"Expected feature names from model: {expected_features}")
    
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
    
    # Load and prepare data
    behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    X_test, y_test = prepare_features(behavioral_df, plan_df, expected_features)
    
    # Transform features
    X_test = pd.DataFrame(transformer.transform(X_test), columns=X_test.columns)
    
    # Encode ground truth labels
    y_test_encoded = le.transform(y_test)
    
    # Predict personas
    y_pred_probas_multi = main_model.predict_proba(X_test)
    
    binary_probas = {}
    for persona, classifier in binary_classifiers.items():
        binary_probas[persona] = classifier.predict_proba(X_test)[:,1]
    
    # Blend probabilities with tuned ratios and apply class weights
    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            if persona in ['dental', 'doctor', 'vision']:
                blend_ratio = 0.7  # Higher weight for dental, doctor, vision
            elif persona in ['csnp']:
                blend_ratio = 0.5
            else:
                blend_ratio = 0.3
            y_pred_probas_multi[:, i] = blend_ratio * y_pred_probas_multi[:, i] + (1-blend_ratio) * binary_probas[persona]
        y_pred_probas_multi[:, i] *= PERSONA_CLASS_WEIGHT.get(persona, 1.0)
    
    # Normalize probabilities
    y_pred_probas_multi = y_pred_probas_multi / y_pred_probas_multi.sum(axis=1, keepdims=True)
    
    # Apply persona-specific thresholds
    y_pred = np.zeros(y_pred_probas_multi.shape[0], dtype=int)
    for i in range(y_pred_probas_multi.shape[0]):
        max_prob = -1
        max_idx = 0
        for j, persona in enumerate(le.classes_):
            prob = y_pred_probas_multi[i, j]
            threshold = PERSONA_THRESHOLD.get(persona, 0.5)
            if prob > threshold and prob > max_prob:
                max_prob = prob
                max_idx = j
        y_pred[i] = max_idx
    
    # Evaluate performance
    overall_accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    logger.info(f"Overall Accuracy on Test Set: {overall_accuracy * 100:.2f}%")
    logger.info(f"Macro F1 Score on Test Set: {macro_f1:.2f}")
    
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
    
    logger.info(f"Feature importances: {dict(zip(X_test.columns, main_model.get_feature_importance()))}")

if __name__ == "__main__":
    main()
