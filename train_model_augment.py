import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025_clean.csv'
PLAN_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/training/plan_derivation_by_zip.csv"
)
MODEL_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/model-persona-1.1.1.pkl"
)
LABEL_ENCODER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/label_encoder_1.pkl"
)
TRANSFORMER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/power_transformer.pkl"
)

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Adjusted for ≥0.8 accuracy
PERSONA_OVERSAMPLING_RATIO = {
    'drug': 4.5,
    'dental': 3.5,
    'doctor': 4.8,
    'dsnp': 4.0,
    'vision': 5.0,  # Increased for better representation
    'csnp': 4.5
}

PERSONA_CLASS_WEIGHT = {
    'drug': 5.5,    # Increased
    'dental': 4.5,
    'doctor': 6.0,  # Increased
    'dsnp': 5.0,    # Increased
    'vision': 4.5,  # Increased
    'csnp': 5.8     # Increased
}

PERSONA_THRESHOLD = {
    'drug': 0.28,
    'dental': 0.25,
    'doctor': 0.22,
    'dsnp': 0.25,
    'vision': 0.30,
    'csnp': 0.24
}

HIGH_PRIORITY_PERSONAS = ['drug', 'dsnp', 'dental', 'doctor', 'csnp']
SUPER_PRIORITY_PERSONAS = ['drug', 'dsnp', 'doctor', 'csnp', 'vision']  # Added vision

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

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Helper Functions ---
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
        logger.warning(f"No valid personas found. Valid personas: {valid_personas}")
    return result

def calculate_persona_weight(row, persona_info, persona):
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    plan_col = persona_info.get('plan_col', None)
    click_col = persona_info.get('click_col', None)
    
    query_value = row.get(query_col, 0) if pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col, np.nan)) else 0
    plan_value = row.get(plan_col, 0) if plan_col and pd.notna(row.get(plan_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    max_val = max([query_value, filter_value, plan_value, click_value, 1])
    if max_val > 0:
        query_value /= max_val
        filter_value /= max_val
        plan_value /= max_val
        click_value /= max_val
    
    weight = 0.25 * (query_value + filter_value + plan_value + click_value)
    return min(max(weight, 0), 1.0)

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        behavioral_df['zip'] = behavioral_df['zip'].fillna('unknown')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].fillna('unknown')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        
        plan_df = pd.read_csv(plan_path)
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def generate_synthetic_persona_examples(X, feature_columns, persona, num_samples=1000):
    synthetic_examples = []
    persona_features = [col for col in feature_columns if persona in col.lower()]
    specific_features = PERSONA_FEATURES.get(persona, [])
    
    if persona in SUPER_PRIORITY_PERSONAS:
        num_samples = 3000  # Increased for better balance
    elif persona == 'dental':
        num_samples = 2000
    else:
        num_samples = 1500
    
    for _ in range(num_samples):
        sample = {col: 0 for col in feature_columns}
        
        if 'recency' in feature_columns:
            sample['recency'] = np.random.randint(1, 30)
        if 'visit_frequency' in feature_columns:
            sample['visit_frequency'] = np.random.uniform(0.1, 0.5)
        if 'time_of_day' in feature_columns:
            sample['time_of_day'] = np.random.randint(0, 4)
        if 'user_cluster' in feature_columns:
            sample['user_cluster'] = np.random.randint(0, 5)
            
        for feature in persona_features:
            if persona in SUPER_PRIORITY_PERSONAS:
                sample[feature] = np.random.uniform(5.0, 8.0)  # Stronger signals
            elif persona == 'dental':
                sample[feature] = np.random.uniform(3.5, 6.5)
            else:
                sample[feature] = np.random.uniform(2.5, 5.5)
            
        for feature in specific_features:
            if feature in feature_columns:
                if persona in SUPER_PRIORITY_PERSONAS:
                    sample[feature] = np.random.uniform(8.0, 12.0)
                elif persona == 'dental':
                    sample[feature] = np.random.uniform(6.0, 10.0)
                else:
                    sample[feature] = np.random.uniform(4.0, 8.0)
        
        plan_col = PERSONA_INFO.get(persona, {}).get('plan_col')
        if plan_col and plan_col in feature_columns:
            sample[plan_col] = 1
            
        if persona in SUPER_PRIORITY_PERSONAS:
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.1)
        elif persona == 'dental':
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.2)
        else:
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.5)
                    
        synthetic_examples.append(sample)
    
    synthetic_df = pd.DataFrame(synthetic_examples)
    logger.info(f"Generated {len(synthetic_examples)} synthetic {persona} examples")
    return synthetic_df

def prepare_features(behavioral_df, plan_df, expected_features=None):
    try:
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
        
        plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network']
        for col in plan_features:
            training_df[col] = training_df.get(col, pd.Series([0] * len(training_df), index=training_df.index)).fillna(0)
        
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_vision', 'accordion_dsnp'
        ]
        
        sparse_features = ['query_dental', 'time_dental_pages', 'query_vision', 'time_vision_pages', 'query_drug', 'query_provider']
        imputer_median = SimpleImputer(strategy='median')
        imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
        
        for col in behavioral_features:
            if col in training_df.columns:
                if col in sparse_features:
                    training_df[col] = imputer_zero.fit_transform(training_df[[col]]).flatten()
                else:
                    training_df[col] = imputer_median.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
        
        for persona in ['dental', 'vision']:
            query_col = PERSONA_INFO[persona]['query_col']
            time_col = PERSONA_INFO[persona].get('time_col', None)
            if query_col in training_df.columns and time_col in training_df.columns:
                strong_signal = (training_df[query_col] > training_df[query_col].quantile(0.9)) | \
                               (training_df[time_col] > training_df[time_col].quantile(0.9))
                training_df.loc[strong_signal, query_col] *= 1.5
                training_df.loc[strong_signal, time_col] *= 1.5
        
        if 'start_time' in training_df.columns:
            try:
                start_time = pd.to_datetime(training_df['start_time'], errors='coerce')
                training_df['recency'] = (pd.to_datetime('2025-04-25') - start_time).dt.days.fillna(30)
                training_df['time_of_day'] = start_time.dt.hour.fillna(12) // 6
                training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30 if 'userid' in training_df.columns else pd.Series([1] * len(training_df), index=training_df.index)
            except:
                training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
                training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
                training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
        else:
            training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
            training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
            training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
        
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        if all(col in training_df.columns for col in cluster_features):
            kmeans = KMeans(n_clusters=5, random_state=42)
            training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
        else:
            training_df['user_cluster'] = pd.Series([0] * len(training_df), index=training_df.index)
        
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
            for col in embedding_cols:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
        
        query_cols = [c for c in behavioral_features if c.startswith('query_') and c in training_df.columns]
        filter_cols = [c for c in behavioral_features if c.startswith('filter_') and c in training_df.columns]
        training_df['query_count'] = training_df[query_cols].sum(axis=1) if query_cols else pd.Series([0] * len(training_df), index=training_df.index)
        training_df['filter_count'] = training_df[filter_cols].sum(axis=1) if filter_cols else pd.Series([0] * len(training_df), index=training_df.index)
        
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        additional_features = []
        for persona in PERSONAS:
            persona_info = PERSONA_INFO.get(persona, {})
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'))
            filter_col = get_feature_as_series(training_df, persona_info.get('filter_col'))
            click_col = get_feature_as_series(training_df, persona_info.get('click_col', 'dummy_col'))
            time_col = get_feature_as_series(training_df, persona_info.get('time_col', 'dummy_col'))
            accordion_col = get_feature_as_series(training_df, persona_info.get('accordion_col', 'dummy_col'))
            plan_col = get_feature_as_series(training_df, persona_info.get('plan_col'))
            
            signal_weights = 3.5 if persona == 'drug' else 3.0
            training_df[f'{persona}_signal'] = (
                query_col * 2.0 +
                filter_col * 2.0 +
                time_col.clip(upper=5) * 1.5 +
                accordion_col * 1.0 +
                click_col * 2.0
            ) * signal_weights
            additional_features.append(f'{persona}_signal')
            
            has_interaction = ((query_col > 0) | (filter_col > 0) | (click_col > 0) | (accordion_col > 0))
            training_df[f'{persona}_interaction'] = safe_bool_to_int(has_interaction, training_df) * 3.0
            additional_features.append(f'{persona}_interaction')
            
            training_df[f'{persona}_primary'] = (
                safe_bool_to_int(query_col > 0, training_df) * 2.0 +
                safe_bool_to_int(filter_col > 0, training_df) * 2.0 +
                safe_bool_to_int(click_col > 0, training_df) * 2.0 +
                safe_bool_to_int(time_col > 2, training_df) * 1.5
            ) * 2.0
            additional_features.append(f'{persona}_primary')
            
            training_df[f'{persona}_plan_correlation'] = plan_col * (
                query_col + filter_col + click_col + time_col.clip(upper=3)
            ) * 2.0
            additional_features.append(f'{persona}_plan_correlation')
        
        dental_query = get_feature_as_series(training_df, 'query_dental')
        dental_filter = get_feature_as_series(training_df, 'filter_dental')
        dental_time = get_feature_as_series(training_df, 'time_dental_pages')
        dental_accordion = get_feature_as_series(training_df, 'accordion_dental')
        dental_benefit = get_feature_as_series(training_df, 'ma_dental_benefit')
        
        training_df['dental_engagement_score'] = (
            dental_query * 3.0 +
            dental_filter * 3.0 +
            dental_time.clip(upper=5) * 2.0 +
            dental_accordion * 2.0 +
            dental_benefit * 4.0
        ) * 3.0
        additional_features.append('dental_engagement_score')
        
        training_df['dental_benefit_multiplier'] = (
            (dental_query + dental_filter + dental_accordion) * 
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
        
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        dsnp_filter = get_feature_as_series(training_df, 'filter_dsnp')
        dsnp_time = get_feature_as_series(training_df, 'time_dsnp_pages')
        dsnp_accordion = get_feature_as_series(training_df, 'accordion_dsnp')
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
            dsnp_accordion * 2.0 +
            dsnp_plan * 5.0
        ) * 3.0
        additional_features.append('dsnp_engagement_score')
        
        training_df['dsnp_plan_multiplier'] = (
            (dsnp_query + dsnp_filter + dsnp_accordion) *
            (dsnp_plan + 0.5) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('dsnp_plan_multiplier')
        
        drug_query = get_feature_as_series(training_df, 'query_drug')
        drug_filter = get_feature_as_series(training_df, 'filter_drug')
        drug_time = get_feature_as_series(training_df, 'time_drug_pages')
        drug_accordion = get_feature_as_series(training_df, 'accordion_drug')
        drug_click = get_feature_as_series(training_df, 'click_drug')
        drug_benefit = get_feature_as_series(training_df, 'ma_drug_benefit')
        
        training_df['drug_engagement_score'] = (
            drug_query * 3.0 +
            drug_filter * 3.0 +
            drug_time.clip(upper=5) * 2.0 +
            drug_accordion * 2.0 +
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
            (drug_query + drug_filter + drug_click + drug_accordion) *
            (drug_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=25)
        additional_features.append('drug_benefit_boost')
        
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        csnp_filter = get_feature_as_series(training_df, 'filter_csnp')
        csnp_time = get_feature_as_series(training_df, 'time_csnp_pages')
        csnp_accordion = get_feature_as_series(training_df, 'accordion_csnp')
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
            csnp_accordion * 2.0 +
            csnp_plan * 5.0
        ) * 4.0
        additional_features.append('csnp_engagement_score')
        
        training_df['csnp_plan_multiplier'] = (
            (csnp_query + csnp_filter + csnp_accordion) *
            (csnp_plan + 0.5) * 6.0
        ).clip(lower=0, upper=24)
        additional_features.append('csnp_plan_multiplier')
        
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster', 'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        variances = X.var()
        valid_features = variances[variances > 1e-5].index.tolist()
        X = X[valid_features]
        logger.info(f"Selected features after variance filtering: {valid_features}")
        
        for persona in PERSONAS:
            num_samples = 3000 if persona in SUPER_PRIORITY_PERSONAS else 2000 if persona == 'dental' else 1500
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
        
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = power_transformer.fit_transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        
        return X, y, power_transformer
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def train_binary_persona_classifier(X_train, y_train, X_val, y_val, persona):
    y_train_binary = (y_train == persona).astype(int)
    y_val_binary = (y_val == persona).astype(int)
    
    class_weight = PERSONA_CLASS_WEIGHT.get(persona, 3.0)
    threshold = PERSONA_THRESHOLD.get(persona, 0.3)
    
    if persona in ['csnp', 'doctor']:
        iterations = 950
        depth = 8
        learning_rate = 0.015
        l2_leaf_reg = 1.5
        early_stopping = 100
    elif persona in ['drug', 'dsnp', 'vision']:  # Added vision
        iterations = 800
        depth = 7
        learning_rate = 0.02
        l2_leaf_reg = 1.8
        early_stopping = 90
    elif persona == 'dental':
        iterations = 650
        depth = 6
        learning_rate = 0.03
        l2_leaf_reg = 2.5
        early_stopping = 70
    else:
        iterations = 300
        depth = 5
        learning_rate = 0.05
        l2_leaf_reg = 3.0
        early_stopping = 50
    
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        class_weights={0: 1.0, 1: class_weight},
        l2_leaf_reg=l2_leaf_reg,
        verbose=0
    )
    
    if persona in SUPER_PRIORITY_PERSONAS:
        sample_weights = np.ones(len(y_train_binary))
        sample_weights[y_train_binary == 1] = 1.8 if persona in ['doctor', 'csnp'] else 1.5
        model.fit(
            X_train, y_train_binary,
            eval_set=(X_val, y_val_binary),
            early_stopping_rounds=early_stopping,
            sample_weight=sample_weights,
            verbose=False
        )
    else:
        model.fit(
            X_train, y_train_binary,
            eval_set=(X_val, y_val_binary),
            early_stopping_rounds=early_stopping,
            verbose=False
        )
    
    calibrated_model = CalibratedClassifierCV(model, cv='prefit')
    calibrated_model.fit(X_val, y_val_binary)
    
    return calibrated_model

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

def custom_ensemble_with_balanced_focus(predictions, binary_probas, le, weights=None, thresholds=None):
    if weights is None:
        weights = {persona: 1.0 for persona in PERSONAS}
    if thresholds is None:
        thresholds = PERSONA_THRESHOLD
    
    weighted_preds = np.copy(predictions)
    
    for i, persona in enumerate(le.classes_):
        weighted_preds[:, i] *= weights.get(persona, 1.0)
    
    if binary_probas:
        for persona, proba in binary_probas.items():
            persona_idx = np.where(le.classes_ == persona)[0][0]
            blend_ratio = 0.35 if persona in ['doctor', 'csnp'] else \
                         0.4 if persona in ['drug', 'dsnp', 'vision'] else 0.5
            weighted_preds[:, persona_idx] = blend_ratio * weighted_preds[:, persona_idx] + \
                                            (1 - blend_ratio) * proba
    
    row_sums = weighted_preds.sum(axis=1, keepdims=True)
    normalized_preds = weighted_preds / (row_sums + 1e-8)
    
    predictions = np.zeros(len(normalized_preds), dtype=np.int32)
    for i in range(len(normalized_preds)):
        max_prob = -1
        max_idx = -1
        for j, persona in enumerate(le.classes_):
            prob = normalized_preds[i, j]
            threshold = thresholds.get(persona, 0.3)
            if prob > threshold and prob > max_prob:
                max_prob = prob
                max_idx = j
        if max_idx >= 0:
            predictions[i] = max_idx
        else:
            predictions[i] = np.argmax(normalized_preds[i])
    
    return predictions, normalized_preds

def create_visualizations(X_test, y_test, y_pred, le):
    try:
        plt.figure(figsize=(12, 9))
        cm = confusion_matrix(y_test, y_pred, labels=range(len(PERSONAS)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=PERSONAS, yticklabels=PERSONAS)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        persona_acc = {}
        for persona in PERSONAS:
            mask = (y_test == le.transform([persona])[0])
            if mask.sum() > 0:
                persona_acc[persona] = accuracy_score(y_test[mask], y_pred[mask]) * 100
            else:
                persona_acc[persona] = 0
        
        acc_df = pd.DataFrame({
            'Persona': list(persona_acc.keys()),
            'Accuracy (%)': list(persona_acc.values())
        }).sort_values('Accuracy (%)', ascending=False)
        
        sns.barplot(data=acc_df, x='Persona', y='Accuracy (%)', palette='viridis')
        plt.title('Model Accuracy by Persona Type')
        plt.axhline(y=acc_df['Accuracy (%)'].mean(), color='r', linestyle='--', label='Mean Accuracy')
        plt.legend()
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nPer-Persona Accuracy:")
        for persona, acc in acc_df.set_index('Persona')['Accuracy (%)'].items():
            print(f"{persona}: {acc:.2f}%")
        
        print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

def main():
    logger.info("Starting persona model training...")
    
    # Load data
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Prepare features
    try:
        X, y, transformer = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train-validation split for binary classifiers
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train binary classifiers
    binary_classifiers = {}
    for persona in PERSONAS:
        binary_classifiers[persona] = train_binary_persona_classifier(
            X_train_main, y_train_main, X_val, y_val, persona
        )
    
    # Train main multi-class model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    
    class_weights = {i: PERSONA_CLASS_WEIGHT.get(persona, 3.0) for i, persona in enumerate(le.classes_)}
    logger.info(f"Using class weights: {class_weights}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train_encoded[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train_encoded[val_idx]
        
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            class_weights=class_weights,
            random_seed=42,
            verbose=0
        )
        
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=50
        )
        models.append(model)
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions
    y_pred_probas_multi = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    
    binary_probas = {}
    for persona, classifier in binary_classifiers.items():
        binary_probas[persona] = classifier.predict_proba(X_test)[:,1]
    
    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            blend_ratio = 0.35 if persona in ['doctor', 'csnp'] else \
                         0.4 if persona in ['drug', 'dsnp', 'vision'] else 0.5
            y_pred_probas_multi[:, i] = blend_ratio * y_pred_probas_multi[:, i] + \
                                       (1-blend_ratio) * binary_probas[persona]
    
    y_pred_indices, _ = custom_ensemble_with_balanced_focus(
        y_pred_probas_multi, binary_probas, le
    )
    y_pred = y_pred_indices
    
    # Evaluate
    overall_acc = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    per_persona_acc = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    
    logger.info(f"\nTraining Evaluation Results:")
    logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
    logger.info(f"Macro F1: {macro_f1:.2f}")
    
    test_sample_counts = pd.Series(y_test).value_counts()
    logger.info("\nPer-Persona Test Metrics:")
    logger.info(f"{'Persona':<12} {'Test Samples':<12} {'Accuracy (%)':<12}")
    logger.info("-" * 36)
    for persona in PERSONAS:
        sample_count = test_sample_counts.get(persona, 0)
        accuracy = per_persona_acc.get(persona, 0.0)
        logger.info(f"{persona:<12} {sample_count:<12} {accuracy:.2f}")
    
    if overall_acc < 0.8:
        logger.warning("Overall accuracy below 0.8. Consider increasing PERSONA_OVERSAMPLING_RATIO or PERSONA_CLASS_WEIGHT.")
    
    # Save models
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models[0], f)
        
    for persona, clf in binary_classifiers.items():
        binary_model_path = MODEL_FILE.replace('.pkl', f'_{persona}_binary.pkl')
        with open(binary_model_path, 'wb') as f:
            pickle.dump(clf, f)
    
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
        
    with open(TRANSFORMER_FILE, 'wb') as f:
        pickle.dump(transformer, f)
        
    logger.info("Saved models, label encoder, and transformer to disk.")
    
    # Visualizations
    create_visualizations(X_test, y_test_encoded, y_pred, le)

if __name__ == "__main__":
    main()
