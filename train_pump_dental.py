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
from imblearn.over_sampling import SMOTE
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
    "data/s-learning-data/models/model-persona-1.1.2.pkl"
)
LABEL_ENCODER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/label_encoder_1.pkl"
)
TRANSFORMER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/power_transformer.pkl"
)

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'csnp']

PERSONA_OVERSAMPLING_RATIO = {
    'drug': 6.0,
    'dental': 20.0,
    'doctor': 8.0,
    'dsnp': 5.0,
    'csnp': 5.0
}

PERSONA_CLASS_WEIGHT = {
    'drug': 7.0,
    'dental': 35.0,
    'doctor': 25.0,
    'dsnp': 8.0,
    'csnp': 6.0
}

PERSONA_THRESHOLD = {
    'drug': 0.20,
    'dental': 0.03,
    'doctor': 0.08,
    'dsnp': 0.25,
    'csnp': 0.25
}

HIGH_PRIORITY_PERSONAS = ['dental', 'csnp', 'doctor', 'dsnp']
SUPER_PRIORITY_PERSONAS = ['dental', 'csnp', 'doctor', 'dsnp']

PERSONA_FEATURES = {
    'dental': ['query_dental', 'filter_dental', 'time_dental_pages', 'accordion_dental', 'ma_dental_benefit'],
    'doctor': ['query_provider', 'filter_provider', 'click_provider', 'ma_provider_network'],
    'dsnp': ['query_dsnp', 'filter_dsnp', 'time_dsnp_pages', 'accordion_dsnp', 'dsnp'],
    'drug': ['query_drug', 'filter_drug', 'time_drug_pages', 'accordion_drug', 'click_drug', 'ma_drug_benefit'],
    'csnp': ['query_csnp', 'filter_csnp', 'time_csnp_pages', 'accordion_csnp', 'csnp']
}

PERSONA_INFO = {
    'dental': {
        'plan_col': 'ma_dental_benefit',
        'query_col': 'query_dental',
        'filter_col': 'filter_dental',
        'time_col': 'time_dental_pages',
        'accordion_col': 'accordion_dental'
    },
    'doctor': {
        'plan_col': 'ma_provider_network',
        'query_col': 'query_provider',
        'filter_col': 'filter_provider',
        'click_col': 'click_provider'
    },
    'dsnp': {
        'plan_col': 'dsnp',
        'query_col': 'query_dsnp',
        'filter_col': 'filter_dsnp',
        'time_col': 'time_dsnp_pages',
        'accordion_col': 'accordion_dsnp'
    },
    'drug': {
        'plan_col': 'ma_drug_benefit',
        'query_col': 'query_drug',
        'filter_col': 'filter_drug',
        'click_col': 'click_drug',
        'time_col': 'time_drug_pages',
        'accordion_col': 'accordion_drug'
    },
    'csnp': {
        'plan_col': 'csnp',
        'query_col': 'query_csnp',
        'filter_col': 'filter_csnp',
        'time_col': 'time_csnp_pages',
        'accordion_col': 'accordion_csnp'
    }
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
    logger.debug(f"Column {col_name} missing, using default {default}")
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
        
        row_copy = row.copy()
        if valid_found:
            row_copy['persona'] = valid_found[0]
        else:
            invalid_personas.update(personas)
            row_copy['persona'] = 'dental'
        new_rows.append(row_copy)
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(f"Rows after persona normalization: {len(result)}")
    if invalid_personas:
        logger.info(f"Invalid personas found: {invalid_personas}")
    if result.empty:
        logger.warning(f"No valid personas found. Valid personas: {valid_personas}")
    return result

def calculate_persona_weight(row, persona_info, persona):
    query_col = persona_info.get('query_col')
    filter_col = persona_info.get('filter_col')
    plan_col = persona_info.get('plan_col')
    click_col = persona_info.get('click_col')
    
    query_value = row.get(query_col, 0) if query_col and pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if filter_col and pd.notna(row.get(filter_col, np.nan)) else 0
    plan_value = row.get(plan_col, 0) if plan_col and pd.notna(row.get(plan_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    max_val = max([query_value, filter_value, plan_value, click_value, 1])
    weight = 0.25 * (query_value/max_val + filter_value/max_val + plan_value/max_val + click_value/max_val)
    return min(max(weight, 0), 1.0)

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw persona distribution:\n{pd.Series(behavioral_df['persona']).value_counts().to_string()}")
        
        required_cols = ['persona', 'zip', 'plan_id']
        missing_cols = [col for col in required_cols if col not in behavioral_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in behavioral_df: {missing_cols}")
        
        persona_mapping = {'fitness': 'otc', 'hdental': 'dental'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        behavioral_df['zip'] = get_feature_as_series(behavioral_df, 'zip', 'unknown').astype(str)
        behavioral_df['plan_id'] = get_feature_as_series(behavioral_df, 'plan_id', 'unknown').astype(str)
        behavioral_df['total_session_time'] = get_feature_as_series(behavioral_df, 'total_session_time', 0).fillna(0)
        
        for col in ['query_dental', 'query_provider', 'query_drug']:
            if col in behavioral_df.columns:
                logger.info(f"{col} stats: mean={behavioral_df[col].mean():.2f}, std={behavioral_df[col].std():.2f}, missing={behavioral_df[col].isna().sum()}, non-zero={len(behavioral_df[behavioral_df[col] > 0])}")
            else:
                logger.warning(f"Key feature {col} missing in behavioral_df")
        
        plan_df = pd.read_csv(plan_path)
        plan_df['zip'] = get_feature_as_series(plan_df, 'zip', 'unknown').astype(str)
        plan_df['plan_id'] = get_feature_as_series(plan_df, 'plan_id', 'unknown').astype(str)
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def generate_synthetic_persona_examples(X, feature_columns, persona, num_samples=1000, real_data_stats=None):
    synthetic_examples = []
    persona_features = [col for col in feature_columns if persona in col.lower()]
    specific_features = PERSONA_FEATURES.get(persona, [])
    
    if persona == 'dental':
        num_samples = 10000
    elif persona == 'doctor':
        num_samples = 6000
    elif persona == 'dsnp':
        num_samples = 3500
    elif persona == 'drug':
        num_samples = 4000
    elif persona == 'csnp':
        num_samples = 3500
    
    for _ in range(num_samples):
        sample = {col: 0 for col in feature_columns}
        
        if 'recency' in feature_columns:
            sample['recency'] = np.random.randint(1, 30)
        if 'visit_frequency' in feature_columns:
            sample['visit_frequency'] = np.random.uniform(0.2, 0.8)
        if 'time_of_day' in feature_columns:
            sample['time_of_day'] = np.random.randint(0, 4)
        if 'user_cluster' in feature_columns:
            sample['user_cluster'] = np.random.randint(0, 5)
            
        for feature in persona_features:
            if real_data_stats and feature in real_data_stats:
                mean, std = real_data_stats[feature]['mean'], real_data_stats[feature]['std']
                sample[feature] = max(0, np.random.normal(mean, std * 0.7))
            else:
                sample[feature] = np.random.uniform(15.0, 25.0) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(4.0, 8.0)
            
        for feature in specific_features:
            if feature in feature_columns:
                if real_data_stats and feature in real_data_stats:
                    mean, std = real_data_stats[feature]['mean'], real_data_stats[feature]['std']
                    sample[feature] = max(0, np.random.normal(mean, std * 0.7))
                else:
                    sample[feature] = np.random.uniform(15.0, 25.0) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(5.0, 10.0)
        
        plan_col = PERSONA_INFO.get(persona, {}).get('plan_col')
        if plan_col and plan_col in feature_columns:
            sample[plan_col] = 1
            
        for other_persona in PERSONAS:
            if other_persona != persona:
                other_features = [col for col in feature_columns if other_persona in col.lower()]
                for feature in other_features:
                    sample[feature] = np.random.uniform(0.0, 0.02) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(0.0, 0.1)
                    
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
        
        plan_features = ['ma_dental_benefit', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network']
        for col in plan_features:
            training_df[col] = get_feature_as_series(training_df, col, 0).fillna(0)
        
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_dsnp'
        ]
        
        sparse_features = ['query_dental', 'time_dental_pages', 'query_drug', 'query_provider']
        imputer_median = SimpleImputer(strategy='median')
        imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
        
        for col in behavioral_features:
            if col in training_df.columns:
                if col in sparse_features:
                    training_df[col] = imputer_median.fit_transform(training_df[[col]]).flatten()
                else:
                    training_df[col] = imputer_median.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
                logger.debug(f"Created missing column {col} with default value 0")
        
        for persona in ['dental', 'csnp', 'doctor', 'dsnp']:
            query_col = PERSONA_INFO[persona]['query_col']
            time_col = PERSONA_INFO[persona].get('time_col')
            if query_col in training_df.columns and time_col in training_df.columns:
                strong_signal = (training_df[query_col] > training_df[query_col].quantile(0.85)) | \
                               (training_df[time_col] > training_df[time_col].quantile(0.85))
                training_df.loc[strong_signal, query_col] *= 3.5
                training_df.loc[strong_signal, time_col] *= 3.5
        
        if 'start_time' in training_df.columns:
            try:
                start_time = pd.to_datetime(training_df['start_time'], errors='coerce')
                training_df['recency'] = (pd.to_datetime('2025-05-29') - start_time).dt.days.fillna(30)
                training_df['time_of_day'] = start_time.dt.hour.fillna(12) // 6
                training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30 if 'userid' in training_df.columns else pd.Series([1] * len(training_df), index=training_df.index)
            except Exception as e:
                logger.warning(f"Failed to process start_time: {e}")
                training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
                training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
                training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
        else:
            training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
            training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
            training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
        
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
            w2v_model = Word2Vec(sentences=plan_sentences, vector_size=8, window=5, min_count=1, workers=4)
            plan_embeddings = training_df['plan_id'].apply(
                lambda x: w2v_model.wv[x] if x in w2v_model.wv else np.zeros(8)
            )
            embedding_cols = [f'plan_emb_{i}' for i in range(8)]
            training_df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=training_df.index)
        else:
            embedding_cols = [f'plan_emb_{i}' for i in range(8)]
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
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'), 0)
            filter_col = get_feature_as_series(training_df, persona_info.get('filter_col'), 0)
            click_col = get_feature_as_series(training_df, persona_info.get('click_col', 'dummy_col'), 0)
            time_col = get_feature_as_series(training_df, persona_info.get('time_col', 'dummy_col'), 0)
            accordion_col = get_feature_as_series(training_df, persona_info.get('accordion_col', 'dummy_col'), 0)
            plan_col = get_feature_as_series(training_df, persona_info.get('plan_col'), 0)
            
            signal_weights = 7.0 if persona in HIGH_PRIORITY_PERSONAS else 4.5
            training_df[f'{persona}_signal'] = (
                query_col * 4.0 +
                filter_col * 4.0 +
                time_col.clip(upper=5) * 3.5 +
                accordion_col * 3.0 +
                click_col * 4.0
            ) * signal_weights
            additional_features.append(f'{persona}_signal')
            
            has_interaction = ((query_col > 0) | (filter_col > 0) | (click_col > 0) | (accordion_col > 0))
            training_df[f'{persona}_interaction'] = safe_bool_to_int(has_interaction, training_df) * 7.0
            additional_features.append(f'{persona}_interaction')
            
            training_df[f'{persona}_primary'] = (
                safe_bool_to_int(query_col > 0, training_df) * 6.0 +
                safe_bool_to_int(filter_col > 0, training_df) * 6.0 +
                safe_bool_to_int(click_col > 0, training_df) * 6.0 +
                safe_bool_to_int(time_col > 2, training_df) * 3.5
            ) * 6.0
            additional_features.append(f'{persona}_primary')
            
            training_df[f'{persona}_plan_correlation'] = plan_col * (
                query_col + filter_col + click_col + time_col.clip(upper=3)
            ) * 6.0
            additional_features.append(f'{persona}_plan_correlation')
        
        dental_query = get_feature_as_series(training_df, 'query_dental', 0)
        dental_filter = get_feature_as_series(training_df, 'filter_dental', 0)
        dental_time = get_feature_as_series(training_df, 'time_dental_pages', 0)
        dental_accordion = get_feature_as_series(training_df, 'accordion_dental', 0)
        dental_benefit = get_feature_as_series(training_df, 'ma_dental_benefit', 0)
        
        training_df['dental_engagement_score'] = (
            dental_query * 12.0 +
            dental_filter * 12.0 +
            dental_time.clip(upper=5) * 10.0 +
            dental_accordion * 8.0 +
            dental_benefit * 15.0
        ) * 10.0
        additional_features.append('dental_engagement_score')
        
        training_df['dental_benefit_multiplier'] = (
            (dental_query + dental_filter + dental_accordion) *
            (dental_benefit + 0.5) * 12.0
        ).clip(lower=0, upper=45)
        additional_features.append('dental_benefit_multiplier')
        
        training_df['dental_specificity'] = (
            dental_query * 10.0 -
            (training_df.get('query_drug', 0) + training_df.get('query_dsnp', 0) + training_df.get('query_csnp', 0)) * 2.5
        ).clip(lower=0) * 10.0
        additional_features.append('dental_specificity')
        
        training_df['dental_combined_signal'] = (
            (dental_query > 0).astype(int) +
            (dental_filter > 0).astype(int) +
            (dental_accordion > 0).astype(int)
        ) * 10.0
        additional_features.append('dental_combined_signal')
        
        provider_query = get_feature_as_series(training_df, 'query_provider', 0)
        provider_filter = get_feature_as_series(training_df, 'filter_provider', 0)
        provider_click = get_feature_as_series(training_df, 'click_provider', 0)
        provider_network = get_feature_as_series(training_df, 'ma_provider_network', 0)
        
        training_df['doctor_interaction_score'] = (
            provider_query * 10.0 +
            provider_filter * 10.0 +
            provider_click * 20.0 +
            provider_network * 12.0
        ) * 8.0
        additional_features.append('doctor_interaction_score')
        
        training_df['doctor_specificity'] = (
            provider_query * 10.0 -
            (training_df.get('query_dental', 0) + training_df.get('query_drug', 0) + training_df.get('query_dsnp', 0) + training_df.get('query_csnp', 0)) * 2.5
        ).clip(lower=0) * 8.0
        additional_features.append('doctor_specificity')
        
        training_df['doctor_query_specificity'] = (
            provider_query /
            (training_df[['query_dental', 'query_drug', 'query_dsnp', 'query_csnp']].sum(axis=1).clip(lower=1e-6))
        ).clip(upper=1.0) * 8.0
        additional_features.append('doctor_query_specificity')
        
        training_df['doctor_network_boost'] = (
            (provider_query + provider_filter + provider_click) *
            (provider_network + 0.5) * 15.0
        ).clip(lower=0, upper=50)
        additional_features.append('doctor_network_boost')
        
        training_df['doctor_exclusive_signal'] = (
            (provider_click > 0).astype(int) * 12.0 +
            (provider_network > 0).astype(int) * 12.0 -
            (training_df.get('query_dental', 0) + training_df.get('query_drug', 0) + training_df.get('query_dsnp', 0) + training_df.get('query_csnp', 0)) * 2.5
        ).clip(lower=0) * 8.0
        additional_features.append('doctor_exclusive_signal')
        
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp', 0)
        dsnp_filter = get_feature_as_series(training_df, 'filter_dsnp', 0)
        dsnp_time = get_feature_as_series(training_df, 'time_dsnp_pages', 0)
        dsnp_accordion = get_feature_as_series(training_df, 'accordion_dsnp', 0)
        dsnp_plan = get_feature_as_series(training_df, 'dsnp', 0)
        csnp_query = get_feature_as_series(training_df, 'query_csnp', 0)
        
        training_df['dsnp_csnp_ratio'] = (
            (dsnp_query + 0.8) / (csnp_query + dsnp_query + 1e-6)
        ).clip(0, 1) * 5.0
        additional_features.append('dsnp_csnp_ratio')
        
        training_df['dsnp_engagement_score'] = (
            dsnp_query * 7.0 +
            dsnp_filter * 7.0 +
            dsnp_time.clip(upper=5) * 6.0 +
            dsnp_accordion * 5.0 +
            dsnp_plan * 9.0
        ) * 7.0
        additional_features.append('dsnp_engagement_score')
        
        training_df['dsnp_plan_multiplier'] = (
            (dsnp_query + dsnp_filter + dsnp_accordion) *
            (dsnp_plan + 0.5) * 7.0
        ).clip(lower=0, upper=35)
        additional_features.append('dsnp_plan_multiplier')
        
        training_df['dsnp_specificity'] = (
            dsnp_query * 6.0 -
            (training_df.get('query_dental', 0) + training_df.get('query_drug', 0) + training_df.get('query_csnp', 0)) * 1.5
        ).clip(lower=0) * 6.0
        additional_features.append('dsnp_specificity')
        
        training_df['dsnp_combined_signal'] = (
            (dsnp_query > 0).astype(int) +
            (dsnp_filter > 0).astype(int) +
            (dsnp_accordion > 0).astype(int)
        ) * 6.0
        additional_features.append('dsnp_combined_signal')
        
        drug_query = get_feature_as_series(training_df, 'query_drug', 0)
        drug_filter = get_feature_as_series(training_df, 'filter_drug', 0)
        drug_time = get_feature_as_series(training_df, 'time_drug_pages', 0)
        drug_accordion = get_feature_as_series(training_df, 'accordion_drug', 0)
        drug_click = get_feature_as_series(training_df, 'click_drug', 0)
        drug_benefit = get_feature_as_series(training_df, 'ma_drug_benefit', 0)
        
        training_df['drug_engagement_score'] = (
            drug_query * 4.0 +
            drug_filter * 4.0 +
            drug_time.clip(upper=5) * 2.5 +
            drug_accordion * 2.5 +
            drug_click * 5.0 +
            drug_benefit * 5.0
        ) * 4.5
        additional_features.append('drug_engagement_score')
        
        training_df['drug_interest_ratio'] = (
            (drug_query + drug_filter) /
            (training_df.get('query_count', 1) + training_df.get('filter_count', 1) + 1e-6)
        ).clip(upper=0.9) * 12.0
        additional_features.append('drug_interest_ratio')
        
        training_df['drug_benefit_boost'] = (
            (drug_query + drug_filter + drug_click + drug_accordion) *
            (drug_benefit + 0.5) * 6.0
        ).clip(lower=0, upper=30)
        additional_features.append('drug_benefit_boost')
        
        csnp_query = get_feature_as_series(training_df, 'query_csnp', 0)
        csnp_filter = get_feature_as_series(training_df, 'filter_csnp', 0)
        csnp_time = get_feature_as_series(training_df, 'time_csnp_pages', 0)
        csnp_accordion = get_feature_as_series(training_df, 'accordion_csnp', 0)
        csnp_plan = get_feature_as_series(training_df, 'csnp', 0)
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp', 0)
        
        training_df['csnp_dsnp_ratio'] = (
            (csnp_query + 0.8) / (dsnp_query + csnp_query + 1e-6)
        ).clip(0, 1) * 4.0
        additional_features.append('csnp_dsnp_ratio')
        
        training_df['csnp_specificity'] = (
            csnp_query * 3.5 -
            (training_df.get('query_dental', 0) + training_df.get('query_drug', 0)) * 0.7
        ).clip(lower=0) * 3.5
        additional_features.append('csnp_specificity')
        
        training_df['csnp_engagement_score'] = (
            csnp_query * 3.5 +
            csnp_filter * 3.5 +
            csnp_time.clip(upper=5) * 2.5 +
            csnp_accordion * 1.5 +
            csnp_plan * 4.0
        ) * 3.5
        additional_features.append('csnp_engagement_score')
        
        training_df['csnp_plan_multiplier'] = (
            (csnp_query + csnp_filter + csnp_accordion) *
            (csnp_plan + 0.0) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('csnp_plan_multiplier')
        
        training_df['dental_doctor_interaction'] = (dental_query * provider_query).clip(upper=10) * 6.0
        additional_features.append('dental_doctor_interaction')
        
        training_df['dental_dsnp_ratio'] = (
            (dental_query + 0.8) / (dsnp_query + dental_query + 1e-6)
        ).clip(0, 1) * 6.0
        additional_features.append('dental_dsnp_ratio')
        
        training_df['dental_drug_ratio'] = (
            (dental_query + 0.8) / (drug_query + dental_query + 1e-6)
        ).clip(0, 1) * 6.0
        additional_features.append('dental_drug_ratio')
        
        training_df['drug_dental_ratio'] = (
            (drug_query + 0.8) / (dental_query + drug_query + 1e-6)
        ).clip(0, 1) * 6.0
        additional_features.append('drug_dental_ratio')
        
        feature_columns = (
            behavioral_features +
            plan_features +
            additional_features +
            ['recency', 'visit_frequency', 'time_of_day', 'user_cluster', 'dental_time_ratio', 'click_ratio'] +
            embedding_cols +
            [f'{persona}_weight' for persona in PERSONAS]
        )
        feature_columns = [
            col for col in feature_columns
            if not (
                col.lower().startswith('vision') or
                col.lower().endswith('_vision') or
                col in ['query_vision', 'filter_vision', 'ma_vision', 'vision_signal',
                        'vision_interaction', 'vision_primary', 'vision_plan_correlation']
            )
        ]
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        real_data_stats = {}
        for col in feature_columns:
            real_data_stats[col] = {'mean': X[col].mean(), 'std': X[col].std()}
            if col in ['dental_engagement_score', 'doctor_interaction_score', 'drug_engagement_score']:
                logger.info(f"{col} stats: mean={X[col].mean():.2f}, std={X[col].std():.2f}")
        
        initial_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0)
        initial_model.fit(X, LabelEncoder().fit_transform(y))
        feature_importance = pd.Series(initial_model.get_feature_importance(), index=X.columns)
        top_features = feature_importance.nlargest(int(len(feature_columns) * 0.85)).index.tolist()
        X = X[top_features]
        logger.info(f"Selected top {len(top_features)} features: {top_features[:10]}...")
        
        for persona in PERSONAS:
            synthetic_examples = generate_synthetic_persona_examples(X, X.columns, persona, real_data_stats=real_data_stats)
            X = pd.concat([X, synthetic_examples], ignore_index=True)
            y = pd.concat([y, pd.Series([persona] * len(synthetic_examples))], ignore_index=True)
            logger.info(f"After adding synthetic {persona} samples: {Counter(y)}")
        
        logger.info(f"Pre-SMOTE class distribution: {Counter(y)}")
        class_counts = pd.Series(y).value_counts()
        sampling_strategy = {
            persona: int(count * PERSONA_OVERSAMPLING_RATIO.get(persona, 2.0))
            for persona, count in class_counts.items()
        }
        logger.info(f"SMOTE sampling strategy: {sampling_strategy}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
        X, y = smote.fit_resample(X, y)
        logger.info(f"Rows after SMOTE: {len(X)}")
        logger.info(f"Post-SMOTE persona distribution: {Counter(y)}")
        
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = power_transformer.fit_transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        
        return X, y, power_transformer
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def train_binary_persona_classifier(X_train, y_train, X_val, y_val, persona):
    try:
        y_train_binary = (y_train == persona).astype(int)
        y_val_binary = (y_val == persona).astype(int)
        
        class_weight = PERSONA_CLASS_WEIGHT.get(persona, 3.0)
        
        if persona == 'dental':
            iterations = 2500
            depth = 13
            learning_rate = 0.005
            l2_leaf_reg = 1.5
            early_stopping = 250
        elif persona in ['csnp', 'doctor', 'dsnp']:
            iterations = 2000
            depth = 12
            learning_rate = 0.006
            l2_leaf_reg = 1.0
            early_stopping = 250
        else:
            iterations = 1000
            depth = 8
            learning_rate = 0.015
            l2_leaf_reg = 1.5
            early_stopping = 100
        
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            class_weights={0: 1.0, 1: class_weight},
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=0.6,
            verbose=0,
            task_type='GPU' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'CPU'
        )
        
        sample_weights = np.ones(len(y_train_binary))
        sample_weights[y_train_binary == 1] = 6.0 if persona in ['dental', 'drug'] else 5.0
        model.fit(
            X_train, y_train_binary,
            eval_set=(X_val, y_val_binary),
            early_stopping_rounds=early_stopping,
            sample_weight=sample_weights,
            verbose=False
        )
        
        calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrated_model.fit(X_val, y_val_binary)
        
        feature_importance = model.get_feature_importance()
        top_features = [X_train.columns[i] for i in np.argsort(feature_importance)[-10:]]
        logger.info(f"Top features for {persona} binary classifier: {top_features}")
        
        return calibrated_model
    except Exception as e:
        logger.error(f"Failed to train binary classifier for {persona}: {e}")
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
            logger.warning(f"No true samples for {cls_name} in test set")
    return per_persona_accuracy

def custom_ensemble_with_balanced_focus(predictions, binary_probas, le, weights=None, thresholds=None):
    try:
        if weights is None:
            weights = {
                'drug': 1.2,
                'dental': 3.5,
                'doctor': 3.0,
                'dsnp': 1.5,
                'csnp': 0.7
            }
        if thresholds is None:
            thresholds = PERSONA_THRESHOLD
        
        weighted_preds = np.copy(predictions)
        
        for i, persona in enumerate(le.classes_):
            weighted_preds[:, i] *= weights.get(persona, 1.0)
        
        if binary_probas:
            for persona, proba in binary_probas.items():
                if persona in le.classes_:
                    persona_idx = np.where(le.classes_ == persona)[0][0]
                    blend_ratio = 0.98 if persona == 'dental' else 0.95 if persona == 'drug' else 0.4
                    weighted_preds[:, persona_idx] = blend_ratio * proba + \
                                                    (1 - blend_ratio) * weighted_preds[:, persona_idx]
        
        row_sums = weighted_preds.sum(axis=1, keepdims=True)
        normalized_preds = weighted_preds / (row_sums + 1e-6)
        
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
    except Exception as e:
        logger.error(f"Failed in ensemble: {e}")
        raise

def create_visualizations(X_test, y_test, y_pred, le):
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred, labels=range(len(PERSONAS)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Normalized Confusion Matrix:\n{cm_normalized.round(2)}")
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=PERSONAS, yticklabels=PERSONAS)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Prediction')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        persona_stats = {}
        test_counts = {}
        for persona in PERSONAS:
            mask = (y_test == le.transform([persona])[0])
            test_counts[persona] = mask.sum()
            if mask.sum() > 0:
                persona_stats[persona] = accuracy_score(y_test[mask], y_pred[mask]) * 100
            else:
                persona_stats[persona] = 0
                logger.warning(f"No test samples for {persona}")
        
        logger.info(f"Test set persona counts: {test_counts}")
        
        acc_df = pd.DataFrame({
            'Persona': list(persona_stats.keys()),
            'Accuracy (%)': list(persona_stats.values())
        }).sort_values(by='Accuracy (%)', ascending=False)
        
        sns.barplot(data=acc_df, x='Persona', y='Accuracy (%)', palette='deep')
        plt.title('Accuracy by Persona')
        plt.axhline(y=acc_df['Accuracy (%)'].mean(), color='red', linestyle='--', label='Mean Accuracy')
        plt.legend()
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        logger.info("\nPer-Persona Accuracy:")
        for persona, acc in acc_df.set_index('Persona')['Accuracy (%)'].items():
            logger.info(f"{persona}: {acc:.2f}%")
        
        logger.info(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def main():
    logger.info("Starting optimized persona model training...")
    
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    try:
        X, y, transformer = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    binary_classifiers = {}
    for persona in PERSONAS:
        try:
            binary_classifiers[persona] = train_binary_persona_classifier(
                X_train_main, y_train_main, X_val, y_val, persona
            )
        except Exception as e:
            logger.error(f"Failed to train binary classifier for {persona}: {e}")
            return
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    
    class_weights = {i: PERSONA_CLASS_WEIGHT.get(persona, 3.0) for i, persona in enumerate(le.classes_)}
    logger.info(f"Class weights: {class_weights}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        try:
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train_encoded[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train_encoded[val_idx]
            
            model = CatBoostClassifier(
                iterations=1800,
                depth=8,
                learning_rate=0.02,
                l2_leaf_reg=1.5,
                loss_function='MultiClass',
                class_weights=class_weights,
                random_state=42+fold,
                bagging_temperature=0.6,
                verbose=0,
                task_type='GPU' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'CPU'
            )
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=100
            )
            models.append(model)
            logger.info(f"Fold {fold+1} training completed")
        except Exception as e:
            logger.error(f"Failed to train fold {fold+1}: {e}")
            return
    
    y_pred_proba = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    
    binary_probas = {}
    for persona, classifier in binary_classifiers.items():
        try:
            binary_probas[persona] = classifier.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.error(f"Failed to get probabilities for {persona}: {e}")
            return
    
    try:
        y_pred, normalized_probs = custom_ensemble_with_balanced_focus(
            y_pred_proba, binary_probas, le
        )
    except Exception as e:
        logger.error(f"Failed to ensemble predictions: {e}")
        return
    
    overall_acc = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    per_persona_acc = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    
    logger.info(f"\nTraining Results:")
    logger.info(f"Total samples evaluated: {len(y_test)}")
    logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    logger.info("\nPer-Persona Accuracy:")
    for persona, acc in per_persona_acc.items():
        logger.info(f"  {persona}: {acc:.2f}%")
    
    feature_importance = models[0].get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    logger.info(f"Top 10 features overall:\n{importance_df.head(10).to_string()}")
    
    if overall_acc < 0.80:
        logger.warning("Overall accuracy below 80%. Check data distribution and feature importance.")
    if per_persona_acc.get('dental', 0) < 30:
        logger.warning("Dental accuracy below 30%. Check data and features.")
    
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(models[0], f)
        for persona, clf in binary_classifiers.items():
            binary_model_path = MODEL_FILE.replace('.pkl', f'_binary_{persona}.pkl')
            with open(binary_model_path, 'wb') as f:
                pickle.dump(clf, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(le, f)
        with open(TRANSFORMER_FILE, 'wb') as f:
            pickle.dump(transformer, f)
        logger.info("Saved models and label encoder.")
    except Exception as e:
        logger.error(f"Failed to save models: {e}")
        return
    
    try:
        create_visualizations(X_test, y_test_encoded, y_pred, le)
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")

if __name__ == "__main__":
    main()
