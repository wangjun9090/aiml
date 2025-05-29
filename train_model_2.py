import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from xgboost import XGBClassifier, callback
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
import sys

# --- Configuration ---
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025_clean.csv'
PLAN_FILE = "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv"
MODEL_FILE = "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.2.0.pkl"
LABEL_ENCODER_FILE = "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl"
TRANSFORMER_FILE = "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/power_transformer.pkl"
FEATURE_NAMES_FILE = "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/feature_names.pkl"

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'csnp']

PERSONA_CLASS_WEIGHT = {
    'dental': 30.0,
    'doctor': 20.0,
    'drug': 20.0,
    'dsnp': 30.0,
    'csnp': 25.0
}

PERSONA_THRESHOLD = {
    'dental': 0.15,
    'doctor': 0.20,
    'drug': 0.25,
    'dsnp': 0.23,
    'csnp': 0.20
}

HIGH_PRIORITY_PERSONAS = ['dental', 'csnp', 'doctor']

PERSONA_FEATURES = {
    'dental': ['query_dental', 'filter_dental', 'time_dental_pages', 'accordion_dental', 'ma_dental_benefit'],
    'doctor': ['query_provider', 'filter_provider', 'click_provider', 'ma_provider_network'],
    'dsnp': ['query_dsnp', 'filter_dsnp', 'time_dsnp_pages', 'accordion_dsnp', 'dsnp'],
    'drug': ['query_drug', 'filter_drug', 'time_drug_pages', 'accordion_drug', 'click_drug', 'ma_drug_benefit'],
    'csnp': ['query_csnp', 'filter_csnp', 'time_csnp_pages', 'accordion_csnp', 'csnp']
}

PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp', 'time_col': 'time_csnp_pages', 'accordion_col': 'accordion_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental', 'time_col': 'time_dental_pages', 'accordion_col': 'accordion_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp', 'time_col': 'time_dsnp_pages', 'accordion_col': 'accordion_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug', 'time_col': 'time_drug_pages', 'accordion_col': 'accordion_drug'}
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
    try:
        if col_name in df.columns:
            return df[col_name]
        logger.debug(f"Column {col_name} missing, using default {default}")
        return pd.Series([default] * len(df), index=df.index)
    except Exception as e:
        logger.error(f"Error in get_feature_as_series for {col_name}: {e}")
        raise

def normalize_persona(df):
    try:
        valid_personas = PERSONAS
        new_rows = []
        invalid_personas = set()
        
        if 'persona' not in df.columns:
            logger.error("Column 'persona' missing in behavioral_df")
            return pd.DataFrame()
        
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
    except Exception as e:
        logger.error(f"Error in normalize_persona: {e}")
        raise

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
        if not os.path.exists(behavioral_path):
            raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}, columns: {list(behavioral_df.columns)}")
        logger.info(f"Raw persona distribution:\n{pd.Series(behavioral_df['persona']).value_counts().to_string()}")
        
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        vision_samples = behavioral_df[behavioral_df['persona'].str.contains('vision', na=False)]
        if len(vision_samples) > 0:
            logger.info(f"Removed {len(vision_samples)} vision persona samples")
            behavioral_df = behavioral_df[~behavioral_df['persona'].str.contains('vision', na=False)]
        
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip().fillna('0')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip().fillna('0')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        
        for col in ['persona', 'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp']:
            if col in behavioral_df.columns:
                if col.startswith('query_'):
                    logger.info(f"{col} stats: mean={behavioral_df[col].mean():.2f}, std={behavioral_df[col].std():.2f}, missing={behavioral_df[col].isna().sum()}, non-zero={len(behavioral_df[behavioral_df[col] > 0])}")
                else:
                    logger.info(f"{col} values: {behavioral_df[col].value_counts().to_dict()}")
            else:
                logger.warning(f"Key column {col} missing in behavioral_df")
        
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan data rows: {len(plan_df)}, columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip().fillna('0')
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip().fillna('0')
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def generate_synthetic_persona_examples(X, feature_columns, persona, num_samples=1000, real_data_stats=None):
    synthetic_examples = []
    persona_features = [col for col in feature_columns if persona in col.lower()]
    specific_features = PERSONA_FEATURES.get(persona, [])
    
    if persona in ['dental', 'csnp', 'doctor']:
        num_samples = 6000
    else:
        num_samples = 3000
    
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
                sample[feature] = np.random.normal(mean, std) if std > 0 else mean
                sample[feature] = max(0, sample[feature])
            else:
                sample[feature] = np.random.uniform(6.0, 12.0) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(4.0, 8.0)
            
        for feature in specific_features:
            if feature in feature_columns:
                if real_data_stats and feature in real_data_stats:
                    mean, std = real_data_stats[feature]['mean'], real_data_stats[feature]['std']
                    sample[feature] = np.random.normal(mean, std) if std > 0 else mean
                    sample[feature] = max(0, sample[feature])
                else:
                    sample[feature] = np.random.uniform(8.0, 15.0) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(5.0, 10.0)
        
        plan_col = PERSONA_INFO.get(persona, {}).get('plan_col')
        if plan_col and plan_col in feature_columns:
            sample[plan_col] = 1
            
        for other_persona in PERSONAS:
            if other_persona != persona:
                other_features = [col for col in feature_columns if other_persona in col.lower()]
                for feature in other_features:
                    sample[feature] = np.random.uniform(0.0, 0.1) if persona in HIGH_PRIORITY_PERSONAS else np.random.uniform(0.0, 0.3)
                    
        synthetic_examples.append(sample)
    
    synthetic_df = pd.DataFrame(synthetic_examples)
    logger.info(f"Generated {len(synthetic_examples)} synthetic {persona} examples")
    return synthetic_df

def prepare_features(behavioral_df, plan_df):
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
            logger.info(f"Rows after merge: {len(training_df)}, columns: {list(training_df.columns)}")
        
        if training_df.empty:
            logger.error("Training_df is empty after merge")
            raise ValueError("Empty training DataFrame after merge")
        
        # Raw features
        raw_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_dsnp',
            'ma_dental_benefit', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network'
        ]
        
        imputer_median = SimpleImputer(strategy='median')
        imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
        
        for col in raw_features:
            if col in training_df.columns:
                logger.info(f"Processing column {col}, non-null count: {training_df[col].notna().sum()}")
                if training_df[col].notna().sum() == 0:
                    logger.warning(f"Column {col} is all NaN, filling with 0")
                    training_df[col] = 0
                else:
                    try:
                        if col.startswith('query_') or col.startswith('time_'):
                            training_df[col] = np.clip(training_df[col], 0, 1e6)
                            transformed = imputer_zero.fit_transform(training_df[[col]])
                            training_df[col] = transformed.flatten()
                        else:
                            training_df[col] = np.clip(training_df[col], 0, 1e6)
                            transformed = imputer_median.fit_transform(training_df[[col]])
                            training_df[col] = transformed.flatten()
                    except Exception as e:
                        logger.error(f"Imputation failed for {col}: {e}")
                        training_df[col] = 0
            else:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
                logger.debug(f"Created column {col} with default value 0")
        
        # Feature engineering
        if 'start_time' in training_df.columns:
            try:
                start_time = pd.to_datetime(training_df['start_time'], errors='coerce')
                training_df['recency'] = (pd.to_datetime('2025-05-29') - start_time).dt.days.fillna(30)
                training_df['time_of_day'] = start_time.dt.hour.fillna(12) // 6
                if 'userid' in training_df.columns:
                    training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
                else:
                    training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
            except Exception as e:
                logger.warning(f"Failed to process start_time: {e}")
                training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
                training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
                training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
        else:
            training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
            training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
            training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
        
        # Clustering
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        if all(col in training_df.columns for col in cluster_features):
            kmeans = KMeans(n_clusters=5, random_state=42)
            training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
        else:
            training_df['user_cluster'] = pd.Series([0] * len(training_df), index=training_df.index)
        
        # Word2Vec embeddings for plan_id
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
        
        # Persona-specific weights
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        # Additional features
        additional_features = []
        for persona in PERSONAS:
            persona_info = PERSONA_INFO.get(persona, {})
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'))
            filter_col = get_feature_as_series(training_df, persona_info.get('filter_col'))
            click_col = get_feature_as_series(training_df, persona_info.get('click_col', 'dummy_col'))
            time_col = get_feature_as_series(training_df, persona_info.get('time_col', 'dummy_col'))
            accordion_col = get_feature_as_series(training_df, persona_info.get('accordion_col', 'dummy_col'))
            plan_col = get_feature_as_series(training_df, persona_info.get('plan_col'))
            
            signal_weights = 4.0 if persona in HIGH_PRIORITY_PERSONAS else 3.5
            training_df[f'{persona}_signal'] = (
                query_col * 2.5 +
                filter_col * 2.5 +
                time_col.clip(upper=5) * 2.0 +
                accordion_col * 1.5 +
                click_col * 2.5
            ) * signal_weights
            additional_features.append(f'{persona}_signal')
            
            has_interaction = ((query_col > 0) | (filter_col > 0) | (click_col > 0) | (accordion_col > 0))
            training_df[f'{persona}_interaction'] = safe_bool_to_int(has_interaction, training_df) * 4.0
            additional_features.append(f'{persona}_interaction')
            
            training_df[f'{persona}_primary'] = (
                safe_bool_to_int(query_col > 0, training_df) * 3.0 +
                safe_bool_to_int(filter_col > 0, training_df) * 3.0 +
                safe_bool_to_int(click_col > 0, training_df) * 3.0 +
                safe_bool_to_int(time_col > 2, training_df) * 2.0
            ) * 3.0
            additional_features.append(f'{persona}_primary')
            
            training_df[f'{persona}_plan_correlation'] = plan_col * (
                query_col + filter_col + click_col + time_col.clip(upper=3)
            ) * 3.0
            additional_features.append(f'{persona}_plan_correlation')
        
        dental_query = get_feature_as_series(training_df, 'query_dental')
        dental_filter = get_feature_as_series(training_df, 'filter_dental')
        dental_time = get_feature_as_series(training_df, 'time_dental_pages')
        dental_accordion = get_feature_as_series(training_df, 'accordion_dental')
        dental_benefit = get_feature_as_series(training_df, 'ma_dental_benefit')
        drug_query = get_feature_as_series(training_df, 'query_drug')
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        provider_query = get_feature_as_series(training_df, 'query_provider')
        provider_filter = get_feature_as_series(training_df, 'filter_provider')
        provider_click = get_feature_as_series(training_df, 'click_provider')
        provider_network = get_feature_as_series(training_df, 'ma_provider_network')
        dsnp_filter = get_feature_as_series(training_df, 'filter_dsnp')
        dsnp_time = get_feature_as_series(training_df, 'time_dsnp_pages')
        dsnp_accordion = get_feature_as_series(training_df, 'accordion_dsnp')
        dsnp_plan = get_feature_as_series(training_df, 'dsnp')
        drug_filter = get_feature_as_series(training_df, 'filter_drug')
        drug_time = get_feature_as_series(training_df, 'time_drug_pages')
        drug_accordion = get_feature_as_series(training_df, 'accordion_drug')
        drug_click = get_feature_as_series(training_df, 'click_drug')
        drug_benefit = get_feature_as_series(training_df, 'ma_drug_benefit')
        csnp_filter = get_feature_as_series(training_df, 'filter_csnp')
        csnp_time = get_feature_as_series(training_df, 'time_csnp_pages')
        csnp_accordion = get_feature_as_series(training_df, 'accordion_csnp')
        csnp_plan = get_feature_as_series(training_df, 'csnp')
        
        training_df['dental_engagement_score'] = (
            dental_query * 5.0 +
            dental_filter * 5.0 +
            dental_time.clip(upper=5) * 4.0 +
            dental_accordion * 3.0 +
            dental_benefit * 6.0
        ) * 5.0
        additional_features.append('dental_engagement_score')
        
        training_df['dental_benefit_multiplier'] = (
            (dental_query + dental_filter + dental_accordion) * 
            (dental_benefit + 0.5) * 7.0
        ).clip(lower=0, upper=30)
        additional_features.append('dental_benefit_multiplier')
        
        training_df['dental_specificity'] = (
            dental_query * 5.0 - 
            (drug_query + dsnp_query) * 1.0
        ).clip(lower=0) * 5.0
        additional_features.append('dental_specificity')
        
        training_df['doctor_interaction_score'] = (
            provider_query * 5.0 +
            provider_filter * 5.0 +
            provider_click * 7.0 +
            provider_network * 6.0
        ) * 5.0
        additional_features.append('doctor_interaction_score')
        
        training_df['doctor_specificity'] = (
            provider_query * 5.0 - 
            (dental_query + drug_query) * 1.0
        ).clip(lower=0) * 5.0
        additional_features.append('doctor_specificity')
        
        training_df['doctor_network_boost'] = (
            (provider_query + provider_filter + provider_click) *
            (provider_network + 0.5) * 8.0
        ).clip(lower=0, upper=35)
        additional_features.append('doctor_network_boost')
        
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
        ).clip(lower=0, upper=24)
        additional_features.append('dsnp_plan_multiplier')
        
        training_df['drug_engagement_score'] = (
            drug_query * 3.0 +
            drug_filter * 3.0 +
            drug_time.clip(upper=5) * 2.0 +
            drug_accordion * 2.0 +
            drug_click * 4.0 +
            drug_benefit * 4.0
        ) * 3.5
        additional_features.append('drug_engagement_score')
        
        training_df['drug_benefit_boost'] = (
            (drug_query + drug_filter + drug_click + drug_accordion) *
            (drug_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=25)
        additional_features.append('drug_benefit_boost')
        
        training_df['csnp_engagement_score'] = (
            csnp_query * 5.0 +
            csnp_filter * 5.0 +
            csnp_time.clip(upper=5) * 4.0 +
            csnp_accordion * 3.0 +
            csnp_plan * 6.0
        ) * 5.0
        additional_features.append('csnp_engagement_score')
        
        training_df['csnp_plan_multiplier'] = (
            (csnp_query + csnp_filter + csnp_accordion) *
            (csnp_plan + 0.0) * 7.0
        ).clip(lower=0, upper=30)
        additional_features.append('csnp_plan_multiplier')
        
        training_df['dental_drug_ratio'] = (
            (dental_query + 0.8) / (drug_query + dental_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dental_drug_ratio')
        
        training_df['drug_dental_ratio'] = (
            (drug_query + 0.8) / (dental_query + drug_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('drug_dental_ratio')
        
        training_df['dental_dsnp_ratio'] = (
            (dental_query + 0.8) / (dsnp_query + dental_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dental_dsnp_ratio')
        
        training_df['drug_csnp_ratio'] = (
            (drug_query + 0.8) / (csnp_query + drug_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('drug_csnp_ratio')
        
        training_df['dsnp_drug_ratio'] = (
            (dsnp_query + 0.8) / (drug_query + dsnp_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dsnp_drug_ratio')
        
        training_df['csnp_dental_ratio'] = (
            (csnp_query + 0.8) / (dental_query + csnp_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('csnp_dental_ratio')
        
        feature_columns = (
            raw_features +
            additional_features +
            ['recency', 'time_of_day', 'visit_frequency', 'user_cluster'] +
            embedding_cols +
            [f'{persona}_weight' for persona in PERSONAS]
        )
        
        feature_columns = [
            col for col in feature_columns
            if not (
                col.lower().startswith('vision') or col.lower().endswith('_vision') or
                col in ['query_vision', 'filter_vision', 'ma_vision', 'vision_signal',
                        'vision_interaction', 'vision_primary', 'vision_plan_correlation']
            )
        ]
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        real_data_stats = {}
        for col in feature_columns:
            real_data_stats[col] = {'mean': X[col].mean(), 'std': X[col].std()}
        
        variances = X.var()
        valid_features = variances[variances > 0].index.tolist()
        X = X[valid_features]
        logger.info(f"Selected features: {valid_features[:10]}...")
        
        # Generate synthetic data
        for persona in PERSONAS:
            synthetic_examples = generate_synthetic_persona_examples(X, valid_features, persona,
                                                                   real_data_stats=real_data_stats)
            X = pd.concat([X, synthetic_examples], ignore_index=True)
            y = pd.concat([y, pd.Series([persona] * len(synthetic_examples))], ignore_index=True)
            logger.info(f"After adding synthetic {persona} samples: {Counter(y)}")
        
        # Apply SMOTE with dynamic sampling strategy
        le_temp = LabelEncoder()
        y_encoded_temp = le_temp.fit_transform(y)
        class_counts = pd.Series(y).value_counts()
        max_count = max(class_counts.values)
        sampling_strategy = {
            le_temp.transform([p])[0]: max(max_count, class_counts.get(p, 0) * 2)
            for p in PERSONAS
        }
        logger.info(f"SMOTE sampling strategy: {sampling_strategy}")
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=3
        )
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded_temp)
        y_resampled = le_temp.inverse_transform(y_resampled)
        logger.info(f"Rows after SMOTE: {len(X_resampled)}")
        logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(y_resampled).value_counts().to_string()}")
        
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = transformer.fit_transform(X_resampled)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        
        return X, y_resampled, transformer, valid_features
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def train_binary_persona_classifier(X_train, y_train, X_val, y_val, persona):
    try:
        y_train_binary = (y_train == persona).astype(int)
        y_val_binary = (y_val == persona).astype(int)
        
        class_weight = PERSONA_CLASS_WEIGHT.get(persona, 3.0)
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7 if persona in HIGH_PRIORITY_PERSONAS else 6,
            learning_rate=0.1,
            reg_lambda=1.5,
            random_state=42,
            scale_pos_weight=class_weight,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        sample_weights = np.ones(len(y_train_binary))
        sample_weights[y_train_binary == 1] = 2.5 if persona in HIGH_PRIORITY_PERSONAS else 1.5
        
        early_stop = callback.EarlyStopping(
            rounds=50,
            metric_name='logloss',
            maximize=False
        )
        
        model.fit(
            X_train, y_train_binary,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_binary)],
            callbacks=[early_stop],
            verbose=False
        )
        
        calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrated_model.fit(X_val, y_val_binary)
        
        feature_importance = model.feature_importances_
        top_features = [X_train.columns[i] for i in np.argsort(feature_importance)[-10:]]
        logger.info(f"Top features for {persona} binary classifier: {top_features}")
        
        return calibrated_model
    except Exception as e:
        logger.error(f"Failed to train binary classifier for {persona}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def custom_ensemble_with_balanced_focus(predictions, binary_probas, le, weights=None, thresholds=None):
    try:
        if weights is None:
            weights = {persona: 1.0 for persona in PERSONAS}
        if thresholds is None:
            thresholds = PERSONA_THRESHOLD
        
        weighted_preds = np.copy(predictions)
        
        for i, persona in enumerate(le.classes_):
            weighted_preds[:, i] *= weights.get(persona, 1.0)
        
        if binary_probas:
            for persona, proba in binary_probas.items():
                if persona in le.classes_:
                    persona_idx = np.where(le.classes_ == persona)[0][0]
                    blend_ratio = 0.6 if persona in HIGH_PRIORITY_PERSONAS else 0.4
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
        logger.error(f"Failed in custom ensemble: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def create_visualizations(X_val, y_val, y_pred, le):
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_val, y_pred, labels=range(len(PERSONAS)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=PERSONAS, yticklabels=PERSONAS)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('training_confusion_matrix.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        persona_acc = {}
        for persona in PERSONAS:
            mask = (y_val == le.transform([persona])[0])
            if mask.sum() > 0:
                persona_acc[persona] = accuracy_score(y_val[mask], y_pred[mask]) * 100
            else:
                persona_acc[persona] = 0
        
        acc_df = pd.DataFrame({
            'Persona': list(persona_acc.keys()),
            'Accuracy (%)': list(persona_acc.values())
        }).sort_values('Accuracy (%)', ascending=False)
        
        sns.barplot(data=acc_df, x='Persona', y='Accuracy (%)', palette='viridis')
        plt.title('Model Accuracy by Persona')
        plt.axhline(y=acc_df['Accuracy (%)'].mean(), color='r', linestyle='--', label='Mean Accuracy')
        plt.legend()
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('training_per_persona_accuracy.png')
        plt.close()
        
        logger.info("\nPer-Persona Accuracy:")
        for persona, acc in acc_df.set_index('Persona')['Accuracy (%)'].items():
            logger.info(f"{persona}: {acc:.2f}%")
        
        logger.info('Saved training_confusion_matrix.png and training_per_persona_accuracy.png')
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    logger.info("Starting consolidated persona model training at 03:36 PM CDT, May 29, 2025...")
    
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
        
        X, y, transformer, feature_columns = prepare_features(behavioral_df, plan_df)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Log class distribution
        class_distribution = pd.Series(y).value_counts()
        logger.info("\nClass distribution in training data:")
        for persona, count in class_distribution.items():
            logger.info(f"{persona}: {count}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"Training set: {len(X_train_main)} samples, Validation set: {len(X_val)} samples, Test set: {len(X_test)} samples")
        
        # Train binary classifiers
        binary_classifiers = {}
        for persona in PERSONAS:
            binary_classifiers[persona] = train_binary_persona_classifier(
                X_train_main, le.inverse_transform(y_train_main), X_val, le.inverse_transform(y_val), persona
            )
        
        # Train main model
        class_weights = {
            le.transform([p])[0]: PERSONA_CLASS_WEIGHT.get(p, 1.0)
            for p in PERSONAS
        }
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]
            
            model_fold = XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                reg_lambda=1.5,
                random_state=42+fold,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            early_stop = callback.EarlyStopping(
                rounds=50,
                metric_name='mlogloss',
                maximize=False
            )
            model_fold.fit(
                X_fold_train, y_fold_train,
                sample_weight=[class_weights.get(y, 1.0) for y in y_fold_train],
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[early_stop],
                verbose=False
            )
            models.append(model_fold)
            logger.info(f"Fold {fold+1} training completed")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"\nCross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        # Ensemble predictions
        y_pred_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
        
        binary_probas = {}
        for persona, classifier in binary_classifiers.items():
            binary_probas[persona] = classifier.predict_proba(X_test)[:, 1]
        
        y_pred, _ = custom_ensemble_with_balanced_focus(
            y_pred_proba, binary_probas, le
        )
        
        # Evaluate
        overall_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"\nValidation Results:")
        logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
        logger.info(f"Macro F1 Score: {macro_f1:.4f}")
        
        per_persona_acc = {}
        for cls_idx, cls_name in enumerate(le.classes_):
            mask = y_test == cls_idx
            if mask.sum() > 0:
                cls_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                per_persona_acc[cls_name] = cls_accuracy * 100
            else:
                per_persona_acc[cls_name] = 0.0
                logger.warning(f"No test samples for {cls_name}")
        
        logger.info("\nPer-Persona Accuracy:")
        for persona, acc in per_persona_acc.items():
            logger.info(f"{persona}: {acc:.2f}%")
        
        if overall_acc < 80:
            logger.warning("Overall accuracy below 80%. Check data distribution and feature importance.")
        
        # Visualizations
        create_visualizations(X_test, y_test, y_pred, le)
        
        # Save artifacts
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
        with open(FEATURE_NAMES_FILE, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        logger.info(f"Saved model to {MODEL_FILE}")
        logger.info(f"Saved label encoder to {LABEL_ENCODER_FILE}")
        logger.info(f"Saved transformer to {TRANSFORMER_FILE}")
        logger.info(f"Saved feature names to {FEATURE_NAMES_FILE}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
