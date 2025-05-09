import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import logging
import sys

# --- Configuration ---
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.1.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
TRANSFORMER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/power_transformer.pkl'

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Persona constants
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

# Define the base weights and thresholds
BASE_PERSONA_CLASS_WEIGHT = {
    'drug': 5.0,
    'dental': 15.5,
    'doctor': 12.5,
    'dsnp': 8.5,
    'vision': 6.5,
    'csnp': 4.5
}
BASE_PERSONA_THRESHOLD = {
    'drug': 0.20,
    'dental': 0.08,
    'doctor': 0.05,
    'dsnp': 0.20,
    'vision': 0.18,
    'csnp': 0.20
}

# Define the search space for tuning dental and doctor parameters
SEARCH_SPACE = {
    'dental_weight': [15.5, 20.0, 30.0, 40.0, 50.0, 60.0],
    'doctor_weight': [12.5, 18.0, 25.0, 35.0, 45.0, 55.0],
    'dental_threshold': [0.03, 0.05, 0.08, 0.10, 0.12],
    'doctor_threshold': [0.02, 0.04, 0.05, 0.07, 0.09]
}

# Optimization objective
OPTIMIZATION_OBJECTIVE = ['dental_accuracy', 'doctor_accuracy', 'overall_accuracy']

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)

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
    dropped_rows = 0
    
    for _, row in df.iterrows():
        persona = row['persona']
        if pd.isna(persona) or not persona:
            dropped_rows += 1
            continue
        
        try:
            personas = [p.strip().lower() for p in str(persona).split(',')]
        except Exception as e:
            logger.warning(f"Error processing persona value {persona}: {str(e)}")
            dropped_rows += 1
            continue
        
        valid_found = [p for p in personas if p in valid_personas]
        if not valid_found:
            invalid_personas.update(personas)
            dropped_rows += 1
            continue
        
        for valid_persona in valid_found:
            row_copy = row.copy()
            row_copy['persona'] = valid_persona
            new_rows.append(row_copy)
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(f"Rows after persona normalization: {len(result)} (Dropped {dropped_rows} rows)")
    if invalid_personas:
        logger.info(f"Invalid personas found: {invalid_personas}")
    if result.empty:
        logger.error(f"No valid personas found. Valid personas: {valid_personas}")
        raise ValueError("No valid personas found")
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
        if not os.path.exists(behavioral_path):
            raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
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
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
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
                training_df[col] = 0
        
        sparsity_cols = ['query_dental', 'time_dental_pages', 'query_vision', 'time_vision_pages', 'query_drug', 'query_provider']
        sparsity_stats = training_df[sparsity_cols].describe().to_dict()
        logger.info(f"Feature sparsity stats:\n{sparsity_stats}")
        
        # Feature-based boost for dental and vision
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
                if 'userid' in training_df.columns:
                    training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
                else:
                    logger.warning("userid column missing, setting visit_frequency to default value 1")
                    training_df['visit_frequency'] = 1
            except Exception as e:
                logger.warning(f"Error parsing start_time: {str(e)}. Using default values.")
                training_df['recency'] = 30
                training_df['time_of_day'] = 2
                training_df['visit_frequency'] = 1
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
            plan_sentences = []
            if 'userid' in training_df.columns:
                plan_groups = training_df.groupby('userid')['plan_id'].apply(list)
                plan_sentences = [plans for plans in plan_groups if len([p for p in plans if pd.notna(p)]) > 1]
            else:
                plan_sentences = [[p] for p in training_df['plan_id'] if pd.notna(p)]
            
            if plan_sentences:
                try:
                    w2v_model = Word2Vec(sentences=plan_sentences, vector_size=10, window=5, min_count=1, workers=4)
                    plan_embeddings = training_df['plan_id'].apply(
                        lambda x: w2v_model.wv[x] if pd.notna(x) and x in w2v_model.wv else np.zeros(10)
                    )
                    embedding_cols = [f'plan_emb_{i}' for i in range(10)]
                    training_df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=training_df.index)
                except Exception as e:
                    logger.warning(f"Error training Word2Vec model: {str(e)}. Setting embeddings to zero.")
                    embedding_cols = [f'plan_emb_{i}' for i in range(10)]
                    training_df[embedding_cols] = 0
            else:
                logger.warning("No valid plan sentences for Word2Vec. Setting embeddings to zero.")
                embedding_cols = [f'plan_emb_{i}' for i in range(10)]
                training_df[embedding_cols] = 0
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
            
            training_df[f'{persona}_plan_correlation'] = plan_col * (
                query_col + filter_col + click_col + time_col.clip(upper=3)
            ) * 2.0
            additional_features.append(f'{persona}_plan_correlation')
        
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
        
        training_df['csnp_time_intensity'] = (
            (csnp_time / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 6.0
        additional_features.append('csnp_time_intensity')
        
        training_df['csnp_plan_multiplier'] = (
            (csnp_query + csnp_filter) *
            (csnp_plan + 0.5) * 6.0
        ).clip(lower=0, upper=24)
        additional_features.append('csnp_plan_multiplier')
        
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster', 'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        if expected_features is not None:
            missing_features = [f for f in expected_features if f not in X.columns]
            extra_features = [f for f in X.columns if f not in expected_features]
            
            if missing_features:
                logger.error(f"Critical missing features: {missing_features}. Cannot proceed.")
                raise ValueError("Model expects features that are missing in the prepared data.")
            
            if extra_features:
                logger.info(f"Removing extra features: {extra_features}")
                X = X[expected_features]
        
        logger.info(f"Generated feature columns: {list(X.columns)}")
        logger.info(f"Test set size: {len(X)} samples")
        logger.info(f"Test persona distribution:\n{y.value_counts(dropna=False).to_string()}")
        
        logger.info(f"Final feature columns: {list(X.columns)}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to prepare features: {str(e)}")
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

def evaluate_model_with_params(class_weights, thresholds, main_model, binary_classifiers, le, transformer, X_test, y_test_encoded, X_test_cols):
    try:
        y_pred_probas_multi = main_model.predict_proba(transformer.transform(X_test))

        binary_probas = {}
        for persona in le.classes_:
            if persona in binary_classifiers:
                binary_probas[persona] = binary_classifiers[persona].predict_proba(transformer.transform(X_test))[:, 1]
            else:
                binary_probas[persona] = np.zeros(len(X_test))

        y_pred_probas_multi_blended = np.copy(y_pred_probas_multi)
        for i, persona in enumerate(le.classes_):
            if persona in binary_probas and binary_probas[persona].sum() > 0:
                blend_ratio = {
                    'dental': 0.4, 'vision': 0.4, 'drug': 0.6, 'doctor': 0.5
                }.get(persona, 0.5)
                y_pred_probas_multi_blended[:, i] = blend_ratio * y_pred_probas_multi_blended[:, i] + (1-blend_ratio) * binary_probas[persona]
            y_pred_probas_multi_blended[:, i] *= class_weights.get(persona, 1.0)

        y_pred_probas_multi_normalized = y_pred_probas_multi_blended / y_pred_probas_multi_blended.sum(axis=1, keepdims=True)

        y_pred = np.zeros(y_pred_probas_multi_normalized.shape[0], dtype=int)
        for i in range(y_pred_probas_multi_normalized.shape[0]):
            max_prob = -1
            max_idx = 0
            for j, persona in enumerate(le.classes_):
                prob = y_pred_probas_multi_normalized[i, j]
                threshold = thresholds.get(persona, 0.5)
                if prob > threshold and prob > max_prob:
                    max_prob = prob
                    max_idx = j
            y_pred[i] = max_idx

        overall_acc = accuracy_score(y_test_encoded, y_pred)
        macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
        per_persona_acc = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
        dental_acc = per_persona_acc.get('dental', 0.0)
        doctor_acc = per_persona_acc.get('doctor', 0.0)

        return overall_acc, dental_acc, doctor_acc, macro_f1, y_pred
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def is_better_combination(current_metrics, best_metrics, objectives, weights=None):
    if weights is None:
        weights = {'dental_accuracy': 0.4, 'doctor_accuracy': 0.4, 'overall_accuracy': 0.2}
    
    current_score = sum(current_metrics[obj] * weights.get(obj, 1.0) for obj in objectives)
    best_score = sum(best_metrics[obj] * weights.get(obj, 1.0) for obj in objectives)
    
    return current_score > best_score + 1e-5

# --- Main Tuning Logic ---
if __name__ == "__main__":
    logger.info("Starting hyperparameter tuning...")
    
    try:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        with open(MODEL_FILE, 'rb') as f:
            main_model = pickle.load(f)
        if not os.path.exists(LABEL_ENCODER_FILE):
            raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_FILE}")
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            le = pickle.load(f)
        if not os.path.exists(TRANSFORMER_FILE):
            raise FileNotFoundError(f"Transformer file not found: {TRANSFORMER_FILE}")
        with open(TRANSFORMER_FILE, 'rb') as f:
            transformer = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model or files: {str(e)}")
        sys.exit(1)
    
    expected_features = getattr(main_model, 'feature_names_', None)
    if expected_features is None:
        logger.error("Model does not provide feature_names_. Please specify expected features manually.")
        sys.exit(1)
    
    binary_classifiers = {}
    for persona in PERSONAS:
        binary_model_path = MODEL_FILE.replace('.pkl', f'_{persona}_binary.pkl')
        try:
            if not os.path.exists(binary_model_path):
                logger.warning(f"Binary classifier for {persona} not found at {binary_model_path}. Skipping this persona.")
                continue
            with open(binary_model_path, 'rb') as f:
                binary_classifiers[persona] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load binary classifier for {persona}: {str(e)}. Skipping.")
            continue
    
    if not binary_classifiers:
        logger.error("No binary classifiers loaded. At least one is required.")
        sys.exit(1)
    
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
        X_data, y_data = prepare_features(behavioral_df, plan_df, expected_features)
        y_data_encoded = le.transform(y_data)
        X_data_cols = X_data.columns
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {str(e)}")
        sys.exit(1)
    
 Plants for Sale
    best_metrics = {obj: -1 for obj in OPTIMIZATION_OBJECTIVE}
    best_params = {}
    
    logger.info("Starting grid search...")
    total_combinations = (len(SEARCH_SPACE['dental_weight']) * len(SEARCH_SPACE['doctor_weight']) *
                         len(SEARCH_SPACE['dental_threshold']) * len(SEARCH_SPACE['doctor_threshold']))
    logger.info(f"Total combinations to test: {total_combinations}")
    tested_combinations = 0
    
    for dw in SEARCH_SPACE['dental_weight']:
        for dow in SEARCH_SPACE['doctor_weight']:
            for dt in SEARCH_SPACE['dental_threshold']:
                for dot in SEARCH_SPACE['doctor_threshold']:
                    tested_combinations += 1
                    current_weights = BASE_PERSONA_CLASS_WEIGHT.copy()
                    current_thresholds = BASE_PERSONA_THRESHOLD.copy()
                    
                    current_weights['dental'] = dw
                    current_weights['doctor'] = dow
                    current_thresholds['dental'] = dt
                    current_thresholds['doctor'] = dot
                    
                    logger.info(f"Testing combination {tested_combinations}/{total_combinations}: Weights={current_weights}, Thresholds={current_thresholds}")
                    
                    overall_acc, dental_acc, doctor_acc, macro_f1, y_pred = evaluate_model_with_params(
                        current_weights, current_thresholds, main_model, binary_classifiers,
                        le, transformer, X_data, y_data_encoded, X_data_cols
                    )
                    
                    logger.info(f"Results: Overall Acc={overall_acc*100:.2f}%, Dental Acc={dental_acc:.2f}%, Doctor Acc={doctor_acc:.2f}%, Macro F1={macro_f1:.2f}")
                    
                    current_metrics = {
                        'dental_accuracy': dental_acc,
                        'doctor_accuracy': doctor_acc,
                        'overall_accuracy': overall_acc * 100
                    }
                    
                    if is_better_combination(current_metrics, best_metrics, OPTIMIZATION_OBJECTIVE):
                        logger.info("Found a better combination!")
                        best_metrics.update(current_metrics)
                        best_params = {'weights': current_weights, 'thresholds': current_thresholds}
    
    logger.info("\n--- Tuning Complete ---")
    if best_params:
        logger.info(f"Best parameters found based on objective {OPTIMIZATION_OBJECTIVE}:")
        logger.info(f"  Weights: {best_params['weights']}")
        logger.info(f"  Thresholds: {best_params['thresholds']}")
        logger.info("Corresponding metrics on the tuning dataset:")
        for obj, value in best_metrics.items():
            logger.info(f"  {obj}: {value:.2f}%" if '_accuracy' in obj else f"  {obj}: {value:.2f}")
    else:
        logger.warning("No better combination found than the starting parameters.")
        overall_acc, dental_acc, doctor_acc, macro_f1, _ = evaluate_model_with_params(
            BASE_PERSONA_CLASS_WEIGHT, BASE_PERSONA_THRESHOLD, main_model, binary_classifiers,
            le, transformer, X_data, y_data_encoded, X_data_cols
        )
        logger.info(f"Starting metrics: Overall Acc={overall_acc*100:.2f}%, Dental Acc={dental_acc:.2f}%, Doctor Acc={doctor_acc:.2f}%, Macro F1={macro_f1:.2f}")
