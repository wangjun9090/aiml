import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import logging
import sys
import os
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import optuna
from collections import Counter

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
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Constants for all personas - enhanced weights and ratios for csnp and doctor
PERSONA_OVERSAMPLING_RATIO = {
    'drug': 4.5,
    'dental': 3.5,
    'doctor': 4.8,   # Increased from 4.0 for stronger doctor representation
    'dsnp': 4.0,
    'vision': 2.5,
    'csnp': 4.5      # Increased from 3.0 for stronger CSNP representation
}

# Boosted class weights for csnp and doctor
PERSONA_CLASS_WEIGHT = {
    'drug': 5.0,
    'dental': 4.5,
    'doctor': 5.5,   # Increased from 4.5 to boost doctor classification
    'dsnp': 4.8,
    'vision': 3.0,
    'csnp': 5.2      # Increased from 3.5 to boost csnp classification
}

# Modified thresholds - more aggressive for csnp and doctor
PERSONA_THRESHOLD = {
    'drug': 0.28,
    'dental': 0.25,
    'doctor': 0.22,  # Lowered from 0.25 to classify more cases as doctor
    'dsnp': 0.25,
    'vision': 0.30,
    'csnp': 0.24     # Lowered from 0.32 to classify more cases as csnp
}

# Enhanced priority personas list - add csnp and ensure doctor is included
HIGH_PRIORITY_PERSONAS = ['drug', 'dsnp', 'dental', 'doctor', 'csnp']
SUPER_PRIORITY_PERSONAS = ['drug', 'dsnp', 'doctor', 'csnp']  # Used for most aggressive overrides

# Define specialized features for each persona
PERSONA_FEATURES = {
    'dental': ['query_dental', 'filter_dental', 'time_dental_pages', 'accordion_dental', 'ma_dental_benefit'],
    'doctor': ['query_provider', 'filter_provider', 'click_provider', 'ma_provider_network'],
    'dsnp': ['query_dsnp', 'filter_dsnp', 'time_dsnp_pages', 'accordion_dsnp', 'dsnp'],
    'drug': ['query_drug', 'filter_drug', 'time_drug_pages', 'accordion_drug', 'click_drug', 'ma_drug_benefit'],
    'vision': ['query_vision', 'filter_vision', 'time_vision_pages', 'accordion_vision', 'ma_vision'],
    'csnp': ['query_csnp', 'filter_csnp', 'time_csnp_pages', 'accordion_csnp', 'csnp']
}

# Enhanced persona_info with additional signal columns for all personas
PERSONA_INFO = {
    'csnp': {
        'plan_col': 'csnp', 
        'query_col': 'query_csnp', 
        'filter_col': 'filter_csnp',
        'time_col': 'time_csnp_pages',
        'accordion_col': 'accordion_csnp'
    },
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
    'vision': {
        'plan_col': 'ma_vision', 
        'query_col': 'query_vision', 
        'filter_col': 'filter_vision',
        'time_col': 'time_vision_pages',
        'accordion_col': 'accordion_vision'
    }
}

# Generic function to generate synthetic examples for any persona
def generate_synthetic_persona_examples(X, feature_columns, persona, num_samples=1000):
    synthetic_examples = []
    persona_features = [col for col in feature_columns if persona in col.lower()]
    specific_features = PERSONA_FEATURES.get(persona, [])
    
    # Increase samples for high-priority personas
    if persona in ['drug', 'dsnp', 'csnp', 'doctor']:
        num_samples = int(num_samples * 1.8)  # 80% more synthetic samples
    elif persona in ['dental']:
        num_samples = int(num_samples * 1.5)
    
    for _ in range(num_samples):
        sample = {col: 0 for col in feature_columns}
        
        # Set base values for common features
        if 'recency' in feature_columns:
            sample['recency'] = np.random.randint(1, 30)
        if 'visit_frequency' in feature_columns:
            sample['visit_frequency'] = np.random.uniform(0.1, 0.5)
        if 'time_of_day' in feature_columns:
            sample['time_of_day'] = np.random.randint(0, 4)
        if 'user_cluster' in feature_columns:
            sample['user_cluster'] = np.random.randint(0, 5)
            
        # Set persona-specific feature values
        for feature in persona_features:
            if persona in ['drug', 'dsnp', 'csnp', 'doctor']:  # Added csnp and doctor to strongest signals
                sample[feature] = np.random.uniform(4.0, 7.0)  # Much stronger signals
            elif persona in ['dental']:
                sample[feature] = np.random.uniform(3.0, 6.0)
            else:
                sample[feature] = np.random.uniform(2.0, 5.0)
            
        # Set specific high values for known important features
        for feature in specific_features:
            if feature in feature_columns:
                if persona in ['drug', 'dsnp', 'csnp', 'doctor']:  # Added csnp and doctor to strongest signals
                    sample[feature] = np.random.uniform(7.0, 12.0)  # Extreme signals
                elif persona in ['dental']:
                    sample[feature] = np.random.uniform(6.0, 10.0)
                else:
                    sample[feature] = np.random.uniform(4.0, 8.0)
        
        # Set plan flag if applicable
        plan_col = PERSONA_INFO.get(persona, {}).get('plan_col')
        if plan_col and plan_col in feature_columns:
            sample[plan_col] = 1
            
        # Create extremely distinct separation for high priority personas
        if persona in ['drug', 'dsnp', 'csnp', 'doctor']:  # Added csnp and doctor to strongest separation
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.1)  # Extremely weak other signals
        elif persona in ['dental', 'doctor']:
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.2)
        else:
            # Regular separation for other personas
            for other_persona in PERSONAS:
                if other_persona != persona:
                    other_features = [col for col in feature_columns if other_persona in col.lower()]
                    for feature in other_features:
                        sample[feature] = np.random.uniform(0.0, 0.5)
                    
        synthetic_examples.append(sample)
    
    synthetic_df = pd.DataFrame(synthetic_examples)
    logger.info(f"Generated {len(synthetic_examples)} synthetic {persona} examples")
    return synthetic_df

# Add helper function for safe boolean to integer conversion
def safe_bool_to_int(boolean_value, df):
    """
    Convert a boolean value (scalar or Series) to an integer Series.
    """
    if isinstance(boolean_value, pd.Series):
        return boolean_value.astype(int)
    else:
        # If it's a scalar boolean, create a Series of appropriate length
        return pd.Series([int(boolean_value)] * len(df), index=df.index)

# Add helper function to ensure Series for feature values
def get_feature_as_series(df, col_name, default=0):
    """Get a column as a Series, or create a Series of default values if the column doesn't exist."""
    if col_name in df.columns:
        return df[col_name]
    else:
        return pd.Series([default] * len(df), index=df.index)

# Add the missing normalize_persona function
def normalize_persona(df):
    """
    Normalize persona values in the dataframe to match the expected PERSONAS list.
    Returns a new dataframe with only valid personas.
    """
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
        # Load behavioral data
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw behavioral columns: {list(behavioral_df.columns)}")
        
        if 'persona' in behavioral_df.columns:
            logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
            logger.info(f"Persona value counts:\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        else:
            logger.warning("Persona column missing in behavioral data")
        
        # Map invalid personas
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        
        # Impute missing values
        behavioral_df['zip'] = behavioral_df['zip'].fillna('unknown')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].fillna('unknown')
        behavioral_df['persona'] = behavioral_df['persona'].fillna('dental')
        
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip()
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip()
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        logger.info(f"Behavioral_df after cleaning: {len(behavioral_df)} rows")
        
        # Load plan data
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan_df columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# Fix the prepare_features function to ensure X is defined before being used
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
            logger.info(f"Rows after merge: {len(training_df)}")
            logger.info(f"training_df columns: {list(training_df.columns)}")
        
        # Initialize plan_features
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
        
        # Impute missing behavioral features
        imputer = SimpleImputer(strategy='median')
        for col in behavioral_features:
            if col in training_df.columns:
                training_df[col] = imputer.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = 0

        # Log feature sparsity
        sparsity_cols = ['query_dsnp', 'time_dsnp_pages', 'query_drug', 'time_drug_pages', 'query_dental', 'query_provider']
        sparsity_stats = training_df[sparsity_cols].describe().to_dict()
        logger.info(f"Feature sparsity stats:\n{sparsity_stats}")
        
        # Temporal features
        if 'start_time' in training_df.columns:
            training_df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(training_df['start_time'])).dt.days.fillna(30)
            training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
            training_df['time_of_day'] = pd.to_datetime(training_df['start_time']).dt.hour.fillna(12) // 6
        else:
            training_df['recency'] = 30
            training_df['visit_frequency'] = 1
            training_df['time_of_day'] = 2
        
        # Clustering feature
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        if all(col in training_df.columns for col in cluster_features):
            kmeans = KMeans(n_clusters=5, random_state=42)
            training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
        else:
            training_df['user_cluster'] = 0
        
        # Robust aggregates
        training_df['dental_time_ratio'] = training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5)
        training_df['click_ratio'] = training_df.get('num_clicks', 0) / (training_df.get('num_pages_viewed', 1) + 1e-5)
        
        # Plan ID embeddings
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
        
        # Aggregate features
        query_cols = [c for c in behavioral_features if c.startswith('query_') and c in training_df.columns]
        filter_cols = [c for c in behavioral_features if c.startswith('filter_') and c in training_df.columns]
        training_df['query_count'] = training_df[query_cols].sum(axis=1) if query_cols else pd.Series(0, index=training_df.index)
        training_df['filter_count'] = training_df[filter_cols].sum(axis=1) if filter_cols else pd.Series(0, index=training_df.index)
        
        # Persona weights
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        # Initialize additional_features list
        additional_features = []
        
        # Generate enhanced features for each persona type
        for persona in PERSONAS:
            persona_info = PERSONA_INFO.get(persona, {})
            
            # Get basic features for this persona
            query_col = get_feature_as_series(training_df, persona_info.get('query_col'))
            filter_col = get_feature_as_series(training_df, persona_info.get('filter_col'))
            click_col = get_feature_as_series(training_df, persona_info.get('click_col', 'dummy_col'))
            time_col = get_feature_as_series(training_df, persona_info.get('time_col', 'dummy_col'))
            accordion_col = get_feature_as_series(training_df, persona_info.get('accordion_col', 'dummy_col'))
            plan_col = get_feature_as_series(training_df, persona_info.get('plan_col'))
            
            # Create combined signal strength
            signal_weights = 3.5 if persona == 'drug' else 3.0
            training_df[f'{persona}_signal'] = (
                query_col * 2.0 +
                filter_col * 2.0 +
                time_col.clip(upper=5) * 1.5 +
                accordion_col * 1.0 +
                click_col * 2.0
            ) * signal_weights
            additional_features.append(f'{persona}_signal')
            
            # Create binary indicators
            has_interaction = (
                (query_col > 0) | 
                (filter_col > 0) | 
                (click_col > 0) | 
                (accordion_col > 0)
            )
            training_df[f'{persona}_interaction'] = safe_bool_to_int(has_interaction, training_df) * 3.0
            additional_features.append(f'{persona}_interaction')
            
            # Create primary indicators
            training_df[f'{persona}_primary'] = (
                safe_bool_to_int(query_col > 0, training_df) * 2.0 +
                safe_bool_to_int(filter_col > 0, training_df) * 2.0 +
                safe_bool_to_int(click_col > 0, training_df) * 2.0 +
                safe_bool_to_int(time_col > 2, training_df) * 1.5
            ) * 2.0
            additional_features.append(f'{persona}_primary')
            
            # Add plan correlation
            training_df[f'{persona}_plan_correlation'] = plan_col * (
                query_col + filter_col + click_col + time_col.clip(upper=3)
            ) * 2.0
            additional_features.append(f'{persona}_plan_correlation')
        
        # Add specialized features for each persona type
        # Enhanced Dental features
        dental_query = get_feature_as_series(training_df, 'query_dental')
        dental_filter = get_feature_as_series(training_df, 'filter_dental')
        dental_time = get_feature_as_series(training_df, 'time_dental_pages')
        dental_accordion = get_feature_as_series(training_df, 'accordion_dental')
        dental_benefit = get_feature_as_series(training_df, 'ma_dental_benefit')
        
        training_df['dental_time_intensity'] = (
            (training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 5.0
        additional_features.append('dental_time_intensity')
        
        # New dental engagement score - combines all dental signals
        training_df['dental_engagement_score'] = (
            dental_query * 3.0 +
            dental_filter * 3.0 +
            dental_time.clip(upper=5) * 2.0 +
            dental_accordion * 2.0 +
            dental_benefit * 4.0
        ) * 3.0
        additional_features.append('dental_engagement_score')
        
        # Dental benefit multiplier - strengthens signal when dental benefit exists
        training_df['dental_benefit_multiplier'] = (
            (dental_query + dental_filter + dental_accordion) * 
            (dental_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('dental_benefit_multiplier')
        
        # Enhanced Doctor features - adding more specialized features
        provider_query = get_feature_as_series(training_df, 'query_provider')
        provider_filter = get_feature_as_series(training_df, 'filter_provider')
        provider_click = get_feature_as_series(training_df, 'click_provider')
        provider_network = get_feature_as_series(training_df, 'ma_provider_network')
        
        # Further enhance doctor_interaction_score with higher weights
        training_df['doctor_interaction_score'] = (
            provider_query * 3.0 +
            provider_filter * 3.0 +
            provider_click * 5.0 +  # Increased from 4.0
            provider_network * 4.0  # Increased from 3.0
        ) * 4.0  # Increased from 3.5
        additional_features.append('doctor_interaction_score')
        
        # Doctor vs Vision & Dental contrast - creates separation
        training_df['doctor_specificity'] = (
            provider_query * 3.0 - 
            (training_df.get('query_dental', 0) + training_df.get('query_vision', 0)) * 0.7
        ).clip(lower=0) * 4.0
        additional_features.append('doctor_specificity')
        
        # Doctor network boost - amplifies signal when provider network exists
        training_df['doctor_network_boost'] = (
            (provider_query + provider_filter + provider_click) *
            (provider_network + 0.5) * 6.0
        ).clip(lower=0, upper=25)
        additional_features.append('doctor_network_boost')
        
        # Provider page depth - measure of how deeply user engaged with provider content
        training_df['doctor_page_depth'] = (
            (provider_click / (provider_query + 0.1)) * 10.0
        ).clip(0, 20)
        additional_features.append('doctor_page_depth')
        
        # Enhanced DSNP features - significantly boosted
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        dsnp_filter = get_feature_as_series(training_df, 'filter_dsnp')
        dsnp_time = get_feature_as_series(training_df, 'time_dsnp_pages')
        dsnp_accordion = get_feature_as_series(training_df, 'accordion_dsnp')
        dsnp_plan = get_feature_as_series(training_df, 'dsnp')
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        
        # DSNP vs CSNP ratio with stronger differentiation
        training_df['dsnp_csnp_ratio'] = (
            (dsnp_query + 0.8) / (csnp_query + dsnp_query + 1e-5)
        ).clip(0, 1) * 5.0  # Increased from 4.0
        additional_features.append('dsnp_csnp_ratio')
        
        # New DSNP engagement score - combines all DSNP signals
        training_df['dsnp_engagement_score'] = (
            dsnp_query * 3.0 +
            dsnp_filter * 3.0 +
            dsnp_time.clip(upper=5) * 2.0 +
            dsnp_accordion * 2.0 +
            dsnp_plan * 5.0
        ) * 3.0
        additional_features.append('dsnp_engagement_score')
        
        # DSNP plan interaction multiplier
        training_df['dsnp_plan_multiplier'] = (
            (dsnp_query + dsnp_filter + dsnp_accordion) *
            (dsnp_plan + 0.5) * 5.0
        ).clip(lower=0, upper=20)
        additional_features.append('dsnp_plan_multiplier')
        
        # Enhanced DRUG features - significantly boosted
        drug_query = get_feature_as_series(training_df, 'query_drug')
        drug_filter = get_feature_as_series(training_df, 'filter_drug')
        drug_time = get_feature_as_series(training_df, 'time_drug_pages')
        drug_accordion = get_feature_as_series(training_df, 'accordion_drug')
        drug_click = get_feature_as_series(training_df, 'click_drug')
        drug_benefit = get_feature_as_series(training_df, 'ma_drug_benefit')
        
        # Drug engagement compound score
        training_df['drug_engagement_score'] = (
            drug_query * 3.0 +
            drug_filter * 3.0 +
            drug_time.clip(upper=5) * 2.0 +
            drug_accordion * 2.0 +
            drug_click * 4.0 +
            drug_benefit * 4.0
        ) * 3.5  # Higher multiplier than dental/doctor
        additional_features.append('drug_engagement_score')
        
        # Drug interest ratio - what proportion of queries were drug-related
        training_df['drug_interest_ratio'] = (
            (drug_query + drug_filter) /
            (training_df.get('query_count', 1) + training_df.get('filter_count', 1) + 1e-5)
        ).clip(upper=0.9) * 10.0
        additional_features.append('drug_interest_ratio')
        
        # Drug benefit boost - amplifies signal when drug benefit exists
        training_df['drug_benefit_boost'] = (
            (drug_query + drug_filter + drug_click + drug_accordion) *
            (drug_benefit + 0.5) * 5.0
        ).clip(lower=0, upper=25)
        additional_features.append('drug_benefit_boost')
        
        # Drug time intensity - how focused was session on drug pages
        training_df['drug_time_intensity'] = (
            (drug_time / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 6.0  # Higher multiplier than dental
        additional_features.append('drug_time_intensity')
        
        # CSNP features - significantly upgraded
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        csnp_filter = get_feature_as_series(training_df, 'filter_csnp')
        csnp_time = get_feature_as_series(training_df, 'time_csnp_pages')
        csnp_accordion = get_feature_as_series(training_df, 'accordion_csnp')
        csnp_plan = get_feature_as_series(training_df, 'csnp')
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        
        # CSNP vs DSNP differentiation (reverse of the dsnp_csnp_ratio) with stronger weighting
        training_df['csnp_dsnp_ratio'] = (
            (csnp_query + 0.8) / (dsnp_query + csnp_query + 1e-5)
        ).clip(0, 1) * 5.0
        additional_features.append('csnp_dsnp_ratio')
        
        # Improved CSNP specificity with stronger negative weights for other features
        training_df['csnp_specificity'] = (
            csnp_query * 3.0 - 
            (training_df.get('query_dental', 0) + 
             training_df.get('query_vision', 0) + 
             training_df.get('query_drug', 0)) * 0.8
        ).clip(lower=0) * 4.0  # Increased from 3.0
        additional_features.append('csnp_specificity')
        
        # New CSNP engagement compound score
        training_df['csnp_engagement_score'] = (
            csnp_query * 3.0 +
            csnp_filter * 3.0 +
            csnp_time.clip(upper=5) * 2.0 +
            csnp_accordion * 2.0 +
            csnp_plan * 5.0
        ) * 4.0  # Higher multiplier than dsnp engagement
        additional_features.append('csnp_engagement_score')
        
        # CSNP plan interaction multiplier
        training_df['csnp_plan_multiplier'] = (
            (csnp_query + csnp_filter + csnp_accordion) *
            (csnp_plan + 0.5) * 6.0
        ).clip(lower=0, upper=24)
        additional_features.append('csnp_plan_multiplier')
        
        # CSNP time intensity - how focused was session on csnp pages
        training_df['csnp_time_intensity'] = (
            (csnp_time / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 6.0
        additional_features.append('csnp_time_intensity')
        
        # First create X and y BEFORE trying to use them for synthetic data
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster', 
            'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        
        X = training_df[feature_columns].fillna(0)
        variances = X.var()
        valid_features = variances[variances > 1e-5].index.tolist()  # Less strict filtering to keep drug features
        X = X[valid_features]
        logger.info(f"Selected features after variance filtering: {valid_features}")
        
        y = training_df['persona']
        training_df = training_df[training_df['persona'].notna()].reset_index(drop=True)
        logger.info(f"Rows after filtering: {len(training_df)}")
        logger.info(f"Pre-SMOTE persona distribution:\n{training_df['persona'].value_counts(dropna=False).to_string()}")
        
        # NOW add synthetic examples for ALL personas AFTER X and y are defined
        for persona in PERSONAS:
            # Generate more synthetic examples for high-priority personas
            num_samples = 2000 if persona in SUPER_PRIORITY_PERSONAS else (
                1500 if persona == 'dental' else 800)
            synthetic_examples = generate_synthetic_persona_examples(X, valid_features, persona, num_samples=num_samples)
            X = pd.concat([X, synthetic_examples], ignore_index=True)
            y = pd.concat([y, pd.Series([persona] * len(synthetic_examples))], ignore_index=True)
            logger.info(f"After adding synthetic {persona} examples: {Counter(y)}")
        
        # Modified SMOTE with balanced sampling for all personas
        class_counts = pd.Series(y).value_counts()
        sampling_strategy = {
            persona: int(count * PERSONA_OVERSAMPLING_RATIO.get(persona, 2.0))
            for persona, count in class_counts.items()
        }
        logger.info(f"Balanced SMOTE sampling strategy: {sampling_strategy}")
        
        # Using SMOTETomek for better quality synthetic samples
        smote = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
        X, y = smote.fit_resample(X, y)
        logger.info(f"Rows after balanced SMOTE: {len(X)}")
        logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(y).value_counts().to_string()}")
        
        # Apply power transformation for better feature distribution
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_transformed = power_transformer.fit_transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        
        return X, y, power_transformer
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

# Enhanced binary classifier with specialized parameters for csnp and doctor
def train_binary_persona_classifier(X_train, y_train, X_val, y_val, persona):
    # Convert to binary classification problem
    y_train_binary = (y_train == persona).astype(int)
    y_val_binary = (y_val == persona).astype(int)
    
    # Get class weight for this persona
    class_weight = PERSONA_CLASS_WEIGHT.get(persona, 3.0)
    threshold = PERSONA_THRESHOLD.get(persona, 0.3)
    
    # Find persona-specific features for logging
    persona_features = [col for col in X_train.columns if persona in col.lower()]
    logger.info(f"Training {persona} classifier with {len(persona_features)} specific features")
    
    # Super-enhanced parameters for all high priority personas including csnp and doctor
    if persona in ['csnp', 'doctor']:  # Specific tuning for csnp and doctor
        iterations = 950  # Even more iterations for doctor and csnp
        depth = 8         # Deeper trees for more complex patterns
        learning_rate = 0.015  # Slower learning rate for better generalization
        l2_leaf_reg = 1.5      # Less regularization to fit patterns better
        early_stopping = 100   # More patience for finding optimal iteration
    elif persona in ['drug', 'dsnp']:
        iterations = 800
        depth = 7
        learning_rate = 0.02
        l2_leaf_reg = 1.8
        early_stopping = 90
    elif persona in ['dental']:
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
    
    # Add boosted sampling weight for high priority personas
    if persona in SUPER_PRIORITY_PERSONAS:  # Now includes csnp and doctor
        # Create sample weights that prioritize positive examples
        sample_weights = np.ones(len(y_train_binary))
        # Give even more weight to doctor and csnp positive examples
        if persona in ['doctor', 'csnp']:
            sample_weights[y_train_binary == 1] = 1.8  # 80% more weight to positive examples
        else:
            sample_weights[y_train_binary == 1] = 1.5
        model.fit(
            X_train, y_train_binary,
            eval_set=(X_val, y_val_binary),
            early_stopping_rounds=early_stopping,
            sample_weight=sample_weights,
            verbose=False
        )
    else:
        # Standard training for other personas
        model.fit(
            X_train, y_train_binary,
            eval_set=(X_val, y_val_binary),
            early_stopping_rounds=early_stopping,
            verbose=False
        )
    
    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, cv='prefit')
    calibrated_model.fit(X_val, y_val_binary)
    
    # Evaluate
    y_pred_proba = calibrated_model.predict_proba(X_val)[:,1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_val_binary, y_pred_proba)
    precision = precision_score(y_val_binary, y_pred)
    recall = recall_score(y_val_binary, y_pred)
    f1 = f1_score(y_val_binary, y_pred)
    
    logger.info(f"Binary {persona.capitalize()} Classifier - AUC: {auc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    
    return calibrated_model

# Add function to compute per-persona accuracy
def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    """
    Calculate accuracy for each persona separately.
    
    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        classes: Class indices array (not used, kept for API compatibility)
        class_names: Names of classes (personas)
        
    Returns:
        Dictionary mapping persona names to their accuracy percentages
    """
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    return per_persona_accuracy

# Modified main function with enhanced blending and override strategies
def main():
    # Load data
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Prepare enhanced features
    try:
        X, y, transformer = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Further split training data for binary classifier validation
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train binary classifiers for each persona
    binary_classifiers = {}
    for persona in PERSONAS:
        binary_classifiers[persona] = train_binary_persona_classifier(
            X_train_main, y_train_main, X_val, y_val, persona
        )
    
    # Train the main multi-class model with balanced weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    feature_importances = []
    
    # Create balanced class weights
    class_weights = {}
    for i, persona in enumerate(le.classes_):
        class_weights[i] = PERSONA_CLASS_WEIGHT.get(persona, 3.0)
    logger.info(f"Using balanced class weights: {class_weights}")
    
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
        feature_importances.append(model.get_feature_importance())
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions with binary classifier boosting for all personas
    y_pred_probas_multi = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    
    # Get binary classifier probabilities for each persona
    binary_probas = {}
    for persona, classifier in binary_classifiers.items():
        # Use the same feature set that was used for training
        binary_probas[persona] = classifier.predict_proba(X_test)[:,1]
    
    # Blend probabilities - custom blending for high priority personas
    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            if persona in ['doctor', 'csnp']:  # Strongest binary influence for doctor and csnp
                blend_ratio = 0.35  # 35% multi-class, 65% binary - extremely strong binary influence
            elif persona in ['drug', 'dsnp']:
                blend_ratio = 0.4   # 40% multi-class, 60% binary
            elif persona in ['dental']:
                blend_ratio = 0.5   # 50-50 blend
            else:
                blend_ratio = 0.6   # 60% multi-class, 40% binary for others
                
            y_pred_probas_multi[:, i] = blend_ratio * y_pred_probas_multi[:, i] + (1-blend_ratio) * binary_probas[persona]
    
    # Initial class prediction based on highest probability
    y_pred = np.argmax(y_pred_probas_multi, axis=1)
    
    # Super-aggressive overrides for high priority personas, with special focus on csnp and doctor
    for persona, classifier in binary_classifiers.items():
        persona_idx = np.where(le.classes_ == persona)[0][0]
        confidence_threshold = PERSONA_THRESHOLD.get(persona, 0.3)
        
        if persona in ['doctor', 'csnp']:  # Most aggressive overrides for doctor and csnp
            # Ultra-aggressive override for doctor and csnp
            high_confidence = binary_probas[persona] >= confidence_threshold * 1.2  # Very low threshold
            medium_confidence = binary_probas[persona] >= confidence_threshold * 0.8  # Even lower medium threshold
            
            # Two-level confidence override with more aggressive thresholds
            strong_signal_mask = high_confidence & (y_pred_probas_multi[:, persona_idx] > confidence_threshold * 0.6)
            y_pred[strong_signal_mask] = persona_idx
            
            # Apply a very aggressive second-pass override with medium confidence
            # But avoid overriding the other critical personas (drug and dsnp)
            critical_personas = ['drug', 'dsnp'] if persona == 'csnp' else ['drug', 'dsnp', 'csnp'] if persona == 'doctor' else []
            weaker_classes = [j for j, p in enumerate(le.classes_) if p not in critical_personas]
            in_weaker_class = np.isin(y_pred, weaker_classes)
            medium_signal_mask = medium_confidence & ~strong_signal_mask & in_weaker_class
            y_pred[medium_signal_mask] = persona_idx
            
            # Apply a third-pass override that's just for extremely borderline cases
            # This applies only to vision/csnp for maximum accuracy boost without hurting others
            if persona == 'csnp' or persona == 'doctor':
                lowest_confidence = binary_probas[persona] >= confidence_threshold * 0.7
                borderline_classes = [j for j, p in enumerate(le.classes_) if p in ['vision'] and p != persona]
                in_borderline_class = np.isin(y_pred, borderline_classes)
                borderline_signal_mask = lowest_confidence & ~medium_signal_mask & ~strong_signal_mask & in_borderline_class
                y_pred[borderline_signal_mask] = persona_idx
        
        elif persona in ['drug', 'dsnp']:
            # Extremely aggressive override for top priority personas
            high_confidence = binary_probas[persona] >= confidence_threshold * 1.4  # Very low threshold
            medium_confidence = binary_probas[persona] >= confidence_threshold
            
            # Two-level confidence override - first strong signals, then moderate signals
            strong_signal_mask = high_confidence & (y_pred_probas_multi[:, persona_idx] > confidence_threshold * 0.7)
            y_pred[strong_signal_mask] = persona_idx
            
            # Apply a second-pass override with medium confidence
            # But only override 'weaker' classes (not drug, dental, doctor if we're dsnp and vice versa)
            priority_personas = ['drug', 'dsnp', 'dental', 'doctor']
            weaker_classes = [j for j, p in enumerate(le.classes_) if p not in priority_personas or p == persona]
            in_weaker_class = np.isin(y_pred, weaker_classes)
            medium_signal_mask = medium_confidence & ~strong_signal_mask & in_weaker_class
            y_pred[medium_signal_mask] = persona_idx
        elif persona in ['dental']:
            high_confidence = binary_probas[persona] >= confidence_threshold * 1.7
            strong_signal_mask = high_confidence & (y_pred_probas_multi[:, persona_idx] > confidence_threshold * 0.8)
            y_pred[strong_signal_mask] = persona_idx
        else:
            high_confidence = binary_probas[persona] >= confidence_threshold * 2.0
            strong_signal_mask = high_confidence & (y_pred_probas_multi[:, persona_idx] > confidence_threshold)
            y_pred[strong_signal_mask] = persona_idx
    
    # Evaluate overall performance
    overall_accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    logger.info(f"Overall Accuracy on Test Set: {overall_accuracy * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
    # Calculate per-persona metrics
    per_persona_accuracy = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    logger.info("Per-Persona Accuracy (%):")
    for persona, acc in per_persona_accuracy.items():
        logger.info(f"  {persona}: {acc:.2f}%")
    
    # Save models and data
    model = models[0]  # Use first model from ensemble for simplicity
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    # Save binary classifiers
    for persona, clf in binary_classifiers.items():
        binary_model_path = MODEL_FILE.replace('.pkl', f'_{persona}_binary.pkl')
        with open(binary_model_path, 'wb') as f:
            pickle.dump(clf, f)
    
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
        
    transformer_file = SCALER_FILE.replace('scaler.pkl', 'power_transformer.pkl')
    with open(transformer_file, 'wb') as f:
        pickle.dump(transformer, f)
        
    logger.info("Saved models, label encoder, and transformer to disk.")

if __name__ == "__main__":
    main()
