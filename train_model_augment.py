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
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.0.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Constants for all personas (balanced approach)
PERSONA_OVERSAMPLING_RATIO = {
    'drug': 3.5,  # Reduced from 5.0 to prevent overfitting
    'dental': 2.5,
    'doctor': 3.0,
    'dsnp': 3.0,
    'vision': 2.5,
    'csnp': 3.0
}

# More balanced class weights across all personas
PERSONA_CLASS_WEIGHT = {
    'drug': 4.0,    # Reduced from 8.0
    'dental': 3.0,  # Increased
    'doctor': 3.0,  # Increased
    'dsnp': 3.5,    # Increased
    'vision': 3.0,  # Increased
    'csnp': 3.5     # Increased
}

# Classification thresholds for each persona
PERSONA_THRESHOLD = {
    'drug': 0.35,
    'dental': 0.30,
    'doctor': 0.30,
    'dsnp': 0.30,
    'vision': 0.30,
    'csnp': 0.30
}

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
            
        # Set persona-specific feature values (high signals)
        for feature in persona_features:
            sample[feature] = np.random.uniform(2.0, 5.0)
            
        # Set specific high values for known important features
        for feature in specific_features:
            if feature in feature_columns:
                sample[feature] = np.random.uniform(4.0, 8.0)
        
        # Set plan flag if applicable
        plan_col = PERSONA_INFO.get(persona, {}).get('plan_col')
        if plan_col and plan_col in feature_columns:
            sample[plan_col] = 1
            
        # Ensure other personas' features are low
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

# Modified prepare_features function
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
        # Dental
        training_df['dental_time_intensity'] = (
            (training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5))
        ).clip(upper=0.8) * 5.0
        additional_features.append('dental_time_intensity')
        
        # Doctor
        training_df['doctor_clicks_ratio'] = (
            training_df.get('click_provider', 0) / (training_df.get('num_clicks', 1) + 1e-5)
        ).clip(upper=0.8) * 5.0
        additional_features.append('doctor_clicks_ratio')
        
        # DSNP
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        training_df['dsnp_csnp_ratio'] = (
            (dsnp_query + 0.5) / (csnp_query + dsnp_query + 1e-5)
        ).clip(0, 1) * 4.0
        additional_features.append('dsnp_csnp_ratio')
        
        # CSNP
        training_df['csnp_specificity'] = (
            training_df.get('query_csnp', 0) * 2.0 - 
            (training_df.get('query_dental', 0) + training_df.get('query_vision', 0)) * 0.5
        ).clip(lower=0) * 3.0
        additional_features.append('csnp_specificity')
        
        # Vision
        training_df['vision_focus'] = (
            training_df.get('query_vision', 0) * 2.0 + 
            training_df.get('filter_vision', 0) * 2.0 - 
            (training_df.get('query_dental', 0) + training_df.get('query_provider', 0)) * 0.3
        ).clip(lower=0) * 2.5
        additional_features.append('vision_focus')
        
        # Calculate relative dominance for each persona
        for persona in PERSONAS:
            persona_col = f'{persona}_signal'
            if persona_col in training_df.columns:
                other_signals = sum(training_df[f'{p}_signal'] for p in PERSONAS if p != persona and f'{p}_signal' in training_df.columns)
                training_df[f'{persona}_dominance'] = (
                    training_df[persona_col] * 2.0 - other_signals * 0.2
                ).clip(lower=0) * 2.0
                additional_features.append(f'{persona}_dominance')
                
                # Add super-signal indicators
                training_df[f'{persona}_super'] = safe_bool_to_int(
                    (training_df[persona_col] > training_df[persona_col].quantile(0.8)) & 
                    (training_df[f'{persona}_primary'] > 3),
                    training_df
                ) * 5.0
                additional_features.append(f'{persona}_super')
        
        # Feature selection
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
        
        # Add synthetic examples for ALL personas
        for persona in PERSONAS:
            num_samples = 1000 if persona == 'drug' else 800  # Balance synthetic data generation
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

# Revised binary classifier for any persona - avoiding feature mismatch
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
    
    # Train a CatBoost model specifically for this persona
    # We'll use a higher iterations count for more important personas
    iterations = 500 if persona in ['drug'] else 300
    
    model = CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        class_weights={0: 1.0, 1: class_weight},
        verbose=0
    )
    
    # Instead of adding new features, we're using a stronger model
    # configuration for personas where accuracy is crucial
    model.fit(
        X_train, y_train_binary,
        eval_set=(X_val, y_val_binary),
        early_stopping_rounds=50,
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

# Modified main function to handle the simplified binary classifier return value
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
    
    # Blend probabilities for each persona
    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            # Blend with a mix of 60% multi-class and 40% binary for each persona
            y_pred_probas_multi[:, i] = 0.6 * y_pred_probas_multi[:, i] + 0.4 * binary_probas[persona]
    
    # Initial class prediction based on highest probability
    y_pred = np.argmax(y_pred_probas_multi, axis=1)
    
    # Apply balanced confidence thresholds for all personas
    for persona, classifier in binary_classifiers.items():
        persona_idx = np.where(le.classes_ == persona)[0][0]
        confidence_threshold = PERSONA_THRESHOLD.get(persona, 0.3)
        high_confidence = binary_probas[persona] >= confidence_threshold * 2  # Higher threshold for overrides
        
        # Only override when binary classifier is highly confident
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
