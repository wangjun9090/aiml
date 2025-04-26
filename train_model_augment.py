import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import logging
import sys
import os
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File paths
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-2.0.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Persona info
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
}

def calculate_persona_weight(row, persona_info, persona):
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    plan_col = persona_info.get('plan_col', None)
    click_col = persona_info.get('click_col', None)
    
    query_value = row.get(query_col, 0) if pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col, np.nan)) else 0
    plan_value = row.get(plan_col, 0) if plan_col and pd.notna(row.get(plan_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    time_col = f'time_{persona}_pages'
    page_views = min(row.get(time_col, 0) / 60, 3) if time_col in row and pd.notna(row.get(time_col, np.nan)) else 0
    
    if persona == 'csnp':
        query_weight = 2.5
        filter_weight = 2.3
    else:
        query_weight = 0.8
        filter_weight = 0.7
    
    click_weight = 0.4 if persona == 'doctor' else 0.3 if persona == 'drug' else 0
    page_view_weight = 0.15
    
    weighted_query = query_value * query_weight
    weighted_filter = filter_value * filter_weight
    weighted_page_views = page_views * page_view_weight
    weighted_click = click_value * click_weight
    
    extra_points = 0
    if filter_value > 0 and click_value > 0:
        extra_points += 0.8
    elif filter_value > 0 or click_value > 0:
        extra_points += 0.4
    
    special_points = 0
    action_count = sum(1 for v in [query_value, filter_value, click_value, page_views] if v > 0)
    interaction_col = f'{persona}_interaction'
    interaction_value = row.get(interaction_col, 0) if interaction_col in row else 0
    
    if persona == 'doctor':
        if click_value >= 1.5:
            special_points += 0.5
        elif click_value >= 0.5:
            special_points += 0.25
    elif persona == 'drug':
        if click_value >= 5:
            special_points += 0.5
        elif click_value >= 2:
            special_points += 0.25
    elif persona == 'dental':
        if action_count >= 2:
            special_points += 0.7
        elif action_count >= 1:
            special_points += 0.4
        if interaction_value > 0:
            special_points += 0.4
        if plan_value > 0.5:
            special_points += 0.6
    elif persona == 'vision':
        if action_count >= 1:
            special_points += 0.6
        if interaction_value > 0:
            special_points += 0.4
        if plan_value > 0.5:
            special_points += 0.6
    elif persona == 'csnp':
        if action_count >= 2:
            special_points += 1.2
        elif action_count >= 1:
            special_points += 0.8
        if interaction_value > 0:
            special_points += 1.2
        if row.get('csnp_type_flag', 0) == 1:
            special_points += 1.0
        if row.get('csnp_drug_interaction', 0) > 0:
            special_points += 0.8
        if row.get('csnp_doctor_interaction', 0) > 0:
            special_points += 0.6
        if plan_value > 0.5:
            special_points += 1.5
    elif persona == 'dsnp':
        pass
    
    weight = (weighted_query + weighted_filter + weighted_page_views + weighted_click + extra_points + special_points)
    return min(max(weight, 0), 5.0)

def load_data():
    try:
        behavioral_df = pd.read_csv(BEHAVIORAL_FILE)
        plan_df = pd.read_csv(PLAN_FILE)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
        logger.info(f"Persona value counts (raw):\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        
        # Validate required columns
        required_behavioral_cols = ['persona', 'zip', 'plan_id']
        missing_behavioral_cols = [col for col in required_behavioral_cols if col not in behavioral_df.columns]
        if missing_behavioral_cols:
            logger.error(f"Missing required columns in BEHAVIORAL_FILE: {missing_behavioral_cols}")
            raise ValueError(f"Missing required columns in BEHAVIORAL_FILE: {missing_behavioral_cols}")
        
        required_plan_cols = ['zip', 'plan_id']
        missing_plan_cols = [col for col in required_plan_cols if col not in plan_df.columns]
        if missing_plan_cols:
            logger.error(f"Missing required columns in PLAN_FILE: {missing_plan_cols}")
            raise ValueError(f"Missing required columns in PLAN_FILE: {missing_plan_cols}")
        
        # Log zip and plan_id overlap
        behavioral_zips = set(behavioral_df['zip'].astype(str))
        plan_zips = set(plan_df['zip'].astype(str))
        zip_overlap = len(behavioral_zips.intersection(plan_zips)) / len(behavioral_zips) * 100
        logger.info(f"Zip overlap between BEHAVIORAL_FILE and PLAN_FILE: {zip_overlap:.2f}%")
        
        behavioral_plan_ids = set(behavioral_df['plan_id'].astype(str))
        plan_plan_ids = set(plan_df['plan_id'].astype(str))
        plan_id_overlap = len(behavioral_plan_ids.intersection(plan_plan_ids)) / len(behavioral_plan_ids) * 100
        logger.info(f"Plan_id overlap between BEHAVIORAL_FILE and PLAN_FILE: {plan_id_overlap:.2f}%")
        
        # Clean and validate persona
        behavioral_df['persona'] = behavioral_df['persona'].apply(lambda x: x.lower() if isinstance(x, str) else 'dental')
        behavioral_df['persona'] = behavioral_df['persona'].apply(lambda x: x if x in PERSONAS else 'dental')
        logger.info(f"Persona value counts (after cleaning):\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip()
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip()
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        
        logger.info(f"Plan_df columns: {list(plan_df.columns)}")
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def prepare_features(behavioral_df, plan_df):
    try:
        # Merge data
        df = behavioral_df.merge(plan_df, on=['zip', 'plan_id'], how='left')
        logger.info(f"Rows after merge: {len(df)}")
        logger.info(f"Merged df columns: {list(df.columns)}")
        logger.info(f"Persona distribution after merge:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Validate merge success
        if not plan_df.columns.difference(['zip', 'plan_id']).empty:
            merge_success_rate = df[plan_df.columns.difference(['zip', 'plan_id'])].notna().any(axis=1).mean()
            logger.info(f"Merge success rate (non-null plan features): {merge_success_rate:.2%}")
        else:
            logger.warning("No plan features to merge (only zip, plan_id)")
            merge_success_rate = 0.0
        
        # Validate persona column
        if 'persona' not in df.columns or df['persona'].isna().all():
            logger.error("Persona column missing or all NaN")
            raise ValueError("Persona column is missing or invalid")
        
        # Ensure valid persona values
        df['persona'] = df['persona'].apply(lambda x: x if x in PERSONAS else 'dental')
        logger.info(f"Persona distribution after validation:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Signal strength calculation (for logging, not filtering)
        query_cols = ['query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp']
        query_cols = [col for col in query_cols if col in df.columns]
        if not query_cols:
            logger.warning("No query columns found, setting signal_strength to 0")
            df['signal_strength'] = pd.Series(0, index=df.index)
        else:
            df['signal_strength'] = df[query_cols].sum(axis=1)
            logger.info(f"Signal strength stats:\n{df['signal_strength'].describe().to_dict()}")
        
        # Skip signal strength filter to retain all samples
        logger.info(f"Rows after processing (no signal strength filter): {len(df)}")
        logger.info(f"Persona distribution after processing:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        if df.empty:
            logger.warning("DataFrame is empty after processing")
            df = pd.DataFrame({
                'persona': PERSONAS,
                'signal_strength': [0.1] * len(PERSONAS),
                'zip': ['00000'] * len(PERSONAS),
                'plan_id': ['dummy'] * len(PERSONAS)
            })
            logger.info(f"Created dummy DataFrame with personas:\n{df['persona'].value_counts().to_string()}")
        
        # Plan features
        plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network']
        available_plan_features = [col for col in plan_features if col in df.columns]
        missing_plan_features = [col for col in plan_features if col not in df.columns]
        logger.info(f"Available plan features: {available_plan_features}")
        logger.info(f"Missing plan features: {missing_plan_features}")
        
        for col in plan_features:
            if col not in df.columns:
                df[col] = pd.Series(0, index=df.index)
            else:
                df[col] = df[col].fillna(0)
        
        # Behavioral features
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_provider', 'accordion_vision', 'accordion_dsnp'
        ]
        available_behavioral_features = [col for col in behavioral_features if col in df.columns]
        missing_behavioral_features = [col for col in behavioral_features if col not in df.columns]
        logger.info(f"Available behavioral features: {available_behavioral_features}")
        logger.info(f"Missing behavioral features: {missing_behavioral_features}")
        
        # Fallback for critical csnp columns
        if 'query_csnp' not in df.columns:
            logger.warning("query_csnp missing, using filter_csnp as fallback")
            df['query_csnp'] = df.get('filter_csnp', pd.Series(0, index=df.index)) * 0.3
        if 'csnp' not in df.columns:
            logger.warning("csnp missing, initializing to 0")
            df['csnp'] = pd.Series(0, index=df.index)
        
        # Initialize and impute behavioral features
        imputer = SimpleImputer(strategy='median')
        for col in behavioral_features:
            if col in df.columns:
                df[col] = imputer.fit_transform(df[[col]]).flatten()
                if 'query_' in col and col.replace('query_', 'accordion_') in df.columns:
                    accordion_col = col.replace('query_', 'accordion_')
                    mask = (df[col] == 0) & (df[accordion_col] > 0)
                    df.loc[mask, col] = df.loc[mask, accordion_col] * (0.5 if 'csnp' in col else 0.3)
                if 'filter_' in col and col.replace('filter_', 'accordion_') in df.columns:
                    accordion_col = col.replace('filter_', 'accordion_')
                    mask = (df[col] == 0) & (df[accordion_col] > 0)
                    df.loc[mask, col] = df.loc[mask, accordion_col] * (0.5 if 'csnp' in col else 0.3)
                if 'time_' in col and col.replace('time_', 'query_') in df.columns:
                    query_col = col.replace('time_', 'query_')
                    mask = (df[col] == 0) & (df[query_col] > 0)
                    df.loc[mask, col] = df.loc[mask, query_col] * (0.7 if 'csnp' in col else 0.3)
            else:
                df[col] = pd.Series(0, index=df.index)
        
        # Add csnp proxy
        df['csnp_type_flag'] = df.get('csnp_type', pd.Series('N', index=df.index)).map({'Y': 1, 'N': 0}).fillna(0).astype(int)
        df.loc[(df['query_csnp'] == 0) & (df['csnp_type_flag'] == 1), 'query_csnp'] = 0.5
        
        # Log sparsity
        sparsity_cols = ['query_csnp', 'query_dsnp', 'time_csnp_pages', 'time_dsnp_pages']
        sparsity_cols = [col for col in sparsity_cols if col in df.columns]
        logger.info(f"Feature sparsity stats:\n{df[sparsity_cols].describe().to_dict()}")
        
        # Temporal features
        if 'start_time' in df.columns:
            df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(df['start_time'])).dt.days.fillna(30)
        else:
            df['recency'] = pd.Series(30, index=df.index)
        
        # Plan ID embeddings
        if 'plan_id' in df.columns and 'userid' in df.columns:
            plan_sentences = df.groupby('userid')['plan_id'].apply(list).tolist()
            w2v_model = Word2Vec(sentences=plan_sentences, vector_size=20, window=5, min_count=1, workers=4)
            plan_embeddings = df['plan_id'].apply(
                lambda x: w2v_model.wv[x] if x in w2v_model.wv else np.zeros(20)
            )
            embedding_cols = [f'plan_emb_{i}' for i in range(20)]
            df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=df.index)
        else:
            embedding_cols = [f'plan_emb_{i}' for i in range(20)]
            df[embedding_cols] = pd.Series(0, index=df.index)
        
        # Persona weights
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                df[f'{persona}_weight'] = df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        # Skip label validation to isolate issue
        # mismatches = df[
        #     ((df['persona'] == 'csnp') & (df['csnp'] == 0) & (df['query_csnp'] > 0.5)) |
        #     ((df['persona'] != 'csnp') & (df['csnp'] == 1) & (df['query_csnp'] > 0.5))
        # ]
        # logger.info(f"Label mismatches: {len(mismatches)}")
        # df.loc[
        #     (df['persona'] == 'csnp') & (df['csnp'] == 0) & (df['query_csnp'] > 0.5),
        #     'persona'
        # ] = 'dental'
        # df.loc[
        #     (df['persona'] != 'csnp') & (df['csnp'] == 1) & (df['query_csnp'] > 0.5),
        #     'persona'
        # ] = 'csnp'
        logger.info(f"Persona distribution after processing (no label validation):\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Ensure all personas are present
        missing_personas = [p for p in PERSONAS if p not in df['persona'].unique()]
        if missing_personas:
            logger.warning(f"Missing personas: {missing_personas}")
            for persona in missing_personas:
                dummy_row = pd.Series(0, index=df.columns)
                dummy_row['persona'] = persona
                dummy_row['signal_strength'] = 0.1
                dummy_row['zip'] = '00000'
                dummy_row['plan_id'] = 'dummy'
                df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
            logger.info(f"Added dummy samples for missing personas. New persona distribution:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Enhanced features
        additional_features = []
        for persona in PERSONAS:
            signal = (
                df.get(f'query_{persona}', pd.Series(0, index=df.index)) +
                df.get(f'filter_{persona}', pd.Series(0, index=df.index)) +
                df.get(f'time_{persona}_pages', pd.Series(0, index=df.index)).clip(upper=5) +
                df.get(f'accordion_{persona}', pd.Series(0, index=df.index))
            ).clip(lower=0, upper=5)
            df[f'{persona}_signal'] = signal * (8.0 if persona == 'csnp' else 6.0 if persona == 'dsnp' else 5.0)
            additional_features.append(f'{persona}_signal')
            
            if persona in ['csnp', 'dsnp']:
                df[f'{persona}_dental_interaction'] = (
                    df[PERSONA_INFO[persona]['plan_col']] * (
                        df.get(f'query_{persona}', pd.Series(0, index=df.index)) +
                        df.get(f'filter_{persona}', pd.Series(0, index=df.index))
                    ) * 4.0 - df.get('ma_dental_benefit', pd.Series(0, index=df.index)) * (
                        df.get('query_dental', pd.Series(0, index=df.index)) +
                        df.get('filter_dental', pd.Series(0, index=df.index))
                    )
                ).clip(lower=0) * (8.0 if persona == 'csnp' else 6.0)
                additional_features.append(f'{persona}_dental_interaction')
                
                df[f'{persona}_vision_interaction'] = (
                    df[PERSONA_INFO[persona]['plan_col']] * (
                        df.get(f'query_{persona}', pd.Series(0, index=df.index)) +
                        df.get(f'filter_{persona}', pd.Series(0, index=df.index))
                    ) * 4.0 - df.get('ma_vision', pd.Series(0, index=df.index)) * (
                        df.get('query_vision', pd.Series(0, index=df.index)) +
                        df.get('filter_vision', pd.Series(0, index=df.index))
                    )
                ).clip(lower=0) * (8.0 if persona == 'csnp' else 6.0)
                additional_features.append(f'{persona}_vision_interaction')
                
                df[f'{persona}_doctor_interaction'] = (
                    df[PERSONA_INFO[persona]['plan_col']] * (
                        df.get(f'query_{persona}', pd.Series(0, index=df.index)) +
                        df.get(f'filter_{persona}', pd.Series(0, index=df.index))
                    ) * 4.0 - df.get('ma_provider_network', pd.Series(0, index=df.index)) * (
                        df.get('query_provider', pd.Series(0, index=df.index)) +
                        df.get('filter_provider', pd.Series(0, index=df.index))
                    )
                ).clip(lower=0) * (8.0 if persona == 'csnp' else 6.0)
                additional_features.append(f'{persona}_doctor_interaction')
        
        df['csnp_drug_interaction'] = (
            df['csnp'] * (
                df.get('query_csnp', pd.Series(0, index=df.index)) +
                df.get('filter_csnp', pd.Series(0, index=df.index)) +
                df.get('time_csnp_pages', pd.Series(0, index=df.index))
            ) * 4.0 - df.get('ma_drug_benefit', pd.Series(0, index=df.index)) * (
                df.get('query_drug', pd.Series(0, index=df.index)) +
                df.get('filter_drug', pd.Series(0, index=df.index)) +
                df.get('time_drug_pages', pd.Series(0, index=df.index))
            )
        ).clip(lower=0) * 8.0
        additional_features.append('csnp_drug_interaction')
        
        df['csnp_specific_signal'] = (
            df.get('query_csnp', pd.Series(0, index=df.index)) +
            df.get('filter_csnp', pd.Series(0, index=df.index)) +
            df.get('csnp_drug_interaction', pd.Series(0, index=df.index)) +
            df.get('csnp_doctor_interaction', pd.Series(0, index=df.index)) +
            df.get('csnp_vision_interaction', pd.Series(0, index=df.index)) +
            df.get('csnp_dental_interaction', pd.Series(0, index=df.index))
        ).clip(upper=5) * 8.0
        additional_features.append('csnp_specific_signal')
        logger.info("Interaction features calculated successfully")
        
        # Feature selection
        feature_cols = (
            [col for col in plan_features if col in df.columns] +
            [col for col in behavioral_features if col in df.columns] +
            additional_features + embedding_cols + [f'{p}_weight' for p in PERSONAS] + ['recency']
        )
        X = df[feature_cols].fillna(0)
        y = df['persona']
        
        # Final persona validation
        if not set(y.unique()).intersection(PERSONAS):
            logger.error(f"No valid personas in y: {y.unique()}")
            logger.warning("Creating minimal DataFrame with all personas")
            df = pd.DataFrame({
                'persona': PERSONAS,
                'signal_strength': [0.1] * len(PERSONAS),
                'zip': ['00000'] * len(PERSONAS),
                'plan_id': ['dummy'] * len(PERSONAS)
            })
            for col in feature_cols:
                df[col] = 0
            X = df[feature_cols].fillna(0)
            y = df['persona']
        
        logger.info(f"Final persona distribution:\n{y.value_counts(dropna=False).to_string()}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def train_model():
    # Load data
    behavioral_df, plan_df = load_data()
    X, y = prepare_features(behavioral_df, plan_df)
    
    # Validate target classes
    missing_personas = [p for p in PERSONAS if p not in y.unique()]
    if missing_personas:
        logger.error(f"Target classes missing: {missing_personas}")
        raise ValueError(f"Target classes {missing_personas} not present in data")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Log pre-SMOTE distribution
    logger.info(f"Pre-SMOTE persona distribution:\n{pd.Series(y).value_counts().to_string()}")
    
    # Apply SMOTE
    class_counts = pd.Series(y).value_counts()
    sampling_strategy = {
        persona: max(count, 400 if persona == 'csnp' else 500 if persona == 'dsnp' else 2000) for persona, count in class_counts.items()
    }
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
    X, y_encoded = smote.fit_resample(X, y_encoded)
    logger.info(f"Rows after SMOTE: {len(X)}")
    logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(le.inverse_transform(y_encoded)).value_counts().to_string()}")
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    logger.info(f"Test set label distribution:\n{pd.Series(le.inverse_transform(y_test)).value_counts().to_string()}")
    
    # Train CatBoost
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    feature_importances = []
    
    # Class weights to prioritize csnp
    class_weights = {le.transform(['csnp'])[0]: 3.0, **{i: 1.0 for i in range(len(le.classes_)) if i != le.transform(['csnp'])[0]}}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        model = CatBoostClassifier(
            iterations=600,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10,
            loss_function='MultiClass',
            class_weights=class_weights,
            random_seed=42,
            verbose=50
        )
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=200
        )
        models.append(model)
        feature_importances.append(model.get_feature_importance())
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions
    y_pred_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    y_pred = np.argmax(y_pred_probas, axis=1)
    
    # Log prediction distribution
    logger.info(f"Prediction distribution:\n{pd.Series(le.inverse_transform(y_pred)).value_counts().to_string()}")
    
    # Log feature importances
    avg_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': avg_importances
    }).sort_values(by='Importance', ascending=False)
    logger.info("Feature Importances:\n" + importance_df.to_string())
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"Overall Accuracy: {acc * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
    if acc < 0.8:
        logger.warning(f"Accuracy {acc * 100:.2f}% is below target of 80%.")
    
    # Per-persona accuracy
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    logger.info("Per-Persona Accuracy (%):")
    for persona, acc in per_persona_accuracy.items():
        logger.info(f"  {persona}: {acc:.2f}%")
    
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models[0], f)
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved model, label encoder, and scaler to disk.")

if __name__ == '__main__':
    train_model()
