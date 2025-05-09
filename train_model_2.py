import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
# Consider adding other imblearn techniques if needed, e.g., from imblearn.combine import SMOTEENN
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
logging.getLogger("py4j").setLevel(logging.ERROR)

# File paths
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.1.0.pkl'
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
    query_col = persona_info.get('query_col', None)
    filter_col = persona_info.get('filter_col', None)
    plan_col = persona_info.get('plan_col', None)
    click_col = persona_info.get('click_col', None)

    query_value = row.get(query_col, 0) if query_col and pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if filter_col and pd.notna(row.get(filter_col, np.nan)) else 0
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
        # Keeping 'dental' as default for now, but consider a more sophisticated imputation strategy
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

def prepare_features(behavioral_df, plan_df):
    try:
        behavioral_df = normalize_persona(behavioral_df)

        if behavioral_df.empty:
            logger.warning("Behavioral_df is empty after normalization. Using plan_df only.")
            training_df = plan_df.copy()
            training_df['persona'] = 'dental' # Default persona if no behavioral data
        else:
            training_df = behavioral_df.merge(
                plan_df.rename(columns={'StateCode': 'state'}),
                how='left', on=['zip', 'plan_id']
            ).reset_index(drop=True)
            logger.info(f"Rows after merge: {len(training_df)}")
            logger.info(f"training_df columns: {list(training_df.columns)}")

        # Initialize plan_features and behavioral_features ensuring all expected columns exist
        plan_features_expected = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network']
        behavioral_features_expected = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_provider', 'accordion_vision', 'accordion_dsnp'
        ]

        for col in plan_features_expected + behavioral_features_expected:
            if col not in training_df.columns:
                training_df[col] = 0
            else:
                # Impute missing numerical features with median
                if training_df[col].dtype in ['int64', 'float64']:
                     training_df[col] = training_df[col].fillna(training_df[col].median() if not training_df[col].isnull().all() else 0)
                # Impute missing categorical features with a placeholder
                else:
                    training_df[col] = training_df[col].fillna('missing')


        behavioral_features = [col for col in behavioral_features_expected if col in training_df.columns]
        plan_features = [col for col in plan_features_expected if col in training_df.columns]


        # Log feature sparsity
        sparsity_cols = [col for col in behavioral_features if training_df[col].dtype in ['int64', 'float64']]
        if sparsity_cols:
             sparsity_stats = training_df[sparsity_cols].describe().to_dict()
             logger.info(f"Feature sparsity stats:\n{sparsity_stats}")
        else:
             logger.info("No numerical behavioral features to calculate sparsity stats.")

        # Temporal features
        if 'start_time' in training_df.columns and pd.api.types.is_datetime64_any_dtype(training_df['start_time']):
            training_df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(training_df['start_time'])).dt.days.fillna(30).clip(lower=0) # Ensure non-negative recency
            training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / (training_df['recency'] + 1) # Frequency based on recency
            training_df['time_of_day'] = pd.to_datetime(training_df['start_time']).dt.hour.fillna(12) // 6
        else:
            logger.warning("start_time column not found or not in datetime format. Temporal features will be defaulted.")
            training_df['recency'] = 30
            training_df['visit_frequency'] = 1
            training_df['time_of_day'] = 2

        # Clustering feature
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        cluster_features_present = [col for col in cluster_features if col in training_df.columns]

        if len(cluster_features_present) == len(cluster_features):
            try:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) # Added n_init for KMeans
                training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
            except Exception as e:
                logger.warning(f"KMeans clustering failed: {e}. User cluster will be defaulted.")
                training_df['user_cluster'] = 0
        else:
            logger.warning("Required features for clustering missing. User cluster will be defaulted.")
            training_df['user_cluster'] = 0

        # Robust aggregates with added smoothing
        training_df['dental_time_ratio'] = training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 0) + 1e-5)
        training_df['click_ratio'] = training_df.get('num_clicks', 0) / (training_df.get('num_pages_viewed', 0) + 1e-5)

        # Plan ID embeddings
        if 'plan_id' in training_df.columns:
            plan_sentences = training_df.groupby('userid')['plan_id'].apply(list).tolist()
            if plan_sentences:
                w2v_model = Word2Vec(sentences=plan_sentences, vector_size=20, window=5, min_count=1, workers=4) # Increased vector size
                plan_embeddings = training_df['plan_id'].apply(
                    lambda x: w2v_model.wv[x] if x in w2v_model.wv else np.zeros(20) # Match increased vector size
                )
                embedding_cols = [f'plan_emb_{i}' for i in range(20)]
                training_df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=training_df.index)
            else:
                 logger.warning("No plan ID sentences for Word2Vec. Plan embeddings will be zero.")
                 embedding_cols = [f'plan_emb_{i}' for i in range(20)]
                 training_df[embedding_cols] = 0
        else:
            logger.warning("plan_id column not found. Plan embeddings will be zero.")
            embedding_cols = [f'plan_emb_{i}' for i in range(20)]
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

        # Domain-specific features (refined and added features for doctor, drug, vision)
        additional_features = []

        # CSNP Features
        training_df['csnp_interaction'] = training_df.get('csnp', 0) * (
            training_df.get('query_csnp', 0) + training_df.get('filter_csnp', 0) +
            training_df.get('time_csnp_pages', 0) + training_df.get('accordion_csnp', 0)
        ) * 2.5
        additional_features.append('csnp_interaction')

        training_df['csnp_type_flag'] = training_df.get('csnp_type', 'N').map({'Y': 1, 'N': 0}).fillna(0).astype(int)
        additional_features.append('csnp_type_flag')

        training_df['csnp_signal_strength'] = (
            training_df.get('query_csnp', 0) + training_df.get('filter_csnp', 0) +
            training_df.get('accordion_csnp', 0) + training_df.get('time_csnp_pages', 0)
        ).clip(upper=5) * 2.5
        additional_features.append('csnp_signal_strength')

        # Dental Features
        training_df['dental_interaction'] = (
            training_df.get('query_dental', 0) + training_df.get('filter_dental', 0)
        ) * training_df.get('ma_dental_benefit', 0) * 1.5
        additional_features.append('dental_interaction')

        training_df['dental_signal_strength'] = (
            training_df.get('query_dental', 0) +
            training_df.get('filter_dental', 0) +
            training_df.get('time_dental_pages', 0).clip(upper=5) +
            training_df.get('accordion_dental', 0)
        ).clip(lower=0, upper=5) * 2.0
        additional_features.append('dental_signal_strength')

        # Vision Features (Enhanced)
        training_df['vision_interaction'] = (
            training_df.get('query_vision', 0) + training_df.get('filter_vision', 0) + training_df.get('time_vision_pages', 0)
        ) * training_df.get('ma_vision', 0) * 2.0 # Increased weight
        additional_features.append('vision_interaction')

        training_df['vision_signal'] = (
            training_df.get('query_vision', 0) * 1.5 + # Increased weight for query
            training_df.get('filter_vision', 0) * 1.2 + # Increased weight for filter
            training_df.get('time_vision_pages', 0).clip(upper=5) +
            training_df.get('accordion_vision', 0)
        ) * 2.5 # Increased overall signal weight
        additional_features.append('vision_signal')

        # Drug Features (Enhanced)
        training_df['drug_interaction'] = (
             training_df.get('query_drug', 0) + training_df.get('filter_drug', 0) + training_df.get('time_drug_pages', 0)
        ) * training_df.get('ma_drug_benefit', 0) * 2.0 # Increased weight
        additional_features.append('drug_interaction')

        training_df['drug_signal'] = (
            training_df.get('query_drug', 0) * 1.5 + # Increased weight for query
            training_df.get('filter_drug', 0) * 1.2 + # Increased weight for filter
            training_df.get('time_drug_pages', 0).clip(upper=5) +
            training_df.get('accordion_drug', 0)
        ) * 2.5 # Increased overall signal weight
        additional_features.append('drug_signal')

        # Doctor Features (Enhanced)
        training_df['doctor_interaction'] = (
            training_df.get('query_provider', 0) + training_df.get('filter_provider', 0) + training_df.get('click_provider', 0)
        ) * training_df.get('ma_provider_network', 0) * 2.0 # Increased weight
        additional_features.append('doctor_interaction')

        training_df['doctor_signal'] = (
            training_df.get('query_provider', 0) * 1.5 + # Increased weight for query
            training_df.get('filter_provider', 0) * 1.2 + # Increased weight for filter
            training_df.get('click_provider', 0)
        ).clip(lower=0, upper=5) * 2.5 # Increased overall signal weight
        additional_features.append('doctor_signal')

        # Interactions between different personas (Refined)
        training_df['csnp_drug_interaction'] = (
            training_df.get('csnp', 0) * (
                training_df.get('query_csnp', 0) + training_df.get('filter_csnp', 0) +
                training_df.get('time_csnp_pages', 0)
            ) * 2.0 - training_df.get('ma_drug_benefit', 0) * (
                training_df.get('query_drug', 0) + training_df.get('filter_drug', 0) +
                training_df.get('time_drug_pages', 0)
            )
        ).clip(lower=0) * 2.5
        additional_features.append('csnp_drug_interaction')

        training_df['csnp_doctor_interaction'] = (
            training_df.get('csnp', 0) * (
                training_df.get('query_csnp', 0) + training_df.get('filter_csnp', 0)
            ) * 1.5 - training_df.get('ma_provider_network', 0) * (
                training_df.get('query_provider', 0) + training_df.get('filter_provider', 0)
            )
        ).clip(lower=0) * 1.5
        additional_features.append('csnp_doctor_interaction')


        training_df['csnp_specific_signal'] = (
            training_df.get('query_csnp', 0) +
            training_df.get('filter_csnp', 0) +
            training_df.get('csnp_drug_interaction', 0) +
            training_df.get('csnp_doctor_interaction', 0)
        ).clip(upper=5) * 3.0
        additional_features.append('csnp_specific_signal')

        training_df['dsnp_signal'] = (
            training_df.get('query_dsnp', 0) +
            training_df.get('filter_dsnp', 0) +
            training_df.get('time_dsnp_pages', 0).clip(upper=5) +
            training_df.get('accordion_dsnp', 0)
        ) * 2.5
        additional_features.append('dsnp_signal')

        training_df['dsnp_proxy_signal'] = (
            training_df.get('query_csnp', 0) * 0.5 +  # Proxy for DSNP
            training_df.get('filter_dsnp', 0) +
            training_df.get('time_dsnp_pages', 0).clip(upper=5)
        ) * 2.0
        additional_features.append('dsnp_proxy_signal')


        training_df['dsnp_drug_interaction'] = (
            training_df.get('dsnp', 0) * (
                training_df.get('query_dsnp', 0) + training_df.get('filter_dsnp', 0) +
                training_df.get('time_dsnp_pages', 0)
            ) * 2.0 - training_df.get('ma_drug_benefit', 0) * (
                training_df.get('query_drug', 0) + training_df.get('filter_drug', 0) +
                training_df.get('time_drug_pages', 0)
            )
        ).clip(lower=0) * 2.5
        additional_features.append('dsnp_drug_interaction')

        training_df['drug_doctor_interaction'] = (
            training_df.get('ma_drug_benefit', 0) * (
                training_df.get('query_drug', 0) + training_df.get('filter_drug', 0) +
                training_df.get('time_drug_pages', 0)
            ) * 1.5 - training_df.get('ma_provider_network', 0) * (
                training_df.get('query_provider', 0) + training_df.get('filter_provider', 0)
            )
        ).clip(lower=0) * 1.5
        additional_features.append('drug_doctor_interaction')


        # Label validation (Refined to handle potential missing plan columns gracefully)
        mismatched_dsnp = training_df[(training_df['persona'] == 'dsnp') & (training_df.get('dsnp', 0) == 0)]
        mismatched_csnp = training_df[(training_df['persona'] == 'csnp') & (training_df.get('csnp', 0) == 0)]
        mismatched_dental = training_df[(training_df['persona'] == 'dental') & (training_df.get('ma_dental_benefit', 0) == 0)]

        logger.info(f"Label mismatches: {len(mismatched_dsnp) + len(mismatched_csnp) + len(mismatched_dental)}")

        if len(mismatched_dsnp) > 0:
             training_df.loc[mismatched_dsnp.index, 'persona'] = training_df.loc[mismatched_dsnp.index, 'dsnp'].map({1: 'dsnp'}).fillna('dental') # Default to dental if dsnp is 0
        if len(mismatched_csnp) > 0:
             training_df.loc[mismatched_csnp.index, 'persona'] = training_df.loc[mismatched_csnp.index, 'csnp'].map({1: 'csnp'}).fillna('dental') # Default to dental if csnp is 0
        if len(mismatched_dental) > 0:
             training_df.loc[mismatched_dental.index, 'persona'] = training_df.loc[mismatched_dental.index, 'ma_dental_benefit'].map({1: 'dental'}).fillna('vision') # Default to vision if dental is 0

        # Feature selection
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster',
            'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]

        # Filter down to columns actually present in the dataframe
        feature_columns_present = [col for col in feature_columns if col in training_df.columns]

        X = training_df[feature_columns_present].fillna(0) # Ensure no NaNs in features before variance check

        variances = X.var()
        valid_features = variances[variances > 1e-4].index.tolist()  # Adjusted variance threshold
        X = X[valid_features]
        logger.info(f"Selected features after variance filtering: {valid_features}")

        y = training_df['persona']

        # Separate labeled and unlabeled data based on persona being NaN
        labeled_mask = y.notna()
        X_labeled = X[labeled_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        y_labeled = y[labeled_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        X_unlabeled = X[~labeled_mask].copy() # Use .copy() to avoid SettingWithCopyWarning

        logger.info(f"Labeled data rows: {len(X_labeled)}")
        logger.info(f"Unlabeled data rows: {len(X_unlabeled)}")

        logger.info(f"Pre-SMOTE persona distribution (Labeled Data):\n{y_labeled.value_counts(dropna=False).to_string()}")

        if X_labeled.empty:
            logger.error("No labeled data with valid personas after filtering")
            raise ValueError("No labeled data with valid personas after filtering")

        # Critical personas that must be present in labeled data for stratification and SMOTE
        critical_personas = ['dental', 'csnp', 'dsnp']
        missing_critical_personas_labeled = [p for p in critical_personas if p not in y_labeled.unique()]

        # Add synthetic samples for missing critical personas in labeled data before SMOTE
        if missing_critical_personas_labeled:
            logger.warning(f"Critical personas missing in labeled data: {missing_critical_personas_labeled}. Adding synthetic samples.")

            synthetic_data_labeled = []
            for persona in missing_critical_personas_labeled:
                logger.info(f"Adding synthetic data for missing labeled persona: {persona}")
                # Add a small number of synthetic samples (e.g., 10)
                for i in range(10):
                    sample = {col: 0 for col in X_labeled.columns}
                    # Set some basic feature values and persona-specific indicators
                    if 'recency' in X_labeled.columns:
                        sample['recency'] = np.random.randint(1, 30)
                    if 'visit_frequency' in X_labeled.columns:
                        sample['visit_frequency'] = np.random.uniform(0.1, 0.5)
                    if 'time_of_day' in X_labeled.columns:
                        sample['time_of_day'] = np.random.randint(0, 4)
                    if 'user_cluster' in X_labeled.columns:
                        sample['user_cluster'] = np.random.randint(0, 5)
                    if persona in ['csnp', 'dsnp', 'dental', 'drug', 'vision', 'doctor']:
                         # Set relevant plan and behavioral features to indicate the persona
                         if PERSONA_INFO[persona].get('plan_col') and PERSONA_INFO[persona]['plan_col'] in X_labeled.columns:
                              sample[PERSONA_INFO[persona]['plan_col']] = 1
                         if PERSONA_INFO[persona].get('query_col') and PERSONA_INFO[persona]['query_col'] in X_labeled.columns:
                              sample[PERSONA_INFO[persona]['query_col']] = np.random.uniform(1, 5)
                         if PERSONA_INFO[persona].get('filter_col') and PERSONA_INFO[persona]['filter_col'] in X_labeled.columns:
                              sample[PERSONA_INFO[persona]['filter_col']] = np.random.uniform(1, 5)

                    synthetic_data_labeled.append(sample)

            if synthetic_data_labeled:
                synthetic_df_labeled = pd.DataFrame(synthetic_data_labeled)
                # Ensure synthetic_df_labeled has the same columns as X_labeled
                synthetic_df_labeled = synthetic_df_labeled.reindex(columns=X_labeled.columns, fill_value=0)

                X_labeled = pd.concat([X_labeled, synthetic_df_labeled], ignore_index=True)
                y_labeled = pd.concat([y_labeled, pd.Series([persona for persona in missing_critical_personas_labeled for _ in range(10)])], ignore_index=True)

                logger.info(f"Added {len(synthetic_data_labeled)} synthetic samples to labeled data.")
                logger.info(f"Labeled data distribution after adding synthetic samples:\n{y_labeled.value_counts().to_string()}")

        # Scale labeled data BEFORE SMOTE
        scaler = StandardScaler()
        X_labeled_scaled = pd.DataFrame(scaler.fit_transform(X_labeled), columns=X_labeled.columns)

        # Apply SMOTE with refined targets - further reduced targets
        class_counts_labeled = pd.Series(y_labeled).value_counts()
        # Adjusted sampling strategy: further reduced targets for less aggressive oversampling
        sampling_strategy_labeled = {
            'dsnp': max(class_counts_labeled.get('dsnp', 0), 1200), # Reduced target
            'doctor': max(class_counts_labeled.get('doctor', 0), 1500), # Reduced target
            'drug': max(class_counts_labeled.get('drug', 0), 1500),     # Reduced target
            'vision': max(class_counts_labeled.get('vision', 0), 1500), # Reduced target
            'csnp': max(class_counts_labeled.get('csnp', 0), 1200), # Reduced target
            'dental': max(class_counts_labeled.get('dental', 0), 1500)  # Reduced target
        }
        logger.info(f"SMOTE sampling strategy (Labeled Data): {sampling_strategy_labeled}")
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy_labeled)
        X_labeled_resampled, y_labeled_resampled = smote.fit_resample(X_labeled_scaled, y_labeled) # Apply SMOTE on scaled data

        logger.info(f"Labeled data rows after SMOTE: {len(X_labeled_resampled)}")
        logger.info(f"Post-SMOTE persona distribution (Labeled Data):\n{pd.Series(y_labeled_resampled).value_counts().to_string()}")


        # Scale unlabeled data using the same scaler fitted on labeled data
        if not X_unlabeled.empty:
             X_unlabeled_scaled = pd.DataFrame(scaler.transform(X_unlabeled), columns=X_unlabeled.columns)
        else:
             X_unlabeled_scaled = pd.DataFrame(columns=X_labeled_scaled.columns) # Create empty df with correct columns


        return X_labeled_resampled, y_labeled_resampled, X_unlabeled_scaled, scaler
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def pseudo_labeling(X_labeled, y_labeled, X_unlabeled, model, confidence_threshold=0.95):
    try:
        if X_labeled.empty or X_unlabeled.empty:
            logger.warning("Labeled or unlabeled data is empty. Skipping pseudo-labeling.")
            return pd.DataFrame(), np.array([])

        model.fit(X_labeled, y_labeled)
        y_unlabeled_pred_proba = model.predict_proba(X_unlabeled)
        confidence = y_unlabeled_pred_proba.max(axis=1)
        y_unlabeled_pred = model.classes_[y_unlabeled_pred_proba.argmax(axis=1)] # Get predicted class labels

        high_conf_mask = confidence > confidence_threshold
        X_pseudo = X_unlabeled[high_conf_mask]
        y_pseudo = y_unlabeled_pred[high_conf_mask] # Use predicted class labels

        logger.info(f"Pseudo-labeled {len(X_pseudo)} samples with confidence > {confidence_threshold}")
        return X_pseudo, y_pseudo
    except Exception as e:
        logger.warning(f"Pseudo-labeling failed: {e}")
        return pd.DataFrame(), np.array([])


def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        # Ensure that the class index exists in y_true before filtering
        if cls_idx in y_true:
            mask = y_true == cls_idx
            if mask.sum() > 0:
                # Ensure that the class index exists in y_pred before calculating accuracy
                if cls_idx in y_pred[mask]:
                     cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                     per_persona_accuracy[cls_name] = cls_accuracy * 100
                else:
                     # If the class is in y_true but not predicted for any of the masked samples
                     per_persona_accuracy[cls_name] = 0.0
            else:
                per_persona_accuracy[cls_name] = 0.0
        else:
             per_persona_accuracy[cls_name] = 0.0 # Assign 0 if class index is not in y_true
    return per_persona_accuracy


def main():
    # Load data
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Prepare features and handle labeled/unlabeled data
    try:
        X_labeled_resampled, y_labeled_resampled, X_unlabeled_scaled, scaler = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return

    # Split labeled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled_resampled, y_labeled_resampled, test_size=0.25, random_state=42, stratify=y_labeled_resampled
    )
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Log prediction distribution to check bias
    logger.info(f"Test set label distribution:\n{pd.Series(y_test).value_counts().to_string()}")

    # Train CatBoost with pseudo-labeling and k-fold
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    models = []
    feature_importances = []

    # Fixed pseudo-labeling confidence threshold - Reverting slightly
    pseudo_label_threshold = 0.95 # Reverted threshold
    logger.info(f"Using fixed pseudo-labeling confidence threshold: {pseudo_label_threshold:.2f}")


    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train_encoded[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train_encoded[val_idx]

        model = CatBoostClassifier(
            iterations=300,  # Further reduced iterations
            depth=4,  # Further reduced depth
            learning_rate=0.05,  # Slightly increased learning rate
            l2_leaf_reg=7,  # Increased L2 regularization
            loss_function='MultiClass',
            auto_class_weights='Balanced', # Keep auto-balancing for now
            random_seed=42,
            verbose=0,
            one_hot_max_size=10,
            eval_metric='Accuracy'
        )
        # Pass the fixed threshold to pseudo_labeling
        X_pseudo, y_pseudo_labels = pseudo_labeling(X_fold_train, y_fold_train, X_unlabeled_scaled, model, confidence_threshold=pseudo_label_threshold)

        if not X_pseudo.empty:
             # Ensure y_pseudo_labels are encoded before concatenating
             y_pseudo_encoded = le.transform(y_pseudo_labels)
             X_fold_train = pd.concat([X_fold_train, X_pseudo])
             y_fold_train = np.concatenate([y_fold_train, y_pseudo_encoded])


        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=40 # Reduced early stopping rounds
        )
        models.append(model)
        # Get feature importances from the trained model before ensemble
        feature_importances.append(model.get_feature_importance())
        logger.info(f"Fold {fold+1} training completed")

    # Ensemble predictions
    # Use predict_proba from all models and average
    y_pred_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    y_pred_encoded = np.argmax(y_pred_probas, axis=1)
    y_pred = le.inverse_transform(y_pred_encoded) # Decode predictions for evaluation

    # Log prediction distribution
    logger.info(f"Prediction distribution:\n{pd.Series(y_pred).value_counts().to_string()}")

    # Log feature importances
    if feature_importances and X_train.columns.tolist(): # Check if feature_importances is not empty and get columns from X_train
        avg_importances = np.mean(feature_importances, axis=0)
        importance_df = pd.DataFrame({
            'Feature': X_train.columns.tolist(), # Use columns from X_train after split
            'Importance': avg_importances
        }).sort_values(by='Importance', ascending=False)
        logger.info("Feature Importances:\n" + importance_df.to_string())
    else:
        logger.warning("No feature importances to display or X_train columns are not available.")


    # Evaluate
    overall_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    macro_f1 = f1_score(y_test_encoded, y_pred_encoded, average='macro')
    logger.info(f"Overall Accuracy on Test Set: {overall_accuracy * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")

    if overall_accuracy < 0.8:
        logger.warning(f"Accuracy {overall_accuracy * 100:.2f}% is below target of 80%.")

    # Compute per-persona accuracy using encoded labels for consistency with metrics
    per_persona_accuracy = compute_per_persona_accuracy(y_test_encoded, y_pred_encoded, le.classes_, le.classes_)
    logger.info("Per-Persona Accuracy (%):")
    for persona, acc in per_persona_accuracy.items():
        logger.info(f"  {persona}: {acc:.2f}%")

    logger.info("Classification Report:\n" + classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_))

    # Save model
    if models:
        # Consider saving the ensemble or a single best model based on validation performance
        # For simplicity, saving the first fold's model for now
        model = models[0]
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(le, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler)
        logger.info("Saved model, label encoder, and scaler to disk.")
    else:
        logger.error("No models were trained successfully.")


if __name__ == "__main__":
    main()
