import pandas as pd
import numpy as np
import pickle
import os
import io
import json
import logging
from azure.cosmos import CosmosClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
import uvicorn

# Set up logging with INFO level (temporarily include DEBUG for debugging)
logging.basicConfig(level=logging.DEBUG)  # Switch back to INFO after debugging
logger = logging.getLogger(__name__)

# Global variables
model = None
behavioral_df_full = None
plan_df_full = None
app_ready = False  # For AKS health check

# File definitions
MODEL_FILE = "rf_model_persona_with_weights_092024_032025_v7.pkl"

# FastAPI app instance
app = FastAPI()

# Input schema for the /score endpoint
class ScoreRequest(BaseModel):
    userid: str

# Function to load data from Cosmos DB into pandas DataFrame
def load_data_from_cosmos(container):
    query = "SELECT * FROM c"
    items = list(container.query_items(query, enable_cross_partition_query=True))
    return pd.DataFrame(items)

def init():
    """Initialize the model and load full datasets."""
    global model, behavioral_df_full, plan_df_full, app_ready
    logger.debug("Entering init function")

    try:
        # Load model
        model_path = MODEL_FILE
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")

        # Cosmos DB connection
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        database_name = os.getenv("COSMOS_PERSONA_DB")
        behavioral_container_name = os.getenv("COSMOS_PERSONA_CONTAINER_BEHAVIOR")
        plan_container_name = os.getenv("COSMOS_PERSONA_CONTAINER_PLAN")

        if not all([endpoint, key, database_name, behavioral_container_name, plan_container_name]):
            missing_vars = [var for var, val in [
                ("COSMOS_ENDPOINT", endpoint),
                ("COSMOS_KEY", key),
                ("COSMOS_PERSONA_DB", database_name),
                ("COSMOS_PERSONA_CONTAINER_BEHAVIOR", behavioral_container_name),
                ("COSMOS_PERSONA_CONTAINER_PLAN", plan_container_name)
            ] if not val]
            logger.error(f"Missing environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")

        # Initialize Cosmos DB client
        try:
            client = CosmosClient(endpoint, credential=key)
            logger.info("Successfully connected to Cosmos DB")
        except Exception as e:
            logger.error(f"Failed to connect to Cosmos DB: {str(e)}")
            raise

        database = client.get_database_client(database_name)
        behavioral_container = database.get_container_client(behavioral_container_name)
        plan_container = database.get_container_client(plan_container_name)

        # Load data
        behavioral_df_full = load_data_from_cosmos(behavioral_container)
        plan_df_full = load_data_from_cosmos(plan_container)

        # Convert numeric columns to appropriate types
        numeric_columns = [
            'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
            'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug',
            'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp', 'accordion_dental',
            'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
            'accordion_vision', 'accordion_csnp', 'accordion_dsnp', 'time_dental_pages',
            'time_transportation_pages', 'time_otc_pages', 'time_drug_pages', 'time_provider_pages',
            'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages', 'rel_time_dental_pages',
            'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
            'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages',
            'rel_time_dsnp_pages', 'total_session_time', 'num_pages_viewed', 'num_plans_selected',
            'num_plans_compared', 'submitted_application', 'dce_click_count', 'pro_click_count',
            'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
            'ma_provider_network', 'ma_drug_coverage'
        ]

        for df in [behavioral_df_full, plan_df_full]:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure 'zip' and 'plan_id' are strings to avoid merge issues
        for df in [behavioral_df_full, plan_df_full]:
            df['zip'] = df['zip'].astype(str).fillna('')
            df['plan_id'] = df['plan_id'].astype(str).fillna('')

        logger.info(f"Behavioral data loaded: {len(behavioral_df_full)} rows")
        logger.info(f"Plan data loaded: {len(plan_df_full)} rows")

        app_ready = True
        logger.info("Initialization completed successfully")

    except Exception as e:
        logger.error(f"Error in init: {str(e)}")
        raise  # Ensure startup fails if init fails

# Feature preparation and scoring functions (aligned with updated score.py)
def normalize_persona(df):
    """Normalize personas by splitting rows with multiple personas (e.g., 'csnp,dsnp') into separate rows."""
    new_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row['persona']):
            new_rows.append(row)
            continue
        personas = [p.strip().lower() for p in str(row['persona']).split(',')]
        if 'dsnp' in personas or 'csnp' in personas:
            first_row = row.copy()
            first_persona = personas[0]
            if first_persona in ['unknown', 'none', 'healthcare', '']:
                first_persona = 'dsnp' if 'dsnp' in personas else 'csnp'
            first_row['persona'] = first_persona
            new_rows.append(first_row)
            second_row = row.copy()
            second_row['persona'] = 'dsnp' if 'dsnp' in personas else 'csnp'
            new_rows.append(second_row)
        else:
            row_copy = row.copy()
            first_persona = personas[0]
            row_copy['persona'] = first_persona
            new_rows.append(row_copy)
    return pd.DataFrame(new_rows).reset_index(drop=True)

def prepare_features(behavioral_df, plan_df):
    """Prepare features and assign quality levels for scoring, joining with plan_df."""
    # Normalize personas and reset index
    behavioral_df = normalize_persona(behavioral_df)
    logger.info(f"Rows after persona normalization: {len(behavioral_df)}")

    # Ensure 'zip' and 'plan_id' are strings (already handled in init, but reinforcing here)
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')

    # Merge behavioral and plan data
    df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    ).reset_index(drop=True)
    logger.info(f"Rows after merge with plan data: {len(df)}")

    df['state'] = df['state_beh'].fillna(df['state_plan'])
    df = df.drop(columns=['state_beh', 'state_plan'], errors='ignore').reset_index(drop=True)

    # Define feature lists
    all_behavioral_features = [
        'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
        'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug',
        'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp', 'accordion_dental',
        'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
        'accordion_vision', 'accordion_csnp', 'accordion_dsnp', 'time_dental_pages',
        'time_transportation_pages', 'time_otc_pages', 'time_drug_pages', 'time_provider_pages',
        'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages', 'rel_time_dental_pages',
        'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
        'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages',
        'rel_time_dsnp_pages', 'total_session_time', 'num_pages_viewed', 'num_plans_selected',
        'num_plans_compared', 'submitted_application', 'dce_click_count', 'pro_click_count'
    ]

    raw_plan_features = [
        'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
        'ma_provider_network', 'ma_drug_coverage'
    ]

    # Ensure all raw_plan_features and csnp_type exist
    for col in raw_plan_features + ['csnp_type']:
        if col not in df.columns:
            logger.warning(f"'{col}' not found in df. Filling with 0.")
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # Compute quality level
    filter_cols = [col for col in df.columns if col.startswith('filter_')]
    query_cols = [col for col in df.columns if col.startswith('query_')]

    def assign_quality_level(row):
        has_plan_id = pd.notna(row['plan_id'])
        has_clicks = (row.get('dce_click_count', 0) > 0 and pd.notna(row.get('dce_click_count'))) or \
                     (row.get('pro_click_count', 0) > 0 and pd.notna(row.get('pro_click_count')))
        has_filters = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in filter_cols)
        has_queries = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in query_cols)
        
        if has_plan_id and (has_clicks or has_filters):
            return 'High'
        elif has_plan_id and not has_clicks and not has_filters and has_queries:
            return 'Medium'
        elif not has_plan_id and not has_clicks and not has_filters and not has_queries:
            return 'Low'
        else:
            return 'Medium'

    df['quality_level'] = df.apply(assign_quality_level, axis=1)
    df = df.reset_index(drop=True)

    # Compute additional features
    additional_features = []
    df['csnp_interaction'] = df['csnp'] * (
        df.get('query_csnp', 0).fillna(0) + df.get('filter_csnp', 0).fillna(0) + 
        df.get('time_csnp_pages', 0).fillna(0) + df.get('accordion_csnp', 0).fillna(0)
    ) * 2.5
    additional_features.append('csnp_interaction')

    df['csnp_type_flag'] = df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    df['csnp_signal_strength'] = (
        df.get('query_csnp', 0).fillna(0) + df.get('filter_csnp', 0).fillna(0) + 
        df.get('accordion_csnp', 0).fillna(0) + df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 2.5
    additional_features.append('csnp_signal_strength')

    df['dental_interaction'] = (
        df.get('query_dental', 0).fillna(0) + df.get('filter_dental', 0).fillna(0)
    ) * df['ma_dental_benefit'] * 1.5
    additional_features.append('dental_interaction')

    df['vision_interaction'] = (
        df.get('query_vision', 0).fillna(0) + df.get('filter_vision', 0).fillna(0)
    ) * df['ma_vision'] * 1.5
    additional_features.append('vision_interaction')

    df['csnp_drug_interaction'] = (
        df['csnp'] * (
            df.get('query_csnp', 0).fillna(0) + df.get('filter_csnp', 0).fillna(0) + 
            df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.0 - df['ma_drug_coverage'] * (
            df.get('query_drug', 0).fillna(0) + df.get('filter_drug', 0).fillna(0) + 
            df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 2.5
    additional_features.append('csnp_drug_interaction')

    df['csnp_doctor_interaction'] = (
        df['csnp'] * (
            df.get('query_csnp', 0).fillna(0) + df.get('filter_csnp', 0).fillna(0)
        ) * 1.5 - df['ma_provider_network'] * (
            df.get('query_provider', 0).fillna(0) + df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 1.5
    additional_features.append('csnp_doctor_interaction')

    df = df.reset_index(drop=True)

    # Debug: Check csnp features
    high_quality_csnp = df[(df['quality_level'] == 'High') & (df['persona'] == 'csnp')]
    logger.info(f"Scoring: High-quality csnp samples: {len(high_quality_csnp)}")
    logger.info(f"Scoring: Non-zero csnp_interaction: {sum(high_quality_csnp['csnp_interaction'] > 0)}")
    logger.info(f"Scoring: Non-zero csnp_drug_interaction: {sum(high_quality_csnp['csnp_drug_interaction'] > 0)}")
    logger.info(f"Scoring: Non-zero csnp_doctor_interaction: {sum(high_quality_csnp['csnp_doctor_interaction'] > 0)}")

    # Define persona weights for weighted features
    persona_weights = {
        'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'pro_click_count'},
        'drug': {'plan_col': 'ma_drug_coverage', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'dce_click_count'},
        'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'},
        'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
        'otc': {'plan_col': 'ma_otc', 'query_col': 'query_otc', 'filter_col': 'filter_otc'},
        'transportation': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation'},
        'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
        'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'}
    }

    # Use the updated constants from the evaluation script
    k1, k3, k4, k7, k8 = 0.15, 0.8, 0.7, 0.3, 0.4
    k9, k10 = 2.5, 2.3
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 3.0, 7.0, 1.2, 1.8

    def calculate_persona_weight(row, persona_info, persona, plan_df):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(row[plan_col], 0.7 if persona == 'csnp' else 0.5)
            if persona == 'csnp' and row.get('csnp_type', 'N') == 'Y':
                base_weight *= W_CSNP_HIGH
            elif persona == 'csnp':
                base_weight *= W_CSNP_BASE
            elif persona == 'dsnp' and row.get('dsnp_type', 'N') == 'Y':
                base_weight *= W_DSNP_HIGH
            elif persona == 'dsnp':
                base_weight *= W_DSNP_BASE
        elif pd.notna(row.get('compared_plan_ids')) and isinstance(row['compared_plan_ids'], str) and row.get('num_plans_compared', 0) > 0:
            compared_ids = row['compared_plan_ids'].split(',')
            compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
            if not compared_plans.empty and plan_col in compared_plans.columns:
                base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
                if persona == 'csnp' and 'csnp_type' in compared_plans.columns:
                    csnp_type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                    base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
                elif persona == 'dsnp' and 'dsnp_type' in compared_plans.columns:
                    dsnp_type_y_ratio = (compared_plans['dsnp_type'] == 'Y').mean()
                    base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)
            else:
                base_weight = 0
        else:
            base_weight = 0
        
        pages_viewed = min(row.get('num_pages_viewed', 0), 3) if pd.notna(row.get('num_pages_viewed')) else 0
        query_value = row.get(query_col, 0) if pd.notna(row.get(query_col)) else 0
        filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col)) else 0
        click_value = row.get(click_col, 0) if click_col and click_col in row and pd.notna(row.get(click_col)) else 0
        
        query_coeff = k9 if persona == 'csnp' else k3
        filter_coeff = k10 if persona == 'csnp' else k4
        click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0
        
        behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value
        
        has_filters = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in filter_cols)
        has_clicks = (row.get('dce_click_count', 0) > 0 and pd.notna(row.get('dce_click_count'))) or \
                     (row.get('pro_click_count', 0) > 0 and pd.notna(row.get('pro_click_count')))
        if has_filters and has_clicks:
            behavioral_score += 0.8
        elif has_filters or has_clicks:
            behavioral_score += 0.4
        
        if persona == 'doctor':
            if click_value >= 1.5: behavioral_score += 0.5
            elif click_value >= 0.5: behavioral_score += 0.25
        elif persona == 'drug':
            if click_value >= 5: behavioral_score += 0.5
            elif click_value >= 2: behavioral_score += 0.25
        elif persona == 'dental':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.7
            elif signal_count >= 1: behavioral_score += 0.4
            if row['quality_level'] == 'High': behavioral_score += 0.6
            if row.get('dental_interaction', 0) > 0: behavioral_score += 0.4
        elif persona == 'vision':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.6
            if row['quality_level'] == 'High': behavioral_score += 0.6
            if row.get('vision_interaction', 0) > 0: behavioral_score += 0.4
        elif persona == 'csnp':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 1.2
            elif signal_count >= 1: behavioral_score += 0.8
            if row.get('csnp_interaction', 0) > 0: behavioral_score += 1.2
            if row.get('csnp_type_flag', 0) == 1: behavioral_score += 1.0
            if row.get('csnp_drug_interaction', 0) > 0: behavioral_score += 0.8
            if row.get('csnp_doctor_interaction', 0) > 0: behavioral_score += 0.6
            if row['quality_level'] == 'High': behavioral_score += 1.5
        elif persona in ['otc', 'transportation']:
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.5
            if row['quality_level'] == 'High': behavioral_score += 0.5
        
        adjusted_weight = base_weight + behavioral_score

        if 'persona' in row and persona == row['persona']:
            non_target_weights = [
                min(row.get(info['plan_col'], 0), 0.5) * (
                    W_CSNP_HIGH if p == 'csnp' and row.get('csnp_type', 'N') == 'Y' else
                    W_CSNP_BASE if p == 'csnp' else
                    W_DSNP_HIGH if p == 'dsnp' and row.get('dsnp_type', 'N') == 'Y' else
                    W_DSNP_BASE if p == 'dsnp' else 1.0
                ) + (
                    k3 * (row.get(info['query_col'], 0) if pd.notna(row.get(info['query_col'])) else 0) +
                    k4 * (row.get(info['filter_col'], 0) if pd.notna(row.get(info['filter_col'])) else 0) +
                    k1 * pages_viewed +
                    (k8 if p == 'doctor' else k7 if p == 'drug' else 0) * 
                    (row.get(info.get('click_col'), 0) if 'click_col' in info and pd.notna(row.get(info.get('click_col'))) else 0) +
                    (0.5 if p == 'doctor' and row.get(info.get('click_col', 'pro_click_count'), 0) >= 1.5 else 
                     0.25 if p == 'doctor' and row.get(info.get('click_col', 'pro_click_count'), 0) >= 0.5 else 
                     0.5 if p == 'drug' and row.get(info.get('click_col', 'dce_click_count'), 0) >= 5 else 
                     0.25 if p == 'drug' and row.get(info.get('click_col', 'dce_click_count'), 0) >= 2 else 
                     0.7 if p == 'dental' and sum([1 for val in [row.get(info['query_col'], 0), row.get(info['filter_col'], 0), pages_viewed] if val > 0]) >= 2 else 
                     0.4 if p == 'dental' and sum([1 for val in [row.get(info['query_col'], 0), row.get(info['filter_col'], 0), pages_viewed] if val > 0]) >= 1 else 
                     0.6 if p == 'vision' and sum([1 for val in [row.get(info['query_col'], 0), row.get(info['filter_col'], 0), pages_viewed] if val > 0]) >= 1 else 
                     0.5 if p in ['csnp', 'otc', 'transportation'] and sum([1 for val in [row.get(info['query_col'], 0), row.get(info['filter_col'], 0), pages_viewed] if val > 0]) >= 1 else 0)
                )
                for p, info in persona_weights.items()
                if p != row['persona'] and info['plan_col'] in row and pd.notna(row.get(info['plan_col']))
            ]
            max_non_target = max(non_target_weights, default=0)
            adjusted_weight = max(adjusted_weight, max_non_target + 0.2)
        
        return min(adjusted_weight, 3.5 if persona == 'csnp' else 1.2)

    # Compute weighted features using Pandas apply
    logger.info("Calculating persona weights using Pandas apply...")
    for persona, info in persona_weights.items():
        click_col = info.get('click_col', 'click_dummy')
        if click_col not in df.columns:
            df[click_col] = 0
        df[f'w_{persona}'] = df.apply(
            lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1
        )

    # Normalize weighted features
    weighted_features = [f'w_{persona}' for persona in persona_weights.keys()]
    weight_sum = df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        df[wf] = df[wf] / weight_sum.where(weight_sum > 0, 1)

    logger.info(f"Weighted features added: {weighted_features}")
    logger.info("Sample weights:")
    logger.info(df[weighted_features].head())

    # Define all feature columns expected by the model
    all_weighted_features = [f'w_{persona}' for persona in [
        'doctor', 'drug', 'vision', 'dental', 'otc', 'transportation', 'csnp', 'dsnp'
    ]]
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + all_weighted_features

    logger.info(f"Feature columns expected: {feature_columns}")

    # Check for missing features and fill with 0
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing features in df: {missing_features}")
        for col in missing_features:
            df[col] = 0

    logger.info(f"Columns in df after filling: {df.columns.tolist()}")

    # Prepare features for prediction, ensuring 'userid' is included in metadata
    X = df[feature_columns].fillna(0)
    metadata = df[['userid', 'zip', 'plan_id', 'quality_level'] + feature_columns]

    return X, metadata

def score_data(model, X, metadata):
    """Score the data and return results with quality level and full persona probability rankings."""
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X)
    
    # Combine metadata with predictions
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred
    
    # Add probability ranking (full ranking, not just top 3)
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])
    
    # Select relevant columns for output
    final_df = output_df[['userid', 'quality_level', 'predicted_persona', 'probability_ranking']]

    # Log prediction distribution
    logger.info("\nDistribution of predicted personas:")
    logger.info(final_df['predicted_persona'].value_counts())

    # Log a sample of user IDs with their predicted personas
    logger.info("\nSample of user IDs with predicted personas:")
    logger.info(final_df[['userid', 'predicted_persona']].head())

    return final_df

# Define the /score endpoint
@app.post("/score")
async def score(request: ScoreRequest):
    """Score endpoint to process user data and return predictions."""
    if not app_ready:
        raise HTTPException(status_code=503, detail="Service is still initializing")

    try:
        userid = request.userid
        if not userid:
            raise HTTPException(status_code=400, detail="Missing 'userid' in request payload")

        behavioral_df = behavioral_df_full[behavioral_df_full['userid'] == userid]
        if behavioral_df.empty:
            raise HTTPException(status_code=404, detail=f"No behavioral data found for userid: {userid}")

        zip_plan_pairs = behavioral_df[['zip', 'plan_id']].drop_duplicates()
        plan_df = plan_df_full[plan_df_full.set_index(['zip', 'plan_id']).index.isin(zip_plan_pairs.set_index(['zip', 'plan_id']).index)]

        logger.info(f"Behavioral data rows for {userid}: {len(behavioral_df)}")
        logger.info(f"Plan data rows for {userid}: {len(plan_df)}")

        X, metadata = prepare_features(behavioral_df, plan_df)
        scored_df = score_data(model, X, metadata)

        result = []
        for _, row in scored_df.iterrows():
            output = {
                'userid': row['userid'],
                'quality_level': row['quality_level'],
                'predicted_persona': row['predicted_persona'],
                'probability_ranking': row['probability_ranking']
            }
            if row['quality_level'] == 'Low':
                output['message'] = 'Prediction result might not be accurate due to low behavioral data quality'
            result.append(output)

        return {'scored_results': result}

    except Exception as e:
        logger.error(f"Error during scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during scoring: {str(e)}")

# Health check endpoint for AKS
@app.get("/actuator/health")
async def health_check():
    """Check if the service is up and running."""
    if not app_ready:
        raise HTTPException(status_code=503, detail="Service is still initializing")
    return Response(content="Healthy", status_code=200)

# Main execution
if __name__ == "__main__":
    logger.debug("Starting application")
    init()  # Call init synchronously before running the server
    logger.debug("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
