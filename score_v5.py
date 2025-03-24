import pandas as pd
import numpy as np
import pickle
import os

# File definitions (for local reference; overridden by arguments or env vars in ML deployment)
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_behavioral_features_0901_2024_0228_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = os.path.join(os.getenv('AZUREML_MODEL_DIR', '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models'), 'rf_model_csnp_focus.pkl')

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")  # Replace with logging in production
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_features(behavioral_df, plan_df):
    """Prepare features and assign quality levels for scoring, joining with plan_df."""
    # Merge behavioral and plan data
    df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    print(f"Rows after merge with plan data: {len(df)}")

    # Resolve state column conflict
    df['state'] = df['state_beh'].fillna(df['state_plan'])
    df = df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    # Define feature sets
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

    # CSNP-specific features
    df['csnp_interaction'] = df['csnp'].fillna(0) * (
        df['query_csnp'].fillna(0) + df['filter_csnp'].fillna(0) + 
        df['time_csnp_pages'].fillna(0) + df['accordion_csnp'].fillna(0)
    ) * 2
    df['csnp_type_flag'] = (df['csnp_type'] == 'Y').astype(int) if 'csnp_type' in df.columns else 0
    df['csnp_signal_strength'] = (
        df['query_csnp'].fillna(0) + df['filter_csnp'].fillna(0) + 
        df['accordion_csnp'].fillna(0) + df['time_csnp_pages'].fillna(0)
    ).clip(upper=5) * 1.5

    additional_features = ['csnp_interaction', 'csnp_type_flag', 'csnp_signal_strength']

    # Persona weights setup (requires plan_df for plan features)
    persona_weights = {
        'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'pro_click_count'},
        'drug': {'plan_col': 'ma_drug_coverage', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'dce_click_count'},
        'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'},
        'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
        'otc': {'plan_col': 'ma_otc', 'query_col': 'query_otc', 'filter_col': 'filter_otc'},
        'transportation': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation'},
        'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
        'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
        'fitness': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation'},
        'hearing': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
    }

    k1, k3, k4, k7, k8 = 0.1, 0.5, 0.4, 0.15, 0.25
    k9, k10 = 1.0, 0.9
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.0, 3.0, 1.0, 1.5

    def calculate_persona_weight(row, persona_info, persona):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        
        base_weight = 0
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(row[plan_col], 0.7 if persona == 'csnp' else 0.5)
            if persona == 'csnp' and 'csnp_type
