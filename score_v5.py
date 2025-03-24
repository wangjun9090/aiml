import pandas as pd
import numpy as np
import pickle
import os
import json
import logging

# Global variable to hold the model
model = None

# File definitions (relative to AZUREML_MODEL_DIR)
BEHAVIORAL_FILE = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'v5/normalized_behavioral_features_0901_2024_0228_2025.csv')
PLAN_FILE = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'v5/plan_derivation_by_zip.csv')
MODEL_FILE = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'v5/rf_model_persona.pkl')

def init():
    """Initialize the model and any global resources for Azure ML endpoint."""
    global model
    try:
        model_path = MODEL_FILE
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def load_model(model_path):
    """Load the trained model from a pickle file (for local testing)."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_features(behavioral_df, plan_df):
    """Prepare features and assign quality levels for scoring, joining with plan_df."""
    df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    logging.info(f"Rows after merge with plan data: {len(df)}")

    df['state'] = df['state_beh'].fillna(df['state_plan'])
    df = df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

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
            if persona == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y':
                base_weight *= W_CSNP_HIGH
            elif persona == 'csnp':
                base_weight *= W_CSNP_BASE
            elif persona == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y':
                base_weight *= W_DSNP_HIGH
            elif persona == 'dsnp':
                base_weight *= W_DSNP_BASE
        elif pd.isna(row['plan_id']) and pd.notna(row['compared_plan_ids']) and isinstance(row['compared_plan_ids'], str) and row['num_plans_compared'] > 0:
            compared_ids = row['compared_plan_ids'].split(',')
            compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
            if not compared_plans.empty:
                base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
                if persona == 'csnp':
                    csnp_type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                    base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
                elif persona == 'dsnp':
                    dsnp_type_y_ratio = (compared_plans['dsnp_type'] == 'Y').mean()
                    base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)
        
        pages_viewed = min(row['num_pages_viewed'], 3) if pd.notna(row['num_pages_viewed']) else 0
        query_value = row[query_col] if pd.notna(row[query_col]) else 0
        filter_value = row[filter_col] if pd.notna(row[filter_col]) else 0
        click_value = row[click_col] if click_col and pd.notna(row[click_col]) else 0
        
        query_coeff = k9 if persona == 'csnp' else k3
        filter_coeff = k10 if persona == 'csnp' else k4
        click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0
        
        behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value
        
        if persona == 'doctor':
            if click_value >= 1.5: behavioral_score += 0.4
            elif click_value >= 0.5: behavioral_score += 0.2
        elif persona == 'drug':
            if click_value >= 5: behavioral_score += 0.4
            elif click_value >= 2: behavioral_score += 0.2
        elif persona == 'dental':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.3
            elif signal_count >= 1: behavioral_score += 0.15
        elif persona == 'vision':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.35
        elif persona == 'csnp':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.6
            elif signal_count >= 1: behavioral_score += 0.5
            if row['csnp_interaction'] > 0: behavioral_score += 0.3
            if row['csnp_type_flag'] == 1: behavioral_score += 0.2
        elif persona in ['fitness', 'hearing']:
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.3
        
        adjusted_weight = base_weight + behavioral_score
        return min(adjusted_weight, 2.0 if persona == 'csnp' else 1.0)

    for persona, info in persona_weights.items():
        df[f'w_{persona}'] = df.apply(lambda row: calculate_persona_weight(row, info, persona), axis=1)

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        df[wf] = df[wf] / weight_sum.where(weight_sum > 0, 1)

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    X = df[feature_columns].fillna(0)
    metadata = df[['userid', 'zip', 'plan_id']]

    def assign_quality_level(row):
        has_plan_id = pd.notna(row['plan_id'])
        has_clicks = (row['dce_click_count'] > 0 and pd.notna(row['dce_click_count'])) or \
                     (row['pro_click_count'] > 0 and pd.notna(row['pro_click_count']))
        has_filters = any(row[col] > 0 and pd.notna(row[col]) for col in df.columns if col.startswith('filter_'))
        has_queries = any(row[col] > 0 and pd.notna(row[col]) for col in df.columns if col.startswith('query_'))
        
        if has_plan_id and (has_clicks or has_filters):
            return 'High'
        elif has_plan_id and not has_clicks and not has_filters and has_queries:
            return 'Medium'
        elif not has_plan_id and not has_clicks and not has_filters and not has_queries:
            return 'Low'
        else:
            return 'Medium'

    df['quality_level'] = df.apply(assign_quality_level, axis=1)
    metadata['quality_level'] = df['quality_level']

    return X, metadata

def score_data(model, X, metadata):
    """Score the data and return results with quality level, predicted persona, and ranking."""
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X)

    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred

    output_df['persona_ranking'] = output_df.apply(
        lambda row: '; '.join([f"{p}: {row[f'prob_{p}']:.4f}" for p in sorted(personas, key=lambda x: row[f'prob_{x}'], reverse=True)]),
        axis=1
    )

    medium_df = output_df[output_df['quality_level'] == 'Medium']
    medium_avg_proba = {}
    if not medium_df.empty:
        medium_avg_proba = medium_df[[f'prob_{p}' for p in personas]].mean().to_dict()
        logging.info("\nAverage Prediction Probabilities for Medium Quality (Level 2):")
        for persona, avg_prob in medium_avg_proba.items():
            logging.info(f"{persona.replace('prob_', '')}: {avg_prob:.4f}")

    quality_summary = output_df['quality_level'].value_counts().to_dict()
    logging.info("\nData Quality Level Distribution:")
    for level, count in quality_summary.items():
        logging.info(f"{level}: {count} rows ({count / len(output_df) * 100:.2f}%)")

    return output_df, medium_avg_proba

def run(raw_data):
    """
    Run inference on input data for Azure ML endpoint.
    
    Args:
        raw_data: JSON string containing 'behavioral_data' and 'plan_data' as dictionaries
    
    Returns:
        JSON string with scored results and medium average probabilities
    """
    try:
        # Parse input JSON
        data = json.loads(raw_data)
        behavioral_data = data.get('behavioral_data', [])
        plan_data = data.get('plan_data', [])

        # Convert to DataFrames
        behavioral_df = pd.DataFrame(behavioral_data)
        plan_df = pd.DataFrame(plan_data)

        if behavioral_df.empty or plan_df.empty:
            raise ValueError("Input data is empty")

        logging.info(f"Behavioral data rows: {len(behavioral_df)}")
        logging.info(f"Plan data rows: {len(plan_df)}")

        # Prepare features
        X, metadata = prepare_features(behavioral_df, plan_df)

        # Score data
        scored_df, medium_avg_proba = score_data(model, X, metadata)

        # Convert results to JSON
        result = {
            'scored_results': scored_df.to_dict(orient='records'),
            'medium_avg_proba': medium_avg_proba
        }
        return json.dumps(result)

    except Exception as e:
        error_msg = f"Error during scoring: {str(e)}"
        logging.error(error_msg)
        return json.dumps({'error': error_msg})

# For local testing (remove for Azure ML deployment)
if __name__ == "__main__":
    # Load model locally
    local_model = load_model(MODEL_FILE)
    
    # Load sample data
    behavioral_df = pd.read_csv(BEHAVIORAL_FILE)
    plan_df = pd.read_csv(PLAN_FILE)
    
    # Prepare and score
    X, metadata = prepare_features(behavioral_df, plan_df)
    scored_df, medium_avg_proba = score_data(local_model, X, metadata)
    
    # Print results
    print(scored_df.head())
    print("Medium Avg Proba:", medium_avg_proba)
