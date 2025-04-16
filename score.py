import pandas as pd
import numpy as np
import pickle
import logging

# Set up logging with INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths (adjust these paths to match your Databricks environment)
MODEL_FILE = "/dbfs/path/to/rf_model_persona_with_weights_092024_032025_v6.pkl"
BEHAVIORAL_FILE = "/dbfs/path/to/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv"
PLAN_FILE = "/dbfs/path/to/plan_derivation_by_zip.csv"

# Hardcoded userid for testing
HARDCODED_USERID = "1743637276169V5LFCHK9S9U863J6JJL355C2RDKTHMRU"

def load_data(behavioral_path, plan_path):
    """Load behavioral and plan data from CSV files."""
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Behavioral data loaded: {len(behavioral_df)} rows")
        logger.info(f"Plan data loaded: {len(plan_df)} rows")

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

        for df in [behavioral_df, plan_df]:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def assign_quality_level(row, filter_cols, query_cols):
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

def prepare_features(behavioral_df, plan_df):
    """Prepare features and assign quality levels for scoring, joining with plan_df."""
    df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    logger.info(f"Rows after merge with plan data: {len(df)}")

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
    df['quality_level'] = df.apply(
        lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
    )

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

    k1, k3, k4, k7, k8 = 0.1, 0.7, 0.6, 0.25, 0.35
    k9, k10 = 2.2, 2.0
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 2.5, 6.0, 1.0, 1.5

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
        return min(adjusted_weight, 3.5 if persona == 'csnp' else 1.2)

    logger.info("Calculating persona weights...")
    for persona, info in persona_weights.items():
        df[f'w_{persona}'] = df.apply(
            lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1
        )

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys()]
    weight_sum = df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        df[wf] = df[wf] / weight_sum.where(weight_sum > 0, 1)

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    X = df[feature_columns].fillna(0)
    metadata = df[['userid']]
    metadata['quality_level'] = df['quality_level']
    return X, metadata

def score_data(model, X, metadata):
    """Score the data and return results with quality level and top 3 persona rankings."""
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)

    def get_top_3_ranking(row):
        probs = {p: row[f'prob_{p}'] for p in personas}
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        return {persona: round(prob, 4) for persona, prob in top_3}

    output_df['prediction_scores'] = output_df.apply(get_top_3_ranking, axis=1)
    final_df = output_df[['userid', 'quality_level', 'prediction_scores']]
    return final_df

def main():
    """Main function to load data, model, and score for the hardcoded userid."""
    try:
        # Load the model
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_FILE}")

        # Load data
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)

        # Filter for the hardcoded userid
        userid = HARDCODED_USERID
        behavioral_df = behavioral_df[behavioral_df['userid'] == userid]
        if behavioral_df.empty:
            logger.error(f"No behavioral data found for userid: {userid}")
            return {"error": f"No behavioral data found for userid: {userid}"}

        zip_plan_pairs = behavioral_df[['zip', 'plan_id']].drop_duplicates()
        plan_df = plan_df[plan_df.set_index(['zip', 'plan_id']).index.isin(zip_plan_pairs.set_index(['zip', 'plan_id']).index)]

        logger.info(f"Behavioral data rows for {userid}: {len(behavioral_df)}")
        logger.info(f"Plan data rows for {userid}: {len(plan_df)}")

        # Prepare features and score
        X, metadata = prepare_features(behavioral_df, plan_df)
        scored_df = score_data(model, X, metadata)

        # Format the result
        result = []
        for _, row in scored_df.iterrows():
            output = {
                'userid': row['userid'],
                'quality_level': row['quality_level'],
                'prediction_scores': row['prediction_scores']
            }
            if row['quality_level'] == 'Low':
                output['message'] = 'prediction result might not be accurate due to low behavioral data quality'
            result.append(output)

        return {'scored_results': result}

    except Exception as e:
        logger.error(f"Error during scoring: {str(e)}")
        return {"error": f"Error during scoring: {str(e)}"}

if __name__ == "__main__":
    # Run the scoring and print the result
    result = main()
    print(result)
