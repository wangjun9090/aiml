import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_092024_032025.csv'  # Combined data
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_with_weights_092024_032025.pkl'
weighted_behavioral_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/weighted_us_dce_pro_behavioral_features_092024_032025.csv'

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Plan data loaded: {len(plan_df)} rows")
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_training_features(behavioral_df, plan_df):
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    print(f"Rows after merge: {len(training_df)}")
    print("Columns after merge:", training_df.columns.tolist())

    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

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

    additional_features = []
    if 'csnp' in training_df.columns:
        training_df['csnp_interaction'] = training_df['csnp'].fillna(0) * (
            training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
            training_df['time_csnp_pages'].fillna(0) + training_df['accordion_csnp'].fillna(0)
        ) * 2
        additional_features.append('csnp_interaction')
    else:
        training_df['csnp_interaction'] = 0
        print("Warning: 'csnp' column not found. Setting 'csnp_interaction' to 0.")

    if 'csnp_type' in training_df.columns:
        training_df['csnp_type_flag'] = (training_df['csnp_type'] == 'Y').astype(int)
        additional_features.append('csnp_type_flag')
    else:
        training_df['csnp_type_flag'] = 0
        print("Warning: 'csnp_type' column not found. Setting 'csnp_type_flag' to 0.")

    training_df['csnp_signal_strength'] = (
        training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
        training_df['accordion_csnp'].fillna(0) + training_df['time_csnp_pages'].fillna(0)
    ).clip(upper=5) * 1.5
    additional_features.append('csnp_signal_strength')

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

    k1, k3, k4, k7, k8 = 0.1, 0.6, 0.5, 0.2, 0.3  # Increased k3, k4, k7, k8 to emphasize queries/filters/clicks
    k9, k10 = 1.2, 1.1  # Increased for csnp-specific signals
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.2, 3.5, 1.0, 1.5  # Boosted csnp weights

    def calculate_persona_weight(row, persona_info, persona, plan_df):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        
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
        
        pages_viewed = min(row['num_pages_viewed'], 3) if pd.notna(row['num_pages_viewed']) else 0
        query_value = row[query_col] if pd.notna(row[query_col]) else 0
        filter_value = row[filter_col] if pd.notna(row[filter_col]) else 0
        click_value = row[click_col] if click_col and click_col in row and pd.notna(row[click_col]) else 0
        
        query_coeff = k9 if persona == 'csnp' else k3
        filter_coeff = k10 if persona == 'csnp' else k4
        click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0
        
        behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value
        
        # Boost for high-quality signals
        has_filters = any(row[col] > 0 and pd.notna(row[col]) for col in training_df.columns if col.startswith('filter_'))
        has_clicks = (row.get('dce_click_count', 0) > 0 and pd.notna(row.get('dce_click_count'))) or \
                     (row.get('pro_click_count', 0) > 0 and pd.notna(row.get('pro_click_count')))
        if has_filters and has_clicks:
            behavioral_score += 0.5  # Extra boost for high-quality data
        
        if persona == 'doctor':
            if click_value >= 1.5: behavioral_score += 0.4
            elif click_value >= 0.5: behavioral_score += 0.2
        elif persona == 'drug':
            if click_value >= 5: behavioral_score += 0.4
            elif click_value >= 2: behavioral_score += 0.2
        elif persona == 'dental':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.4  # Increased
            elif signal_count >= 1: behavioral_score += 0.2
        elif persona == 'vision':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.4  # Increased
        elif persona == 'csnp':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.8  # Increased
            elif signal_count >= 1: behavioral_score += 0.6
            if 'csnp_interaction' in row and row['csnp_interaction'] > 0: behavioral_score += 0.4
            if 'csnp_type_flag' in row and row['csnp_type_flag'] == 1: behavioral_score += 0.3
        elif persona in ['fitness', 'hearing']:
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.4  # Increased
        
        adjusted_weight = base_weight + behavioral_score
        
        if 'persona' in row and persona == row['persona']:
            non_target_weights = [
                min(row[info['plan_col']], 0.5) * (
                    W_CSNP_HIGH if p == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y' else
                    W_CSNP_BASE if p == 'csnp' else
                    W_DSNP_HIGH if p == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y' else
                    W_DSNP_BASE if p == 'dsnp' else 1.0
                ) + (
                    k3 * (row[info['query_col']] if pd.notna(row[info['query_col']]) else 0) +
                    k4 * (row[info['filter_col']] if pd.notna(row[info['filter_col']]) else 0) +
                    k1 * pages_viewed +
                    (k8 if p == 'doctor' else k7 if p == 'drug' else 0) * 
                    (row[info.get('click_col')] if 'click_col' in info and pd.notna(row[info.get('click_col')]) else 0) +
                    (0.4 if p == 'doctor' and row[info.get('click_col', 'pro_click_count')] >= 1.5 else 
                     0.2 if p == 'doctor' and row[info.get('click_col', 'pro_click_count')] >= 0.5 else 
                     0.4 if p == 'drug' and row[info.get('click_col', 'dce_click_count')] >= 5 else 
                     0.2 if p == 'drug' and row[info.get('click_col', 'dce_click_count')] >= 2 else 
                     0.4 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 2 else 
                     0.2 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                     0.4 if p == 'vision' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                     0.4 if p in ['csnp', 'fitness', 'hearing'] and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 0)
                )
                for p, info in persona_weights.items()
                if p != row['persona'] and info['plan_col'] in row and pd.notna(row[info['plan_col']])
            ]
            max_non_target = max(non_target_weights, default=0)
            adjusted_weight = max(adjusted_weight, max_non_target + 0.15)
        
        return min(adjusted_weight, 2.0 if persona == 'csnp' else 1.0)

    print("Calculating persona weights...")
    for persona, info in persona_weights.items():
        training_df[f'w_{persona}'] = training_df.apply(lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1)

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)

    print(f"Weighted features added: {[col for col in training_df.columns if col.startswith('w_')]}")
    print("Sample weights:")
    print(training_df[[f'w_{persona}' for persona in persona_weights.keys()]].head())

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    valid_mask = training_df['persona'].notna() & (training_df['persona'] != '') & (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare']))
    training_df_valid = training_df[valid_mask]
    print(f"Rows after filtering invalid personas: {len(training_df_valid)}")

    X = training_df_valid[feature_columns].fillna(0)
    y = training_df_valid['persona']

    try:
        training_df.to_csv(weighted_behavioral_output_file, index=False)
        print(f"Behavioral data with weights saved to: {weighted_behavioral_output_file}")
    except Exception as e:
        print(f"Error saving weighted CSV: {e}")
        raise

    return X, y

def train_model(X, y):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X, y)
    print("Random Forest model trained.")
    return rf_model

def save_model(model, output_path):
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    print("Training Random Forest model with weighted features...")
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)
    X, y = prepare_training_features(behavioral_df, plan_df)
    rf_model = train_model(X, y)
    save_model(rf_model, model_output_file)

if __name__ == "__main__":
    main()
