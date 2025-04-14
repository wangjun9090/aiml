import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import resample

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_092024_032025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_with_weights_092024_032025_v3.pkl'
weighted_behavioral_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/weighted_us_dce_pro_behavioral_features_092024_032025_v3.csv'

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

    # Add missing plan features with zeros
    for col in raw_plan_features:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0

    # Compute quality level for oversampling
    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]
    training_df['quality_level'] = training_df.apply(
        lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
    )

    additional_features = []
    if 'csnp' in training_df.columns:
        csnp_col = training_df['csnp'].fillna(0)
    else:
        csnp_col = pd.Series(0, index=training_df.index)
    training_df['csnp_interaction'] = csnp_col * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 2
    additional_features.append('csnp_interaction')

    if 'csnp_type' in training_df.columns:
        training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    else:
        training_df['csnp_type_flag'] = 0
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 1.5
    additional_features.append('csnp_signal_strength')

    if 'ma_dental_benefit' in training_df.columns:
        dental_col = training_df['ma_dental_benefit'].fillna(0)
    else:
        dental_col = pd.Series(0, index=training_df.index)
    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * dental_col * 1.5
    additional_features.append('dental_interaction')

    if 'ma_vision' in training_df.columns:
        vision_col = training_df['ma_vision'].fillna(0)
    else:
        vision_col = pd.Series(0, index=training_df.index)
    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * vision_col * 1.5
    additional_features.append('vision_interaction')

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
    k9, k10 = 1.5, 1.3
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.5, 4.0, 1.0, 1.5

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
            if signal_count >= 2: behavioral_score += 0.8
            elif signal_count >= 1: behavioral_score += 0.5
            if row.get('csnp_interaction', 0) > 0: behavioral_score += 0.5
            if row.get('csnp_type_flag', 0) == 1: behavioral_score += 0.4
            if row['quality_level'] == 'High': behavioral_score += 0.6
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
        
        return min(adjusted_weight, 2.5 if persona == 'csnp' else 1.2)

    print("Calculating persona weights...")
    for persona, info in persona_weights.items():
        training_df[f'w_{persona}'] = training_df.apply(
            lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1
        )

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys()]
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)

    print(f"Weighted features added: {[col for col in training_df.columns if col.startswith('w_')]}")
    print("Sample weights:")
    print(training_df[[f'w_{persona}' for persona in persona_weights.keys()]].head())

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]

    # Filter invalid personas and exclude fitness, hearing
    valid_mask = (
        training_df['persona'].notna() & 
        (training_df['persona'] != '') & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    training_df_valid = training_df[valid_mask]
    print(f"Rows after filtering invalid personas and fitness/hearing: {len(training_df_valid)}")

    # Oversample high-quality dental and vision
    minority_personas = ['dental', 'vision']
    oversampled_dfs = [training_df_valid[~training_df_valid['persona'].isin(minority_personas)]]
    
    for persona in minority_personas:
        persona_df = training_df_valid[training_df_valid['persona'] == persona]
        high_quality_df = persona_df[persona_df['quality_level'] == 'High']
        if len(high_quality_df) > 0:
            oversampled_high = resample(
                high_quality_df, replace=True, n_samples=max(100, len(high_quality_df) * 2), random_state=42
            )
            oversampled_dfs.append(oversampled_high)
        oversampled_dfs.append(persona_df[persona_df['quality_level'] != 'High'])
    
    training_df_oversampled = pd.concat(oversampled_dfs, ignore_index=True)
    print(f"Rows after oversampling high-quality dental/vision: {len(training_df_oversampled)}")

    X = training_df_oversampled[feature_columns].fillna(0)
    y = training_df_oversampled['persona']

    try:
        training_df.to_csv(weighted_behavioral_output_file, index=False)
        print(f"Behavioral data with weights saved to: {weighted_behavioral_output_file}")
    except Exception as e:
        print(f"Error saving weighted CSV: {e}")
        raise

    return X, y

def train_model(X, y):
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced_subsample'
    )
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
