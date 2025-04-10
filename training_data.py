import pandas as pd
import numpy as np

# Step 1: Load the datasets
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_20250301_20250302.parquet'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'

print(f"Loading normalized behavioral features file: {behavioral_file}")
try:
    behavioral_df = pd.read_parquet(behavioral_file)
except Exception as e:
    print(f"Error loading behavioral file: {e}")
    raise

print(f"Loading plan derivation file: {plan_file}")
try:
    plan_df = pd.read_csv(plan_file)
except Exception as e:
    print(f"Error loading plan file: {e}")
    raise

print(f"Behavioral rows loaded: {len(behavioral_df)}")
print(f"Plan rows loaded: {len(plan_df)}")
print("Behavioral columns:", list(behavioral_df.columns))
print("Plan columns:", list(plan_df.columns))
print("Behavioral dtypes:\n", behavioral_df.dtypes)
print("Plan dtypes:\n", plan_df.dtypes)

# Step 2: Prepare plan_df for merging
plan_df = plan_df.rename(columns={'StateCode': 'state'})

# Ensure consistent data types for merge columns
behavioral_df['zip'] = behavioral_df['zip'].astype(str)
behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str)
plan_df['zip'] = plan_df['zip'].astype(str)
plan_df['plan_id'] = plan_df['plan_id'].astype(str)

# Step 3: Merge datasets
training_df = behavioral_df.merge(
    plan_df,
    how='left',
    on=['zip', 'plan_id'],
    suffixes=('_beh', '_plan')
)

print(f"Rows after merge: {len(training_df)}")
print("Sample after merge:")
print(training_df[['userid', 'zip', 'plan_id', 'compared_plan_ids', 'state_beh', 'state_plan', 'persona', 'ma_dental_benefit', 'ma_vision']].head())

# Step 4: Resolve state column conflict
training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
training_df = training_df.drop(columns=['state_beh', 'state_plan'])

# Step 5: Define the target variable
training_df['target_persona'] = training_df['persona']

# Step 6: Select all behavioral and plan features
all_behavioral_features = [
    'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
    'query_csnp', 'query_dsnp',
    'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug', 'filter_provider', 'filter_vision',
    'filter_csnp', 'filter_dsnp',
    'accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
    'accordion_vision', 'accordion_csnp', 'accordion_dsnp',
    'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
    'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
    'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
    'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages', 'rel_time_dsnp_pages',
    'total_session_time', 'num_pages_viewed', 'num_plans_selected', 'num_plans_compared',
    'submitted_application', 'dce_click_count', 'pro_click_count'
]

raw_plan_features = [
    'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
    'ma_provider_network', 'ma_drug_coverage'
]

# Add CSNP-specific features
training_df['csnp_interaction'] = training_df['csnp'] * (training_df['query_csnp'] + training_df['filter_csnp'] + training_df['time_csnp_pages'])
training_df['csnp_type_flag'] = (training_df['csnp_type'] == 'Y').astype(int)  # Binary indicator for csnp_type = 'Y'
training_df['csnp_signal_strength'] = (training_df['query_csnp'] + training_df['filter_csnp'] + training_df['accordion_csnp'] + training_df['time_csnp_pages']).clip(upper=3)

additional_features = ['csnp_interaction', 'csnp_type_flag', 'csnp_signal_strength']

# Step 7: Revised persona weight calculation with focus on csnp
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

k1 = 0.1   # Pages (capped at 3)
k3 = 0.5   # Query
k4 = 0.4   # Filter
k7 = 0.15  # Drug click count
k8 = 0.25  # Provider click count
k9 = 0.7   # CSNP-specific query coefficient (increased)
k10 = 0.6  # CSNP-specific filter coefficient (increased)

W_CSNP_BASE = 1.0
W_CSNP_HIGH = 2.5  # Increased from 2.0 for stronger boost
W_DSNP_BASE = 1.0
W_DSNP_HIGH = 1.5

def calculate_persona_weight(row, persona_info, persona):
    plan_col = persona_info['plan_col']
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    click_col = persona_info.get('click_col', None)
    
    weight_cap = 0.7 if persona == 'csnp' else 0.5  # Higher cap for csnp
    
    if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
        base_weight = min(row[plan_col], weight_cap)
        if persona == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y':
            base_weight *= W_CSNP_HIGH
        elif persona == 'csnp':
            base_weight *= W_CSNP_BASE
        elif persona == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y':
            base_weight *= W_DSNP_HIGH
        elif persona == 'dsnp':
            base_weight *= W_DSNP_BASE
    elif pd.isna(row['plan_id']) and pd.notna(row['compared_plan_ids']) and row['num_plans_compared'] > 0:
        compared_ids = row['compared_plan_ids'].split(',')
        compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
        if not compared_plans.empty:
            base_weight = min(compared_plans[plan_col].mean(), weight_cap)
            if persona == 'csnp':
                csnp_type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
            elif persona == 'dsnp':
                dsnp_type_y_ratio = (compared_plans['dsnp_type'] == 'Y').mean()
                base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)
        else:
            base_weight = 0
    else:
        base_weight = 0
    
    pages_viewed = min(row['num_pages_viewed'], 3) if pd.notna(row['num_pages_viewed']) else 0
    query_value = row[query_col] if pd.notna(row[query_col]) else 0
    filter_value = row[filter_col] if pd.notna(row[filter_col]) else 0
    click_value = row[click_col] if click_col and pd.notna(row[click_col]) else 0
    
    query_coeff = k9 if persona == 'csnp' else k3
    filter_coeff = k10 if persona == 'csnp' else k4
    click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0
    
    behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value
    
    if persona == 'doctor':
        if click_value >= 1.5:
            behavioral_score += 0.4
        elif click_value >= 0.5:
            behavioral_score += 0.2
    elif persona == 'drug':
        if click_value >= 5:
            behavioral_score += 0.4
        elif click_value >= 2:
            behavioral_score += 0.2
    elif persona == 'dental':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 2:
            behavioral_score += 0.3
        elif signal_count >= 1:
            behavioral_score += 0.15
    elif persona == 'vision':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 2:
            behavioral_score += 0.45
        elif signal_count >= 1:
            behavioral_score += 0.25
    elif persona == 'csnp':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 2:
            behavioral_score += 0.5  # Increased from 0.4
        elif signal_count >= 1:
            behavioral_score += 0.3  # Increased from 0.2
        # Add boosts from new features
        if 'csnp_interaction' in row and row['csnp_interaction'] > 0:
            behavioral_score += 0.3
        if 'csnp_type_flag' in row and row['csnp_type_flag'] == 1:
            behavioral_score += 0.2
        if 'csnp_signal_strength' in row and row['csnp_signal_strength'] > 1:
            behavioral_score += 0.2
    elif persona in ['fitness', 'hearing']:
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 1:
            behavioral_score += 0.3
    
    adjusted_weight = base_weight + behavioral_score
    
    if persona == row['target_persona']:
        non_target_weights = [
            min(row[info['plan_col']], (0.7 if p == 'csnp' else 0.5)) * (
                W_CSNP_HIGH if p == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y' else
                W_CSNP_BASE if p == 'csnp' else
                W_DSNP_HIGH if p == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y' else
                W_DSNP_BASE if p == 'dsnp' else
                1.0
            ) + (
                (k9 if p == 'csnp' else k3) * (row[info['query_col']] if pd.notna(row[info['query_col']]) else 0) +
                (k10 if p == 'csnp' else k4) * (row[info['filter_col']] if pd.notna(row[info['filter_col']]) else 0) +
                k1 * pages_viewed +
                (k8 if p == 'doctor' else k7 if p == 'drug' else 0) * 
                (row[info.get('click_col')] if 'click_col' in info and pd.notna(row[info.get('click_col')]) else 0) +
                (0.4 if p == 'doctor' and row[info.get('click_col', 'pro_click_count')] >= 1.5 else 
                 0.2 if p == 'doctor' and row[info.get('click_col', 'pro_click_count')] >= 0.5 else 
                 0.4 if p == 'drug' and row[info.get('click_col', 'dce_click_count')] >= 5 else 
                 0.2 if p == 'drug' and row[info.get('click_col', 'dce_click_count')] >= 2 else 
                 0.3 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 2 else 
                 0.15 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                 0.45 if p == 'vision' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 2 else 
                 0.25 if p == 'vision' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                 0.5 if p == 'csnp' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 2 else 
                 0.3 if p == 'csnp' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                 0.3 if p == 'csnp' and 'csnp_interaction' in row and row['csnp_interaction'] > 0 else 
                 0.2 if p == 'csnp' and 'csnp_type_flag' in row and row['csnp_type_flag'] == 1 else 
                 0.2 if p == 'csnp' and 'csnp_signal_strength' in row and row['csnp_signal_strength'] > 1 else 
                 0.3 if p in ['fitness', 'hearing'] and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 0)
            )
            for p, info in persona_weights.items()
            if p != row['target_persona'] and pd.notna(row[info['plan_col']])
        ]
        max_non_target = max(non_target_weights, default=0)
        adjusted_weight = max(adjusted_weight, max_non_target + (0.25 if persona == 'csnp' else 0.15))
    
    return min(adjusted_weight, 1.5 if persona == 'csnp' else 1.0)  # Higher cap for csnp

# Calculate weights
for persona, info in persona_weights.items():
    training_df[f'w_{persona}'] = training_df.apply(
        lambda row: calculate_persona_weight(row, info, persona), axis=1
    )

# Normalize weights for all except csnp
weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
weight_sum = training_df[weighted_features].sum(axis=1)
for wf in weighted_features:
    training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)

# Combine all features
feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]

# Step 8: Create the final training dataset
final_columns = ['userid', 'zip', 'plan_id', 'dsnp_type', 'csnp_type', 'state'] + feature_columns + ['target_persona']
final_training_df = training_df[final_columns].copy()

# Handle missing values in features
final_training_df[feature_columns] = final_training_df[feature_columns].fillna(0)

# Deduplicate state per userid
def deduplicate_state(group):
    state_values = group.dropna()
    return state_values.iloc[0] if not state_values.empty else np.nan

final_training_df['state'] = final_training_df.groupby('userid')['state'].transform(deduplicate_state)

# Debug: Check final dataset
print("Missing values in final training dataset:")
print(final_training_df.isnull().sum())
print("Final training dataset columns:", list(final_training_df.columns))
print("Sample of final training dataset:")
print(final_training_df.head())
