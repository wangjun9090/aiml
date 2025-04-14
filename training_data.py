import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler  # Added for normalization

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_092024_032025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_092024_032025_v7.pkl'
weighted_behavioral_output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/processed_us_dce_pro_behavioral_features_092024_032025_v7.csv'

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Plan data loaded: {len(plan_df)} rows")
        print(f"Plan_df columns: {plan_df.columns.tolist()}")
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

    # Ensure all raw_plan_features and csnp_type exist
    for col in raw_plan_features + ['csnp_type']:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    # Compute quality level
    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]
    training_df['quality_level'] = training_df.apply(
        lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
    )

    # Normalize continuous features (NEW)
    continuous_features = [
        'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
        'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
        'total_session_time', 'num_pages_viewed', 'num_plans_selected', 'num_plans_compared',
        'dce_click_count', 'pro_click_count'
    ]
    scaler = StandardScaler()
    training_df[continuous_features] = scaler.fit_transform(training_df[continuous_features].fillna(0))

    additional_features = []
    training_df['csnp_interaction'] = training_df['csnp'] * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 3.0
    additional_features.append('csnp_interaction')

    training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 3.0
    additional_features.append('csnp_signal_strength')

    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * training_df['ma_dental_benefit'] * 2.0
    additional_features.append('dental_interaction')

    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * training_df['ma_vision'] * 2.0
    additional_features.append('vision_interaction')

    training_df['csnp_drug_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.5 - training_df['ma_drug_coverage'] * (
            training_df.get('query_drug', 0).fillna(0) + training_df.get('filter_drug', 0).fillna(0) + 
            training_df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 3.0
    additional_features.append('csnp_drug_interaction')

    training_df['csnp_doctor_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0)
        ) * 2.0 - training_df['ma_provider_network'] * (
            training_df.get('query_provider', 0).fillna(0) + training_df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 2.0
    additional_features.append('csnp_doctor_interaction')

    # New interaction features for dsnp and vision (NEW)
    training_df['dsnp_interaction'] = training_df['dsnp'] * (
        training_df.get('query_dsnp', 0).fillna(0) + training_df.get('filter_dsnp', 0).fillna(0) + 
        training_df.get('time_dsnp_pages', 0).fillna(0) + training_df.get('accordion_dsnp', 0).fillna(0)
    ) * 3.0
    additional_features.append('dsnp_interaction')

    training_df['vision_signal_strength'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0) + 
        training_df.get('accordion_vision', 0).fillna(0) + training_df.get('time_vision_pages', 0).fillna(0)
    ).clip(upper=5) * 2.0
    additional_features.append('vision_signal_strength')

    # Debug: Check feature distributions
    high_quality_csnp = training_df[(training_df['quality_level'] == 'High') & (training_df['persona'] == 'csnp')]
    print(f"High-quality csnp samples: {len(high_quality_csnp)}")
    print(f"Non-zero csnp_interaction: {sum(high_quality_csnp['csnp_interaction'] > 0)}")
    print(f"Non-zero csnp_drug_interaction: {sum(high_quality_csnp['csnp_drug_interaction'] > 0)}")
    print(f"Non-zero csnp_doctor_interaction: {sum(high_quality_csnp['csnp_doctor_interaction'] > 0)}")

    feature_columns = all_behavioral_features + raw_plan_features + additional_features

    # Filter invalid personas and exclude fitness, hearing
    valid_mask = (
        training_df['persona'].notna() & 
        (training_df['persona'] != '') & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    training_df_valid = training_df[valid_mask]
    print(f"Rows after filtering invalid personas and fitness/hearing: {len(training_df_valid)}")

    # Downsample low-quality data
    low_quality_df = training_df_valid[training_df_valid['quality_level'] == 'Low']
    if len(low_quality_df) > 800:
        low_quality_df = resample(
            low_quality_df, replace=False, n_samples=800, random_state=42
        )
        print(f"Downsampled low-quality data to: {len(low_quality_df)}")

    # Enhanced oversampling for minority personas (MODIFIED)
    minority_personas = ['csnp', 'dental', 'vision', 'dsnp']
    oversampled_dfs = [training_df_valid[~training_df_valid['persona'].isin(minority_personas)]]
    
    for persona in minority_personas:
        persona_df = training_df_valid[training_df_valid['persona'] == persona]
        high_quality_df = persona_df[persona_df['quality_level'] == 'High']
        if len(high_quality_df) > 0:
            # Increased oversampling ratios
            n_samples = max(500, len(high_quality_df) * 4) if persona == 'csnp' else \
                        max(300, len(high_quality_df) * 3) if persona == 'dental' else \
                        max(300, len(high_quality_df) * 3) if persona == 'vision' else \
                        max(300, len(high_quality_df) * 3)  # dsnp
            oversampled_high = resample(
                high_quality_df, replace=True, n_samples=n_samples, random_state=42
            )
            oversampled_dfs.append(oversampled_high)
        oversampled_dfs.append(persona_df[persona_df['quality_level'] != 'High'])
    
    training_df_oversampled = pd.concat(oversampled_dfs + [low_quality_df], ignore_index=True)
    print(f"Rows after oversampling high-quality csnp/dental/vision/dsnp and downsampling low-quality: {len(training_df_oversampled)}")

    X = training_df_oversampled[feature_columns].fillna(0)
    y = training_df_oversampled['persona']

    # Enhanced sample weights (MODIFIED)
    sample_weights = training_df_oversampled['quality_level'].map({
        'High': 3.0,  # Increased to emphasize high-quality data
        'Medium': 1.5,
        'Low': 0.3   # Reduced to de-emphasize low-quality data
    })

    try:
        training_df.to_csv(weighted_behavioral_output_file, index=False)
        print(f"Behavioral data saved to: {weighted_behavioral_output_file}")
        print(f"Saved columns: {training_df.columns.tolist()}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise

    return X, y, sample_weights

def train_model(X, y, sample_weights):
    class_weights = {
        'csnp': 2.0,      # Reduced slightly to balance
        'dental': 2.5,    # Increased
        'vision': 2.5,    # Increased
        'doctor': 1.5,    # Increased
        'drug': 1.5,      # Increased
        'dsnp': 2.5,      # Increased
        'otc': 1.0,
        'transportation': 1.0
    }
    rf_model = RandomForestClassifier(
        n_estimators=200,        # Increased for better robustness
        max_depth=20,           # Limit depth to prevent overfitting
        min_samples_split=5,    # Increase to reduce noise sensitivity
        min_samples_leaf=2,     # Increase for smoother splits
        max_features='sqrt',    # Reduce features per split
        random_state=42,
        class_weight=class_weights
    )
    rf_model.fit(X, y, sample_weight=sample_weights * 1.2)  # Scale sample weights slightly
    print("Random Forest model trained with tuned parameters.")
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
    print("Training Random Forest model...")
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)
    X, y, sample_weights = prepare_training_features(behavioral_df, plan_df)
    rf_model = train_model(X, y, sample_weights)
    save_model(rf_model, model_output_file)

if __name__ == "__main__":
    main()
