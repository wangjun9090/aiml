import pandas as pd
import pickle

# File paths (adjust as needed)
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_behavioral_features_0901_2024_0228_2025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_csnp_focus.pkl'

# Hardcoded userid
USERID = '16052055999993VDN490U04DLREL3E69SDG5HVPVUKQJC'

def load_model(model_path):
    """Load the trained Random Forest model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_data(behavioral_path, plan_path):
    """Load behavioral and plan data."""
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Plan data loaded: {len(plan_df)} rows")
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_features(behavioral_df, plan_df, userid):
    """Prepare features for a given userid matching the training dataset."""
    # Filter for the specific user
    user_data = behavioral_df[behavioral_df['userid'] == userid].copy()
    if user_data.empty:
        raise ValueError(f"No data found for userid: {userid}")

    # Merge with plan data
    user_data = user_data.merge(
        plan_df,
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )

    # Resolve state column conflict if columns exist
    if 'state_beh' in user_data.columns and 'state_plan' in user_data.columns:
        user_data['state'] = user_data['state_beh'].fillna(user_data['state_plan'])
        user_data = user_data.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    # Define feature columns (same as in training)
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

    # Add CSNP-specific features
    user_data['csnp_interaction'] = user_data['csnp'] * (user_data['query_csnp'] + user_data['filter_csnp'] + user_data['time_csnp_pages'])
    user_data['csnp_type_flag'] = (user_data['csnp_type'] == 'Y').astype(int)
    user_data['csnp_signal_strength'] = (user_data['query_csnp'] + user_data['filter_csnp'] + user_data['accordion_csnp'] + user_data['time_csnp_pages']).clip(upper=3)

    additional_features = ['csnp_interaction', 'csnp_type_flag', 'csnp_signal_strength']

    # Persona weights
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

    k1, k3, k4, k7, k8, k9, k10 = 0.1, 0.5, 0.4, 0.15, 0.25, 0.8, 0.7
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.0, 3.0, 1.0, 1.5

    def calculate_persona_weight(row, persona_info, persona):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        
        weight_cap = 0.7 if persona == 'csnp' else 0.5
        
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(row[plan_col], weight_cap)
            if persona == 'csnp' and row['csnp_type'] == 'Y':
                base_weight *= W_CSNP_HIGH
            elif persona == 'csnp':
                base_weight *= W_CSNP_BASE
            elif persona == 'dsnp' and row['dsnp_type'] == 'Y':
                base_weight *= W_DSNP_HIGH
            elif persona == 'dsnp':
                base_weight *= W_DSNP_BASE
                if row['csnp_signal_strength'] > 1:
                    base_weight *= 0.8
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
            if signal_count >= 2: behavioral_score += 0.45
            elif signal_count >= 1: behavioral_score += 0.25
        elif persona == 'csnp':
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 2: behavioral_score += 0.5
            elif signal_count >= 1: behavioral_score += 0.3
            if row['csnp_interaction'] > 0: behavioral_score += 0.2
            if row['csnp_type_flag'] == 1: behavioral_score += 0.15
            if row['csnp_signal_strength'] > 1: behavioral_score += 0.15
        elif persona in ['fitness', 'hearing']:
            signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
            if signal_count >= 1: behavioral_score += 0.3
        
        return min(base_weight + behavioral_score, 1.5 if persona == 'csnp' else 1.0)

    # Calculate weights for each persona
    for persona, info in persona_weights.items():
        user_data[f'w_{persona}'] = user_data.apply(lambda row: calculate_persona_weight(row, info, persona), axis=1)

    # Normalize weights for all except csnp
    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = user_data[weighted_features].sum(axis=1)
    for wf in weighted_features:
        user_data[wf] = user_data[wf] / weight_sum.where(weight_sum > 0, 1)

    # Final feature set
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    # Select features and handle missing values
    X = user_data[feature_columns].fillna(0)
    
    # Debugging: Print persona weights
    print("\nPersona Weights for User:")
    for persona in persona_weights.keys():
        print(f"{persona}: {user_data[f'w_{persona}'].iloc[0]:.4f}")
    
    return X

def predict_persona(model, X):
    """Predict the persona using the loaded model."""
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)[0]
    class_names = model.classes_
    print("\nPrediction Probabilities:")
    for cls, prob in zip(class_names, probabilities):
        print(f"{cls}: {prob:.4f}")
    return prediction[0]

def main():
    print(f"Predicting persona for hardcoded userid: {USERID}")

    # Load model and data
    rf_model = load_model(model_file)
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)

    # Prepare features
    try:
        X = prepare_features(behavioral_df, plan_df, USERID)
    except ValueError as e:
        print(e)
        return

    # Make prediction
    predicted_persona = predict_persona(rf_model, X)
    print(f"Predicted Persona: {predicted_persona}")

if __name__ == "__main__":
    main()
