import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_behavioral_features_0901_2024_0228_2025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_csnp_focus.pkl'

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

def prepare_features(behavioral_df, plan_df, disable_component=None, persona_to_disable=None):
    """Prepare features with optional disabling of weight components."""
    # Merge with plan data
    data = behavioral_df.merge(
        plan_df,
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )

    # Resolve state column conflict if columns exist
    if 'state_beh' in data.columns and 'state_plan' in data.columns:
        data['state'] = data['state_beh'].fillna(data['state_plan'])
        data = data.drop(columns=['state_beh', 'state_plan'], errors='ignore')

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
    data['csnp_interaction'] = data['csnp'] * (data['query_csnp'] + data['filter_csnp'] + data['time_csnp_pages'])
    data['csnp_type_flag'] = (data['csnp_type'] == 'Y').astype(int)
    data['csnp_signal_strength'] = (data['query_csnp'] + data['filter_csnp'] + data['accordion_csnp'] + data['time_csnp_pages']).clip(upper=3)

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

    def calculate_persona_weight(row, persona_info, persona, disable_component=None):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        
        weight_cap = 0.7 if persona == 'csnp' else 0.5
        
        # Base weight from plan data
        if disable_component == 'plan_col' and persona == persona_to_disable:
            base_weight = 0
        elif pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
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
        
        # Behavioral components
        pages_viewed = min(row['num_pages_viewed'], 3) if pd.notna(row['num_pages_viewed']) else 0
        query_value = 0 if disable_component == 'query_col' and persona == persona_to_disable else row[query_col] if pd.notna(row[query_col]) else 0
        filter_value = 0 if disable_component == 'filter_col' and persona == persona_to_disable else row[filter_col] if pd.notna(row[filter_col]) else 0
        click_value = 0 if disable_component == 'click_col' and persona == persona_to_disable and click_col else row[click_col] if click_col and pd.notna(row[click_col]) else 0
        
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

    # Calculate weights for each persona with optional disabling
    for persona, info in persona_weights.items():
        data[f'w_{persona}'] = data.apply(
            lambda row: calculate_persona_weight(row, info, persona, disable_component=disable_component if persona_to_disable == persona else None),
            axis=1
        )

    # Normalize weights for all except csnp
    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = data[weighted_features].sum(axis=1)
    for wf in weighted_features:
        data[wf] = data[wf] / weight_sum.replace(0, 1)

    # Final feature set
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    # Select features and handle missing values
    X = data[feature_columns].fillna(0)
    
    # Get actual personas for comparison
    y_true = data['persona'] if 'persona' in data.columns else None
    
    return X, y_true

def evaluate_predictions(model, X, y_true, test_name):
    """Evaluate predictions and report accuracy."""
    if y_true is None:
        print(f"No ground truth 'persona' column found for {test_name}. Skipping accuracy computation.")
        return None, None, None
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\n--- {test_name} ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Rows evaluated: {len(y_true)}")
    print(f"Correct predictions: {sum(y_pred == y_true)}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy, report, y_pred

def analyze_weight_components(model, X_baseline, y_true, persona_weights):
    """Analyze the impact of each weight component by persona."""
    # Baseline evaluation (full weights)
    baseline_accuracy, baseline_report, _ = evaluate_predictions(model, X_baseline, y_true, "Baseline (Full Weights)")
    
    if baseline_accuracy is None:
        return

    # Components to test
    components = ['plan_col', 'query_col', 'filter_col', 'click_col']
    
    # Store results
    results = {persona: {} for persona in persona_weights.keys()}
    
    # Evaluate each persona with each component disabled
    for persona in persona_weights.keys():
        print(f"\nAnalyzing Weight Components for Persona: {persona}")
        for component in components:
            # Skip click_col for personas without it
            if component == 'click_col' and 'click_col' not in persona_weights[persona]:
                continue
            
            X, y_true = prepare_features(behavioral_df, plan_df, disable_component=component, persona_to_disable=persona)
            accuracy, report, _ = evaluate_predictions(model, X, y_true, f"Disabled {component} for {persona}")
            
            if accuracy is not None:
                accuracy_drop = baseline_accuracy - accuracy
                persona_precision_drop = baseline_report[persona]['precision'] - report[persona]['precision'] if persona in report else 0
                persona_recall_drop = baseline_report[persona]['recall'] - report[persona]['recall'] if persona in report else 0
                results[persona][component] = {
                    'accuracy_drop': accuracy_drop * 100,
                    'precision_drop': persona_precision_drop,
                    'recall_drop': persona_recall_drop
                }
    
    # Summarize key factors
    print("\n--- Summary of Key Weight Components ---")
    for persona, comp_results in results.items():
        print(f"\nPersona: {persona}")
        for comp, metrics in comp_results.items():
            print(f"  {comp}:")
            print(f"    Overall Accuracy Drop: {metrics['accuracy_drop']:.2f}%")
            print(f"    Precision Drop: {metrics['precision_drop']:.2f}")
            print(f"    Recall Drop: {metrics['recall_drop']:.2f}")
        # Identify most impactful component
        if comp_results:
            key_component = max(comp_results, key=lambda c: comp_results[c]['accuracy_drop'])
            print(f"  Most Impactful Component: {key_component} (Accuracy Drop: {comp_results[key_component]['accuracy_drop']:.2f}%)")

def main():
    global behavioral_df, plan_df  # Make these available globally for prepare_features
    print("Analyzing weight components impact on accuracy...")

    # Load model and data
    rf_model = load_model(model_file)
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)

    # Prepare baseline features
    X_baseline, y_true = prepare_features(behavioral_df, plan_df)

    # Analyze weight components
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
    analyze_weight_components(rf_model, X_baseline, y_true, persona_weights)

if __name__ == "__main__":
    main()
