import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0301_0302_2025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona.pkl'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/032025/eval_results_0301_0302_with_quality_levels_tweaked.csv'

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

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

    # Resolve state column conflict
    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    # Define features
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
    training_df['csnp_interaction'] = training_df['csnp'].fillna(0) * (
        training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
        training_df['time_csnp_pages'].fillna(0) + training_df['accordion_csnp'].fillna(0)
    ) * 2
    training_df['csnp_type_flag'] = (training_df['csnp_type'] == 'Y').astype(int)
    training_df['csnp_signal_strength'] = (
        training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
        training_df['accordion_csnp'].fillna(0) + training_df['time_csnp_pages'].fillna(0)
    ).clip(upper=5) * 1.5

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

    k1, k3, k4, k7, k8 = 0.1, 0.5, 0.4, 0.15, 0.25
    k9, k10 = 1.0, 0.9
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.0, 3.0, 1.0, 1.5

    def calculate_persona_weight(row, persona_info, persona):
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
            if not compared_plans.empty:
                base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
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
                     0.3 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 2 else 
                     0.15 if p == 'dental' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                     0.35 if p == 'vision' and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 
                     0.3 if p in ['csnp', 'fitness', 'hearing'] and sum([1 for val in [row[info['query_col']], row[info['filter_col']], pages_viewed] if val > 0]) >= 1 else 0)
                )
                for p, info in persona_weights.items()
                if p != row['persona'] and pd.notna(row[info['plan_col']])
            ]
            max_non_target = max(non_target_weights, default=0)
            adjusted_weight = max(adjusted_weight, max_non_target + 0.15)
        
        return min(adjusted_weight, 2.0 if persona == 'csnp' else 1.0)

    # Calculate weights
    for persona, info in persona_weights.items():
        training_df[f'w_{persona}'] = training_df.apply(lambda row: calculate_persona_weight(row, info, persona), axis=1)

    # Normalize weights, excluding csnp
    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)

    # Final feature set
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    # Select features and handle missing values
    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona'] if 'persona' in training_df.columns else None

    # Define quality levels
    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]

    def assign_quality_level(row):
        has_plan_id = pd.notna(row['plan_id'])
        has_clicks = (row['dce_click_count'] > 0 and pd.notna(row['dce_click_count'])) or \
                     (row['pro_click_count'] > 0 and pd.notna(row['pro_click_count']))
        has_filters = any(row[col] > 0 and pd.notna(row[col]) for col in filter_cols)
        has_queries = any(row[col] > 0 and pd.notna(row[col]) for col in query_cols)
        
        if has_plan_id and (has_clicks or has_filters):
            return 'High'
        elif has_plan_id and not has_clicks and not has_filters and has_queries:
            return 'Medium'
        elif not has_plan_id and not has_clicks and not has_filters and not has_queries:
            return 'Low'
        else:
            return 'Medium'  # Remaining cases (e.g., no plan_id with signals) go to Medium

    training_df['quality_level'] = training_df.apply(assign_quality_level, axis=1)

    # Metadata for output
    metadata = training_df[['userid', 'zip', 'plan_id', 'persona', 'quality_level'] + feature_columns]

    return X, y_true, metadata

def evaluate_predictions(model, X, y_true, metadata):
    if y_true is None:
        print("No ground truth 'persona' column found in the data. Predictions will be made without accuracy evaluation.")
        y_pred_proba = model.predict_proba(X)
        personas = model.classes_
        proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
        y_pred = model.predict(X)
        
        output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
        output_df['predicted_persona'] = y_pred
        
        for i in range(len(output_df)):
            probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
            ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])
        
        output_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return

    # Filter out blank personas
    valid_mask = y_true.notna() & (y_true != '')
    if not valid_mask.all():
        excluded_count = (~valid_mask).sum()
        print(f"Warning: Excluding {excluded_count} rows with blank or NaN 'persona' values from evaluation.")
        print(f"Sample of excluded personas: {y_true[~valid_mask].head().tolist()}")
    
    X_valid = X[valid_mask]
    y_true_valid = y_true[valid_mask]
    metadata_valid = metadata[valid_mask]

    if len(y_true_valid) == 0:
        print("ERROR: No valid (non-blank) persona values remain for evaluation.")
        raise ValueError("Cannot evaluate predictions with no valid ground truth data.")

    # Predict probabilities and top predictions
    y_pred_proba = model.predict_proba(X_valid)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X_valid)
    
    # Combine predictions with metadata
    output_df = pd.concat([metadata_valid.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred
    output_df['is_correct'] = (output_df['persona'] == output_df['predicted_persona'])

    # Add probability ranking
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])

    # Save to CSV (includes all rows, even those not evaluated)
    full_output_df = metadata.copy()
    full_output_df['predicted_persona'] = np.nan
    full_output_df.loc[valid_mask, 'predicted_persona'] = y_pred
    full_output_df['is_correct'] = np.nan
    full_output_df.loc[valid_mask, 'is_correct'] = output_df['is_correct']
    full_output_df = pd.concat([full_output_df, pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas], index=metadata.index[valid_mask])], axis=1)
    full_output_df['probability_ranking'] = np.nan
    full_output_df.loc[valid_mask, 'probability_ranking'] = output_df['probability_ranking']
    full_output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Evaluate overall accuracy
    accuracy = accuracy_score(y_true_valid, y_pred)
    print(f"\nOverall Accuracy (non-blank personas): {accuracy * 100:.2f}%")
    print(f"Rows evaluated: {len(y_true_valid)}")
    print(f"Correct predictions: {sum(y_pred == y_true_valid)}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_valid, y_pred))
    
    print("\nIndividual Persona Accuracy Rates (Overall):")
    for persona in set(y_true_valid):
        persona_true = y_true_valid == persona
        persona_count = sum(persona_true)
        if persona_count > 0:
            persona_accuracy = accuracy_score(y_true_valid[persona_true], y_pred[persona_true]) * 100
            print(f"Accuracy for '{persona}': {persona_accuracy:.2f}% (Count: {persona_count})")

    # Analyze by quality level
    quality_levels = ['High', 'Medium', 'Low']
    for level in quality_levels:
        level_mask = output_df['quality_level'] == level
        level_true = y_true_valid[level_mask]
        level_pred = y_pred[level_mask]
        print(f"\n{level} Quality Data Results:")
        print(f"Rows: {len(level_true)}")
        if len(level_true) > 0:
            level_accuracy = accuracy_score(level_true, level_pred) * 100
            print(f"Accuracy: {level_accuracy:.2f}%")
            print(f"Individual Persona Accuracy Rates ({level}):")
            for persona in set(level_true):
                persona_true = level_true == persona
                persona_count = sum(persona_true)
                if persona_count > 0:
                    persona_accuracy = accuracy_score(level_true[persona_true], level_pred[persona_true]) * 100
                    print(f"Accuracy for '{persona}': {persona_accuracy:.2f}% (Count: {persona_count})")
        else:
            print("No records in this quality level.")

    print("\nConfusion Matrix (Overall, non-blank personas):")
    cm = confusion_matrix(y_true_valid, y_pred, labels=list(set(y_true_valid)))
    print(pd.DataFrame(cm, index=list(set(y_true_valid)), columns=list(set(y_true_valid))))

def main():
    print("Testing all data with tweaked quality level categorization...")

    # Load model and data
    rf_model = load_model(model_file)
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)

    # Prepare features
    X, y_true, metadata = prepare_training_features(behavioral_df, plan_df)

    # Evaluate predictions and save
    evaluate_predictions(rf_model, X, y_true, metadata)

if __name__ == "__main__":
    main()
