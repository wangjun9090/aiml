import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/weighted_us_dce_pro_behavioral_features_0301_0302_2025.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_with_weights.pkl'
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

def prepare_evaluation_features(behavioral_df, plan_df):
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    print(f"Rows after merge: {len(training_df)}")
    print("Columns after merge:", training_df.columns.tolist())

    # Resolve state column conflict
    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    # Define features (same as training)
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

    # CSNP-specific features (consistent with training)
    additional_features = []
    if 'csnp' in training_df.columns:
        training_df['csnp_interaction'] = training_df['csnp'].fillna(0) * (
            training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
            training_df['time_csnp_pages'].fillna(0) + training_df['accordion_csnp'].fillna(0)
        ) * 2
        additional_features.append('csnp_interaction')
    else:
        training_df['csnp_interaction'] = 0
    if 'csnp_type' in training_df.columns:
        training_df['csnp_type_flag'] = (training_df['csnp_type'] == 'Y').astype(int)
        additional_features.append('csnp_type_flag')
    else:
        training_df['csnp_type_flag'] = 0
    training_df['csnp_signal_strength'] = (
        training_df['query_csnp'].fillna(0) + training_df['filter_csnp'].fillna(0) + 
        training_df['accordion_csnp'].fillna(0) + training_df['time_csnp_pages'].fillna(0)
    ).clip(upper=5) * 1.5
    additional_features.append('csnp_signal_strength')

    # Weighted features (expected from training)
    weighted_features = [
        'w_doctor', 'w_drug', 'w_vision', 'w_dental', 'w_otc', 'w_transportation',
        'w_csnp', 'w_dsnp', 'w_fitness', 'w_hearing'
    ]

    # Final feature set (must match training)
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + weighted_features

    # Check for missing weighted features and warn
    missing_weights = [col for col in weighted_features if col not in training_df.columns]
    if missing_weights:
        print(f"Warning: The following weighted features are missing: {missing_weights}. Filling with 0.")
        for col in missing_weights:
            training_df[col] = 0

    # Select features and handle missing values
    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona'] if 'persona' in training_df.columns else None

    # Define quality levels (for evaluation output)
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
            return 'Medium'

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
    y_true_valid = y_true[valid_mask].reset_index(drop=True)
    metadata_valid = metadata[valid_mask].reset_index(drop=True)

    if len(y_true_valid) == 0:
        print("ERROR: No valid (non-blank) persona values remain for evaluation.")
        raise ValueError("Cannot evaluate predictions with no valid ground truth data.")

    # Predict probabilities and top predictions
    y_pred_proba = model.predict_proba(X_valid)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X_valid)
    
    # Combine predictions with metadata
    output_df = pd.concat([metadata_valid, proba_df], axis=1)
    output_df['predicted_persona'] = y_pred
    output_df['is_correct'] = (output_df['persona'] == output_df['predicted_persona'])

    # Add probability ranking
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])

    # Save to CSV (includes all rows)
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
        level_pred = y_pred[level_mask.values]
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
    print("Evaluating data with pre-trained Random Forest model...")

    # Load model and data
    rf_model = load_model(model_file)
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)

    # Prepare features (no weight recalculation)
    X, y_true, metadata = prepare_evaluation_features(behavioral_df, plan_df)

    # Evaluate predictions and save
    evaluate_predictions(rf_model, X, y_true, metadata)

if __name__ == "__main__":
    main()
