import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hardcoded file paths for Databricks (using /dbfs/ prefix)
BEHAVIORAL_FILE = '/dbfs/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/dbfs/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/dbfs/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_with_weights_092024_032025_v6.pkl'
OUTPUT_FILE = '/dbfs/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/032025/eval_results_092024_032025_v6.csv'

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
        # Load data using Pandas
        # Note: For larger datasets, consider using Spark (spark.read.csv) as in score.py
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Plan data loaded: {len(plan_df)} rows")
        print(f"Plan_df columns: {plan_df.columns.tolist()}")
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    """Normalize personas by splitting rows with multiple personas (e.g., 'csnp,dsnp') into separate rows."""
    new_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row['persona']):
            new_rows.append(row)
            continue
        personas = [p.strip().lower() for p in str(row['persona']).split(',')]
        if 'dsnp' in personas or 'csnp' in personas:
            first_row = row.copy()
            first_persona = personas[0]
            if first_persona in ['unknown', 'none', 'healthcare', '']:
                first_persona = 'dsnp' if 'dsnp' in personas else 'csnp'
            first_row['persona'] = first_persona
            new_rows.append(first_row)
            second_row = row.copy()
            second_row['persona'] = 'dsnp' if 'dsnp' in personas else 'csnp'
            new_rows.append(second_row)
        else:
            row_copy = row.copy()
            first_persona = personas[0]
            row_copy['persona'] = first_persona
            new_rows.append(row_copy)
    return pd.DataFrame(new_rows)

def prepare_evaluation_features(behavioral_df, plan_df):
    behavioral_df = normalize_persona(behavioral_df)
    print(f"Rows after persona normalization: {len(behavioral_df)}")

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

    for col in raw_plan_features + ['csnp_type']:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]

    def assign_quality_level(row):
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

    training_df['quality_level'] = training_df.apply(assign_quality_level, axis=1)

    additional_features = []
    training_df['csnp_interaction'] = training_df['csnp'] * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 2.5
    additional_features.append('csnp_interaction')

    training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 2.5
    additional_features.append('csnp_signal_strength')

    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * training_df['ma_dental_benefit'] * 1.5
    additional_features.append('dental_interaction')

    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * training_df['ma_vision'] * 1.5
    additional_features.append('vision_interaction')

    training_df['csnp_drug_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.0 - training_df['ma_drug_coverage'] * (
            training_df.get('query_drug', 0).fillna(0) + training_df.get('filter_drug', 0).fillna(0) + 
            training_df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 2.5
    additional_features.append('csnp_drug_interaction')

    training_df['csnp_doctor_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0)
        ) * 1.5 - training_df['ma_provider_network'] * (
            training_df.get('query_provider', 0).fillna(0) + training_df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 1.5
    additional_features.append('csnp_doctor_interaction')

    high_quality_csnp = training_df[(training_df['quality_level'] == 'High') & (training_df['persona'] == 'csnp')]
    print(f"Eval: High-quality csnp samples: {len(high_quality_csnp)}")
    print(f"Eval: Non-zero csnp_interaction: {sum(high_quality_csnp['csnp_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_drug_interaction: {sum(high_quality_csnp['csnp_drug_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_doctor_interaction: {sum(high_quality_csnp['csnp_doctor_interaction'] > 0)}")

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
        
        return min(adjusted_weight, 3.5 if persona == 'csnp' else 1.2)

    # For scalability: Consider parallelizing this operation for large datasets
    # Example: Use pandaparallel for parallel_apply
    # from pandaparallel import pandaparallel
    # pandaparallel.initialize()
    # training_df[f'w_{persona}'] = training_df.parallel_apply(lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1)

    print("Calculating persona weights...")
    for persona, info in persona_weights.items():
        training_df[f'w_{persona}'] = training_df.apply(
            lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1
        )

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys()]
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)

    all_weighted_features = [f'w_{persona}' for persona in [
        'doctor', 'drug', 'vision', 'dental', 'otc', 'transportation', 'csnp', 'dsnp'
    ]]

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + all_weighted_features

    print(f"Feature columns expected: {feature_columns}")

    missing_features = [col for col in feature_columns if col not in training_df.columns]
    if missing_features:
        print(f"Warning: Missing features in training_df: {missing_features}")
        for col in missing_features:
            training_df[col] = 0

    print(f"Columns in training_df after filling: {training_df.columns.tolist()}")

    valid_mask = (
        training_df['persona'].notna() & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    training_df = training_df[valid_mask]
    print(f"Rows after filtering fitness/hearing: {len(training_df)}")

    metadata = training_df[['userid', 'zip', 'plan_id', 'persona', 'quality_level'] + feature_columns]

    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona'] if 'persona' in training_df.columns else None

    if y_true is not None:
        print(f"Unique personas before filtering: {y_true.unique().tolist()}")

    return X, y_true, metadata

def evaluate_model(model, X, y_true, metadata):
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X)

    # Combine metadata with predictions
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred

    # Add probability ranking
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])

    # Compute overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Per-persona accuracy
    print("\nIndividual Persona Accuracy Rates (Overall):")
    for persona in personas:
        mask = y_true == persona
        if mask.sum() > 0:
            persona_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            print(f"Accuracy for '{persona}': {persona_accuracy * 100:.2f}% (Count: {mask.sum()})")

    # Breakdown by quality level
    for quality in ['High', 'Medium', 'Low']:
        mask = output_df['quality_level'] == quality
        if mask.sum() > 0:
            y_true_quality = y_true[mask]
            y_pred_quality = y_pred[mask]
            accuracy = accuracy_score(y_true_quality, y_pred_quality)
            print(f"\n{quality} Quality Data Results:")
            print(f"Rows: {mask.sum()}")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Individual Persona Accuracy Rates ({quality}):")
            for persona in personas:
                persona_mask = y_true_quality == persona
                if persona_mask.sum() > 0:
                    persona_accuracy = accuracy_score(y_true_quality[persona_mask], y_pred_quality[persona_mask])
                    print(f"Accuracy for '{persona}': {persona_accuracy * 100:.2f}% (Count: {persona_mask.sum()})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=personas)
    cm_df = pd.DataFrame(cm, index=personas, columns=personas)
    print("\nConfusion Matrix (Overall, valid personas only):")
    print(cm_df)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=personas))

    # Save evaluation results to DBFS
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nEvaluation results saved to {OUTPUT_FILE}")

def main():
    print("Evaluating Random Forest model...")
    model = load_model(MODEL_FILE)
    behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    X, y_true, metadata = prepare_evaluation_features(behavioral_df, plan_df)
    if y_true is not None:
        evaluate_model(model, X, y_true, metadata)
    else:
        print("No ground truth labels available for evaluation.")

if __name__ == "__main__":
    main()
