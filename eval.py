import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score

# Hardcoded file paths (unchanged)
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-0.0.2.pkl'
OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_results_0401_2025_0420_2025.csv'

# Other functions (load_model, load_data, normalize_persona, prepare_evaluation_features, main) remain unchanged
# ...

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

    # Compute overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Compute per-persona match rates
    print("\nPer-Persona Match Rates:")
    persona_match_rates = {}
    for persona in personas:
        mask = y_true == persona
        if mask.sum() > 0:
            persona_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            persona_match_rates[persona] = persona_accuracy
            print(f"Match Rate for '{persona}': {persona_accuracy * 100:.2f}% (Count: {mask.sum()})")
        else:
            persona_match_rates[persona] = 0.0
            print(f"Match Rate for '{persona}': N/A (Count: 0)")

    # Compute match rates by quality level
    print("\nMatch Rates by Quality Level:")
    quality_match_rates = {}
    for quality in ['High', 'Medium', 'Low']:
        mask = output_df['quality_level'] == quality
        if mask.sum() > 0:
            y_true_quality = y_true[mask]
            y_pred_quality = y_pred[mask]
            quality_accuracy = accuracy_score(y_true_quality, y_pred_quality)
            quality_match_rates[quality] = quality_accuracy
            print(f"{quality} Quality Match Rate: {quality_accuracy * 100:.2f}% (Count: {mask.sum()})")
            
            # Per-persona match rates for this quality level
            print(f"Individual Persona Match Rates ({quality}):")
            for persona in personas:
                persona_mask = y_true_quality == persona
                if persona_mask.sum() > 0:
                    persona_accuracy = accuracy_score(y_true_quality[persona_mask], y_pred_quality[persona_mask])
                    print(f"Match Rate for '{persona}': {persona_accuracy * 100:.2f}% (Count: {persona_mask.sum()})")
        else:
            quality_match_rates[quality] = 0.0
            print(f"{quality} Quality Match Rate: N/A (Count: 0)")

    # Compute top-2 accuracy for additional context
    top_2_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=2, labels=personas)
    print(f"\nTop-2 Match Rate: {top_2_accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=personas)
    cm_df = pd.DataFrame(cm, index=personas, columns=personas)
    print("\nConfusion Matrix (Overall, valid personas only):")
    print(cm_df)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=personas, target_names=personas))

    # Add metrics to output dataframe
    output_df['overall_accuracy'] = overall_accuracy  # Single value for all rows
    output_df['persona_match_rates'] = '; '.join([f"{p}: {persona_match_rates[p]:.4f}" for p in personas])  # Formatted string
    for quality in ['High', 'Medium', 'Low']:
        output_df[f'{quality.lower()}_quality_match_rate'] = quality_match_rates.get(quality, 0.0)  # Single value per quality level
    output_df['top_2_accuracy'] = top_2_accuracy  # Single value for all rows

    # Save evaluation results to DBFS
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nEvaluation results saved to {OUTPUT_FILE}")

    return output_df

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
