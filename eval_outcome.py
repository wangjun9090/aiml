import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# File paths
OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_results_0401_2025_0420_2025.csv'
NEW_OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_outcomes_0401_2025_0420_2025.csv'

def load_output_file(file_path):
    """Load the evaluation results CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def compute_evaluation_outcomes(df):
    """Compute accuracy, match rate, and confidence results."""
    # Ensure required columns exist
    required_cols = ['persona', 'predicted_persona', 'confidence_score', 'probability_ranking']
    weight_cols = [col for col in df.columns if col.startswith('w_')]  # e.g., w_csnp, w_dental
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute per-record match (1 if predicted_persona == persona, 0 otherwise)
    df['match'] = (df['predicted_persona'] == df['persona']).astype(int)

    # Compute overall accuracy
    overall_accuracy = accuracy_score(df['persona'], df['predicted_persona'])
    df['overall_accuracy'] = overall_accuracy

    # Compute per-persona correction rates (match rates)
    personas = sorted(df['persona'].unique())
    persona_correction_rates = {}
    persona_counts = {}
    for persona in personas:
        mask = df['persona'] == persona
        count = mask.sum()
        persona_counts[persona] = count
        if count > 0:
            persona_accuracy = accuracy_score(df['persona'][mask], df['predicted_persona'][mask])
            persona_correction_rates[persona] = persona_accuracy
            print(f"Correction Rate for '{persona}': {persona_accuracy * 100:.2f}% (Matches: {df['match'][mask].sum()}/{count})")
        else:
            persona_correction_rates[persona] = 0.0
            print(f"Correction Rate for '{persona}': N/A (Count: 0)")

    # Compare with provided rates
    provided_rates = {
        'csnp': 0.8811, 'dental': 0.9091, 'doctor': 0.9583,
        'drug': 0.8830, 'dsnp': 1.0000, 'vision': 1.0000
    }
    print("\nProvided vs. Computed Correction Rates:")
    for persona in provided_rates:
        computed = persona_correction_rates.get(persona, 0.0)
        print(f"{persona}: Provided = {provided_rates[persona]:.4f}, Computed = {computed:.4f}")

    # Select columns for output
    output_cols = [
        'userid', 'zip', 'plan_id', 'persona', 'predicted_persona'
    ] + weight_cols + [
        'probability_ranking', 'confidence_score', 'match', 'overall_accuracy'
    ]
    available_cols = [col for col in output_cols if col in df.columns]
    output_df = df[available_cols].copy()

    # Add per-record persona match indicator (already in 'match')
    output_df['persona_match'] = output_df['match']  # Alias for clarity

    return output_df, persona_correction_rates, overall_accuracy

def save_evaluation_outcomes(output_df, file_path):
    """Save the evaluation outcomes to a new CSV."""
    try:
        output_df.to_csv(file_path, index=False)
        print(f"Saved evaluation outcomes to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        raise

def main():
    # Load the output file
    df = load_output_file(OUTPUT_FILE)

    # Compute evaluation outcomes
    output_df, persona_correction_rates, overall_accuracy = compute_evaluation_outcomes(df)

    # Print summary
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
    print("Sample of Evaluation Outcomes:")
    print(output_df[['persona', 'predicted_persona', 'match', 'confidence_score', 'probability_ranking']].head())

    # Save the new output
    save_evaluation_outcomes(output_df, NEW_OUTPUT_FILE)

if __name__ == "__main__":
    main()
