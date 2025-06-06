import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# File paths
OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_results_0401_2025_0420_2025.csv'
NEW_OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_outcomes_0401_2025_0420_2025.csv'
SUMMARY_OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_summary_outcomes_0401_2025_0420_2025.csv'

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
    """Compute per-record accuracy_rate (binary), summary accuracy, and confidence results."""
    # Ensure required columns exist
    required_cols = ['persona', 'predicted_persona', 'confidence_score', 'probability_ranking']
    weight_cols = [col for col in df.columns if col.startswith('w_')]  # e.g., w_csnp, w_dental
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute per-record accuracy_rate (1 if predicted_persona == persona, 0 otherwise)
    df['accuracy_rate'] = (df['predicted_persona'] == df['persona']).astype(int)

    # Compute overall accuracy
    overall_accuracy = accuracy_score(df['persona'], df['predicted_persona'])
    df['overall_accuracy'] = overall_accuracy

    # Compute per-persona metrics
    personas = sorted(df['persona'].unique())
    persona_metrics = []
    for persona in personas:
        mask = df['persona'] == persona
        count = mask.sum()
        matches = df['accuracy_rate'][mask].sum()
        accuracy = accuracy_score(df['persona'][mask], df['predicted_persona'][mask]) if count > 0 else 0.0
        avg_confidence = df['confidence_score'][mask].mean() if count > 0 else 0.0
        persona_metrics.append({
            'persona': persona,
            'total_records': count,
            'matches': matches,
            'accuracy_rate': round(accuracy, 2),  # 2 decimal places
            'avg_confidence': round(avg_confidence, 2)  # 2 decimal places
        })
        print(f"Persona '{persona}':")
        print(f"  Total Records: {count}")
        print(f"  Matches: {matches}")
        print(f"  Accuracy Rate: {accuracy * 100:.2f}%")
        print(f"  Average Confidence: {avg_confidence:.2f}")

    # Overall metrics
    overall_metrics = {
        'persona': 'Overall',
        'total_records': len(df),
        'matches': df['accuracy_rate'].sum(),
        'accuracy_rate': round(overall_accuracy, 2),  # 2 decimal places
        'avg_confidence': round(df['confidence_score'].mean(), 2)  # 2 decimal places
    }
    print("\nOverall Metrics:")
    print(f"  Total Records: {overall_metrics['total_records']}")
    print(f"  Matches: {overall_metrics['matches']}")
    print(f"  Accuracy Rate: {overall_metrics['accuracy_rate'] * 100:.2f}%")
    print(f"  Average Confidence: {overall_metrics['avg_confidence']:.2f}")

    # Confidence analysis for correct vs. incorrect predictions
    print("\nConfidence Analysis by Correctness:")
    for persona in personas:
        mask = df['persona'] == persona
        correct_mask = mask & (df['accuracy_rate'] == 1)
        incorrect_mask = mask & (df['accuracy_rate'] == 0)
        correct_conf = df['confidence_score'][correct_mask].mean() if correct_mask.sum() > 0 else 0.0
        incorrect_conf = df['confidence_score'][incorrect_mask].mean() if incorrect_mask.sum() > 0 else 0.0
        print(f"Persona '{persona}':")
        print(f"  Avg Confidence (Correct): {correct_conf:.2f} (Count: {correct_mask.sum()})")
        print(f"  Avg Confidence (Incorrect): {incorrect_conf:.2f} (Count: {incorrect_mask.sum()})")

    # Create summary DataFrame
    summary_df = pd.DataFrame(persona_metrics + [overall_metrics])

    # Compare with provided correction rates
    provided_rates = {
        'csnp': 0.8811, 'dental': 0.9091, 'doctor': 0.9583,
        'drug': 0.8830, 'dsnp': 1.0000, 'vision': 1.0000
    }
    print("\nProvided vs. Computed Correction Rates:")
    for persona in provided_rates:
        computed = next((m['accuracy_rate'] for m in persona_metrics if m['persona'] == persona), 0.0)
        print(f"{persona}: Provided = {provided_rates[persona]:.4f}, Computed = {computed:.4f}")

    # Select columns for output
    output_cols = [
        'userid', 'zip', 'plan_id', 'persona', 'predicted_persona'
    ] + weight_cols + [
        'probability_ranking', 'confidence_score', 'accuracy_rate', 'overall_accuracy'
    ]
    available_cols = [col for col in output_cols if col in df.columns]
    output_df = df[available_cols].copy()

    return output_df, summary_df, overall_accuracy

def save_outputs(output_df, summary_df, output_file, summary_file):
    """Save the evaluation outcomes and summary to CSV files."""
    try:
        output_df.to_csv(output_file, index=False)
        print(f"Saved evaluation outcomes to {output_file}")
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
    except Exception as e:
        print(f"Error saving files: {e}")
        raise

def main():
    # Load the output file
    df = load_output_file(OUTPUT_FILE)

    # Compute evaluation outcomes and summary
    output_df, summary_df, overall_accuracy = compute_evaluation_outcomes(df)

    # Save outputs
    save_outputs(output_df, summary_df, NEW_OUTPUT_FILE, SUMMARY_OUTPUT_FILE)

if __name__ == "__main__":
    main()
