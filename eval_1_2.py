import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BEHAVIORAL_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv"
)
PLAN_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/training/plan_derivation_by_zip.csv"
)
MODEL_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/model-persona-1.1.2.pkl"
)
LABEL_ENCODER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/label_encoder_1.pkl"
)
TRANSFORMER_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/power_transformer.pkl"
)

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'csnp']

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Helper Functions ---
def get_feature_as_series(df, col_name, default=0):
    try:
        if col_name in df.columns:
            return df[col_name]
        logger.debug(f"Column {col_name} missing, using default {default}")
        return pd.Series([default] * len(df), index=df.index)
    except Exception as e:
        logger.error(f"Error in get_feature_as_series for {col_name}: {e}")
        raise

def normalize_persona(df):
    try:
        valid_personas = PERSONAS
        new_rows = []
        invalid_personas = set()
        
        if 'persona' not in df.columns:
            logger.error("Column 'persona' missing in behavioral_df")
            return pd.DataFrame()
        
        for _, row in df.iterrows():
            persona = row['persona']
            if pd.isna(persona) or not persona:
                continue
            
            personas = [p.strip().lower() for p in str(persona).split(',')]
            valid_found = [p for p in personas if p in valid_personas]
            
            if not valid_found:
                invalid_personas.update(personas)
                continue
            
            row_copy = row.copy()
            row_copy['persona'] = valid_found[0]
            new_rows.append(row_copy)
        
        result = pd.DataFrame(new_rows).reset_index(drop=True)
        logger.info(f"Rows after persona normalization: {len(result)}")
        if invalid_personas:
            logger.info(f"Invalid personas found: {invalid_personas}")
        if result.empty:
            logger.warning(f"No valid personas found. Valid personas: {valid_personas}")
        return result
    except Exception as e:
        logger.error(f"Error in normalize_persona: {e}")
        raise

def load_data(behavioral_path, plan_path):
    try:
        if not os.path.exists(behavioral_path):
            raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}, columns: {list(behavioral_df.columns)}")
        
        persona_mapping = {'fitness': 'otc', 'hearing': 'vision'}
        behavioral_df['persona'] = behavioral_df['persona'].replace(persona_mapping)
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
        
        behavioral_df['zip'] = behavioral_df['zip'].fillna('unknown')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].fillna('unknown')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        
        for col in ['query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp']:
            if col in behavioral_df.columns:
                logger.info(f"{col} stats: mean={behavioral_df[col].mean():.2f}, std={behavioral_df[col].std():.2f}, missing={behavioral_df[col].isna().sum()}, non-zero={len(behavioral_df[behavioral_df[col] > 0])}")
            else:
                logger.warning(f"Key feature {col} missing in behavioral_df")
        
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan data rows: {len(plan_df)}, columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def prepare_features(behavioral_df, plan_df, expected_features):
    try:
        logger.info(f"Behavioral_df shape: {behavioral_df.shape}, columns: {list(behavioral_df.columns)}")
        logger.info(f"Plan_df shape: {plan_df.shape}, columns: {list(plan_df.columns)}")
        
        behavioral_df = normalize_persona(behavioral_df)
        if behavioral_df.empty:
            logger.warning("Behavioral_df is empty after normalization. Using plan_df with default persona.")
            training_df = plan_df.copy()
            training_df['persona'] = 'dental'
        else:
            training_df = behavioral_df.merge(
                plan_df.rename(columns={'StateCode': 'state'}),
                how='left', on=['zip', 'plan_id']
            ).reset_index(drop=True)
            logger.info(f"Rows after merge: {len(training_df)}, columns: {list(training_df.columns)}")
        
        if training_df.empty:
            logger.error("Training_df is empty after merge")
            raise ValueError("Empty training DataFrame after merge")
        
        # Raw features
        raw_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_dsnp_pages',
            'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_dsnp',
            'ma_dental_benefit', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_network'
        ]
        
        imputer_median = SimpleImputer(strategy='median')
        imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
        
        for col in raw_features:
            if col in training_df.columns:
                if col.startswith('query_') or col.startswith('time_'):
                    training_df[col] = imputer_zero.fit_transform(training_df[[col]]).flatten()
                else:
                    training_df[col] = imputer_median.fit_transform(training_df[[col]]).flatten()
            else:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
                logger.debug(f"Created column {col} with default value 0")
        
        # Minimal feature engineering (to be moved to training script)
        additional_features = []
        
        dental_query = get_feature_as_series(training_df, 'query_dental')
        drug_query = get_feature_as_series(training_df, 'query_drug')
        provider_query = get_feature_as_series(training_df, 'query_provider')
        csnp_query = get_feature_as_series(training_df, 'query_csnp')
        dsnp_query = get_feature_as_series(training_df, 'query_dsnp')
        
        training_df['dental_drug_ratio'] = (
            (dental_query + 0.8) / (drug_query + dental_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dental_drug_ratio')
        
        training_df['drug_dental_ratio'] = (
            (drug_query + 0.8) / (dental_query + drug_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('drug_dental_ratio')
        
        training_df['dental_doctor_interaction'] = (
            dental_query * provider_query
        ).clip(upper=10) * 6.0
        additional_features.append('dental_doctor_interaction')
        
        training_df['dental_dsnp_ratio'] = (
            (dental_query + 0.8) / (dsnp_query + dental_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dental_dsnp_ratio')
        
        if 'start_time' in training_df.columns:
            try:
                start_time = pd.to_datetime(training_df['start_time'], errors='coerce')
                training_df['recency'] = (pd.to_datetime('2025-05-29') - start_time).dt.days.fillna(30)
                training_df['time_of_day'] = start_time.dt.hour.fillna(12) // 6
                if 'userid' in training_df.columns:
                    training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
                else:
                    training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
            except Exception as e:
                logger.warning(f"Failed to process start_time: {e}")
                training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
                training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
                training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
        else:
            training_df['recency'] = pd.Series([30] * len(training_df), index=training_df.index)
            training_df['visit_frequency'] = pd.Series([1] * len(training_df), index=training_df.index)
            training_df['time_of_day'] = pd.Series([2] * len(training_df), index=training_df.index)
        
        feature_columns = raw_features + additional_features + ['recency', 'time_of_day', 'visit_frequency']
        X = training_df[feature_columns].fillna(0)
        
        if 'persona' not in training_df.columns:
            logger.error("Column 'persona' missing in training_df")
            raise KeyError("Column 'persona' missing")
        y = training_df['persona']
        
        if expected_features:
            missing_features = [f for f in expected_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for f in missing_features:
                    X[f] = 0
            X = X[expected_features]
        
        logger.info(f"Generated features: {list(X.columns)}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def compute_per_persona_accuracy(y_true, y_pred, classes):
    try:
        per_persona_accuracy = {}
        for cls_idx, cls_name in enumerate(classes):
            if cls_name not in PERSONAS:
                continue
            mask = y_true == cls_idx
            if mask.sum() > 0:
                cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                per_persona_accuracy[cls_name] = cls_accuracy * 100
            else:
                per_persona_accuracy[cls_name] = 0.0
                logger.warning(f"No test samples for {cls_name}")
        return per_persona_accuracy
    except Exception as e:
        logger.error(f"Error in compute_per_persona_accuracy: {e}")
        raise

def evaluate_model(main_model, le, transformer, X_test, y_test_encoded):
    try:
        logger.info(f"Evaluating model with X_test shape: {X_test.shape}")
        X_test_transformed = transformer.transform(X_test)
        y_pred_proba = main_model.predict_proba(X_test_transformed)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        overall_acc = accuracy_score(y_test_encoded, y_pred)
        macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
        per_persona_acc = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_)
        
        class_distribution = pd.Series(y_test_encoded).value_counts().sort_index()
        logger.info("\nClass distribution in test data:")
        for idx, count in class_distribution.items():
            if idx < len(le.classes_):
                logger.info(f"{le.classes_[idx]}: {count}")
        
        prediction_distribution = pd.Series(y_pred).value_counts().sort_index()
        logger.info("\nPrediction distribution:")
        for idx, count in prediction_distribution.items():
            if idx < len(le.classes_):
                logger.info(f"{le.classes_[idx]}: {count}")
        
        if hasattr(main_model, 'get_feature_importance'):
            logger.info("\nTop 10 features by importance:")
            importance = main_model.get_feature_importance()
            feature_importance = list(zip(X_test.columns, importance))
            for feature, imp in sorted(feature_importance, key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"{feature}: {imp:.2f}")
        
        for persona in ['dental', 'doctor', 'csnp']:
            signal_count = sum(1 for i in range(len(y_test_encoded)) if y_test_encoded[i] == le.transform([persona])[0])
            correct_predictions = sum(1 for i in range(len(y_pred)) if y_pred[i] == le.transform([persona])[0] and y_test_encoded[i] == le.transform([persona])[0])
            if signal_count > 0:
                logger.info(f"\n{persona} signal analysis:")
                logger.info(f"Total {persona} samples: {signal_count}")
                logger.info(f"Correctly predicted: {correct_predictions} ({correct_predictions / signal_count * 100:.2f}%)")
                
                misclassified = [le.classes_[y_pred[i]] for i in range(len(y_pred)) if y_test_encoded[i] == le.transform([persona])[0] and y_pred[i] != le.transform([persona])[0])
                if misclassified:
                    misclassified_counts = pd.Series(misclassified).value_counts()
                    logger.info(f"{persona} misclassified as: {dict(misclassified_counts)}")
        
        return overall_acc, macro_f1, per_persona_acc, y_pred, y_pred_proba
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def create_detailed_report(X_test, y_test, y_pred, y_proba, le):
    try:
        result_df = X_test.copy()
        result_df['true_persona'] = le.inverse_transform(y_test)
        result_df['predicted_persona'] = le.inverse_transform(y_pred)
        result_df['correct'] = result_df['true_persona'] == result_df['predicted_persona']
        
        for i, persona in enumerate(le.classes_):
            result_df[f'prob_{persona}'] = y_proba[:, i]
        
        for persona in ['dental', 'doctor', 'csnp']:
            samples = result_df[result_df['true_persona'] == persona]
            logger.info(f"\n{persona} samples analysis (total: {len(samples)}):")
            if len(samples) > 0:
                correct = samples[samples['correct']]
                logger.info(f"Correctly classified: {len(correct)} ({len(correct)/len(samples)*100:.2f}%)")
                
                features = [f'query_{persona}', f'filter_{persona}', f'time_{persona}_pages'] if persona != 'doctor' else ['query_provider', 'filter_provider', 'click_provider']
                features += ['dental_drug_ratio', 'drug_dental_ratio', 'dental_doctor_interaction', 'dental_dsnp_ratio']
                logger.info(f"Average feature values for {persona} samples:")
                for feature in features:
                    if feature in samples.columns:
                        avg_val = samples[feature].mean()
                        logger.info(f"{feature}: {avg_val:.4f}")
        
        report_path = os.path.join(os.path.dirname(MODEL_FILE), "evaluation_detailed_report.csv")
        result_df.to_csv(report_path, index=False)
        logger.info(f"Detailed report saved to: {report_path}")
        
        return result_df
    except Exception as e:
        logger.error(f"Error creating detailed report: {e}")
        return None

def create_visualizations(X_test, y_test, y_pred, le):
    try:
        persona_indices = [i for i, p in enumerate(le.classes_) if p in PERSONAS]
        filtered_classes = [le.classes_[i] for i in persona_indices]
        
        valid_mask = np.isin(y_test, persona_indices) & np.isin(y_pred, persona_indices)
        y_test_filtered = y_test[valid_mask]
        y_pred_filtered = y_pred[valid_mask]
        
        y_test_mapped = np.array([filtered_classes.index(le.classes_[x]) for x in y_test_filtered])
        y_pred_mapped = np.array([filtered_classes.index(le.classes_[x]) for x in y_pred_filtered])
        
        cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=range(len(filtered_classes)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_classes, yticklabels=filtered_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        persona_acc = []
        for idx, persona in enumerate(filtered_classes):
            mask = y_test_mapped == idx
            acc = accuracy_score(y_test_mapped[mask], y_pred_mapped[mask]) * 100 if mask.sum() > 0 else 0.0
            persona_acc.append(acc)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=filtered_classes, y=persona_acc)
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Persona Accuracy')
        plt.tight_layout()
        plt.savefig('per_persona_accuracy.png')
        plt.close()
        
        logger.info("Saved confusion_matrix.png and per_persona_accuracy.png")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise

def main():
    logger.info("Starting evaluation at 01:57 PM CDT, May 29, 2025...")
    
    try:
        for file_path in [MODEL_FILE, LABEL_ENCODER_FILE, TRANSFORMER_FILE]:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(MODEL_FILE, 'rb') as f:
            main_model = pickle.load(f)
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            le = pickle.load(f)
        with open(TRANSFORMER_FILE, 'rb') as f:
            transformer = pickle.load(f)
        
        expected_features = getattr(main_model, 'feature_names_', None)
        if expected_features is None:
            logger.warning("Model does not provide feature_names_. Using default feature set.")
            expected_features = [
                'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp',
                'filter_dental', 'filter_drug', 'filter_provider', 'filter_csnp', 'filter_dsnp',
                'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
                'time_csnp_pages', 'time_drug_pages', 'time_dsnp_pages',
                'accordion_csnp', 'accordion_dental', 'accordion_drug', 'accordion_dsnp',
                'ma_dental_benefit', 'csnp', 'dsnp', 'ma_drug_benefit', 'ma_provider_dental',
                'recency', 'time_of_day', 'visit_frequency',
                'dental_drug_ratio', 'drug_dental_ratio', 'dental_doctor_interaction', 'dental_dsnp_ratio'
            ]
        logger.info(f"Expected features: {expected_features}")
    except Exception as e:
        logger.error(f"Failed to load model or files: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise SystemExit(1)
    
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
        
        if 'persona' in behavioral_df.columns:
            for persona in PERSONAS:
                count = len(behavioral_df[behavioral_dfbrica['persona'].str.lower().str.contains(persona, na=False)])
                logger.info(f"{persona} samples in raw data: {count}")
        
        X_data, y_data = prepare_features(behavioral_df, plan_df, expected_features)
        
        # Validate y_data
        invalid_labels = set(y_data) - set(le.classes_)
        if invalid_labels:
            logger.warning(f"Invalid labels in y_data: {invalid_labels}. Replacing with 'dental'.")
            y_data = y_data.apply(lambda x: 'dental' if x in invalid_labels else x)
        
        y_data_encoded = le.transform(y_data)
        
        class_distribution = pd.Series(y_data).value_counts()
        logger.info("\nClass distribution in evaluation data:")
        for persona, count in class_distribution.items():
            logger.info(f"{persona}: {count}")
        
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise SystemExit(1)
    
    try:
        overall_acc, macro_f1, per_persona_acc, y_pred, y_proba = evaluate_model(
            main_model, le, transformer, X_data, y_data_encoded
        )
        
        logger.info(f"\nEvaluation Results:")
        logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
        logger.info(f"Macro F1: {macro_f1:.2f}")
        logger.info("\nPer-Persona Accuracy:")
        for persona, acc in per_persona_acc.items():
            logger.info(f"{persona}: {acc:.2f}%")
        
        if overall_acc < 0.8:
            logger.warning("Overall accuracy below 80%. Consider retraining with adjusted parameters.")
        
        detailed_report = create_detailed_report(X_data, y_data_encoded, y_pred, y_proba, le)
        
        create_visualizations(X_data, y_data_encoded, y_pred, le)
        
        logger.info("\nRecommendations for training:")
        logger.info("1. Increase oversampling for 'dental' and 'doctor':")
        logger.info("   PERSONA_OVERSAMPLING_RATIO = {'dental': 50.0, 'doctor': 40.0, 'drug': 10.0, 'dsnp': 15.0, 'csnp': 15.0}")
        logger.info("2. Adjust class weights:")
        logger.info("   PERSONA_CLASS_WEIGHT = {'dental': 50.0, 'doctor': 40.0, 'drug': 10.0, 'dsnp': 15.0, 'csnp': 15.0}")
        logger.info("3. Save feature engineering pipeline:")
        logger.info("   Save feature_names.pkl and consider a preprocessing pipeline")
        
        logger.info("\n===== TOTAL SUMMARY REPORT =====")
        logger.info(f"Total samples evaluated: {len(y_data)}")
        logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
        logger.info(f"Macro F1 Score: {macro_f1:.4f}")
        logger.info("Per-Persona Accuracy:")
        for persona in PERSONAS:
            acc = per_persona_acc.get(persona, 0.0)
            logger.info(f"  {persona}: {acc:.2f}%")
        logger.info("====================================")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise SystemExit(1)

if __name__ == "__main__":
    main()
