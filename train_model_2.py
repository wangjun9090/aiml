import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import logging
import sys

# --- Configuration ---
BEHAVIORAL_FILE = ('/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025_clean.csv'
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
FEATURE_NAMES_FILE = (
    "/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/"
    "data/s-learning-data/models/feature_names.pkl"
)

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'csnp']

# Oversampling target counts (balanced)
PERSONA_OVERSAMPLING_COUNTS = {
    'dental': 30000,  # ~50x for 605 raw samples
    'doctor': 30000,  # ~11x for 2741
    'drug': 30000,    # ~12.5x for 2401
    'dsnp': 30000,    # ~37x for 809
    'csnp': 30000     # ~21x for 1409
}

# Class weights
PERSONA_CLASS_WEIGHT = {
    'dental': 30.0,
    'doctor': 20.0,
    'drug': 20.0,
    'dsnp': 30.0,
    'csnp': 25.0
}

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
        
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip().fillna('0')
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip().fillna('0')
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
        
        for col in ['persona', 'query_dental', 'query_drug', 'query_provider', 'query_csnp', 'query_dsnp']:
            if col in behavioral_df.columns:
                if col.startswith('query_'):
                    logger.info(f"{col} stats: mean={behavioral_df[col].mean():.2f}, std={behavioral_df[col].std():.2f}, missing={behavioral_df[col].isna().sum()}, non-zero={len(behavioral_df[behavioral_df[col] > 0])}")
                else:
                    logger.info(f"{col} values: {behavioral_df[col].value_counts().to_dict()}")
            else:
                logger.warning(f"Key column {col} missing in behavioral_df")
        
        plan_df = pd.read_csv(plan_path)
        logger.info(f"Plan data rows: {len(plan_df)}, columns: {list(plan_df.columns)}")
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip().fillna('0')
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip().fillna('0')
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def prepare_features(behavioral_df, plan_df):
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
                logger.info(f"Processing column {col}, non-null count: {training_df[col].notna().sum()}")
                if training_df[col].notna().sum() == 0:
                    logger.warning(f"Column {col} is all NaN, filling with 0")
                    training_df[col] = 0
                else:
                    try:
                        if col.startswith('query_') or col.startswith('time_'):
                            training_df[col] = np.clip(training_df[col], 0, 1e6)  # Prevent overflow
                            transformed = imputer_zero.fit_transform(training_df[[col]])
                            training_df[col] = transformed.flatten()
                        else:
                            training_df[col] = np.clip(training_df[col], 0, 1e6)
                            transformed = imputer_median.fit_transform(training_df[[col]])
                            training_df[col] = transformed.flatten()
                    except Exception as e:
                        logger.error(f"Imputation failed for {col}: {e}")
                        training_df[col] = 0
            else:
                training_df[col] = pd.Series([0] * len(training_df), index=training_df.index)
                logger.debug(f"Created column {col} with default value 0")
        
        # Feature engineering
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
        
        training_df['drug_csnp_ratio'] = (
            (drug_query + 0.8) / (csnp_query + drug_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('drug_csnp_ratio')
        
        training_df['dsnp_drug_ratio'] = (
            (dsnp_query + 0.8) / (drug_query + dsnp_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('dsnp_drug_ratio')
        
        training_df['csnp_dental_ratio'] = (
            (csnp_query + 0.8) / (dental_query + csnp_query + 1e-6)
        ).clip(lower=0, upper=1) * 6.0
        additional_features.append('csnp_dental_ratio')
        
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
        
        logger.info(f"Generated features: {list(X.columns)}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y, feature_columns
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def create_visualizations(X_val, y_val, y_pred, le):
    try:
        cm = confusion_matrix(y_val, y_pred, labels=range(len(le.classes_)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('training_confusion_matrix.png')
        plt.close()
        
        per_persona_acc = []
        for cls_idx, cls_name in enumerate(le.classes_):
            if cls_name not in PERSONAS:
                continue
            mask = y_val == cls_idx
            acc = accuracy_score(y_val[mask], y_pred[mask]) * 100 if mask.sum() > 0 else 0
            per_persona_acc.append(acc)
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=le.classes_, y=per_persona_acc)
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Persona Accuracy')
        plt.tight_layout()
        plt.savefig('training_per_persona_accuracy.png')
        plt.close()
        
        logger.info('Saved training_confusion_matrix.png and training_per_persona_accuracy.png')
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def main():
    logger.info("Starting training at 02:50 PM CDT, May 29, 2025...")
    
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
        
        X, y, feature_columns = prepare_features(behavioral_df, plan_df)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Log class distribution
        class_distribution = pd.Series(y).value_counts()
        logger.info("\nClass distribution in training data:")
        for persona, count in class_distribution.items():
            logger.info(f"{persona}: {count}")
        
        # Split data (80/20 rule)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        logger.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        
        # Apply PowerTransformer
        transformer = PowerTransformer(method='yeo-johnson')
        X_train_transformed = transformer.fit_transform(X_train)
        X_val_transformed = transformer.transform(X_val)
        
        # Oversampling
        smote = SMOTE(
            sampling_strategy={
                le.transform([p])[0]: PERSONA_OVERSAMPLING_COUNTS.get(p, 1000)
                for p in PERSONAS
            },
            random_state=42
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        # Log resampled distribution
        resampled_distribution = pd.Series(le.inverse_transform(y_train_resampled)).value_counts()
        logger.info("\nResampled class distribution:")
        for persona, count in resampled_distribution.items():
            logger.info(f"{persona}: {count}")
        
        # Class weights
        class_weights = {
            le.transform([p])[0]: PERSONA_CLASS_WEIGHT.get(p, 1.0)
            for p in PERSONAS
        }
        
        # Train model
        model = XGBClassifier(
            n_estimators=200,  # Increased
            max_depth=7,      # Increased
            learning_rate=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_resampled, y_train_resampled, sample_weight=[class_weights.get(y, 1.0) for y in y_train_resampled])
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
        logger.info(f"\nCross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        # Validate
        y_pred = model.predict(X_val_transformed)
        overall_acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        logger.info(f"\nValidation Results:")
        logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        
        # Per-persona accuracy
        per_persona_acc = {}
        for cls_idx, cls_name in enumerate(le.classes_):
            if cls_name not in PERSONAS:
                continue
            mask = y_val == cls_idx
            if mask.sum() > 0:
                cls_accuracy = accuracy_score(y_val[mask], y_pred[mask])
                per_persona_acc[cls_name] = cls_accuracy * 100
            else:
                per_persona_acc[cls_name] = 0.0
                logger.warning(f"No validation samples for {cls_name}")
        
        logger.info("\nPer-Persona Accuracy:")
        for persona, acc in per_persona_acc.items():
            logger.info(f"{persona}: {acc:.2f}%")
        
        # Visualizations
        create_visualizations(X_val, y_val, y_pred, le)
        
        # Save artifacts
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(le, f)
        with open(TRANSFORMER_FILE, 'wb') as f:
            pickle.dump(transformer, f)
        with open(FEATURE_NAMES_FILE, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        logger.info(f"Saved model to {MODEL_FILE}")
        logger.info(f"Saved label encoder to {LABEL_ENCODER_FILE}")
        logger.info(f"Saved transformer to {TRANSFORMER_FILE}")
        logger.info(f"Saved feature names to {FEATURE_NAMES_FILE}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
