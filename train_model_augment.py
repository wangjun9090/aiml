import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
import logging
import sys
import os
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)

# File paths
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.0.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp', 'transportation', 'otc']

# Persona info
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
}

def calculate_persona_weight(row, persona_info, persona):
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    plan_col = persona_info.get('plan_col', None)
    click_col = persona_info.get('click_col', None)
    
    query_value = row.get(query_col, 0) if pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col, np.nan)) else 0
    plan_value = row.get(plan_col, 0) if plan_col and pd.notna(row.get(plan_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    max_val = max([query_value, filter_value, plan_value, click_value, 1])
    if max_val > 0:
        query_value /= max_val
        filter_value /= max_val
        plan_value /= max_val
        click_value /= max_val
    
    weight = 0.25 * (query_value + filter_value + plan_value + click_value)
    return min(max(weight, 0), 1.0)

def load_data(behavioral_path, plan_path):
    try:
        # Check file existence
        if not os.path.exists(behavioral_path):
            raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
        # Load behavioral data
        behavioral_df = pd.read_csv(behavioral_path)
        logger.info(f"Raw behavioral data rows: {len(behavioral_df)}")
        logger.info(f"Raw behavioral columns: {list(behavioral_df.columns)}")
        
        # Log unique personas before cleaning
        if 'persona' in behavioral_df.columns:
            logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
            logger.info(f"Persona value counts:\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        else:
            logger.warning("Persona column missing in behavioral data")
        
        # Clean behavioral data
        required_cols = ['zip', 'plan_id']
        initial_rows = len(behavioral_df)
        behavioral_df = behavioral_df.dropna(subset=required_cols)
        logger.info(f"Rows after dropna: {len(behavioral_df)} (dropped {initial_rows - len(behavioral_df)})")
        
        # Standardize columns
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip()
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip()
        if 'persona' in behavioral_df.columns:
            behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.lower().str.strip()
            behavioral_df['persona'] = behavioral_df['persona'].replace(['nan', ''], np.nan)
        else:
            behavioral_df['persona'] = np.nan
        
        # Relaxed session time filter
        if 'total_session_time' in behavioral_df.columns:
            behavioral_df['total_session_time'] = behavioral_df['total_session_time'].fillna(0)
            behavioral_df = behavioral_df[behavioral_df['total_session_time'].between(0, 7200)]  # 0s to 2hr
        logger.info(f"Behavioral_df after cleaning: {len(behavioral_df)} rows")
        
        # Load plan data
        plan_df = pd.read_csv(plan_path)
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()  # Fixed typo: .strip() → .str.strip()
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def normalize_persona(df):
    valid_personas = PERSONAS
    new_rows = []
    invalid_personas = set()
    
    for _, row in df.iterrows():
        persona = row['persona']
        if pd.isna(persona) or not persona:
            logger.debug(f"Skipping row with missing persona: {row.get('userid', 'unknown')}")
            continue
        
        personas = [p.strip().lower() for p in str(persona).split(',')]
        valid_found = [p for p in personas if p in valid_personas]
        
        if not valid_found:
            invalid_personas.update(personas)
            logger.debug(f"No valid personas in: {personas} for user {row.get('userid', 'unknown')}")
            continue
        
        row_copy = row.copy()
        row_copy['persona'] = valid_found[0]  # Take first valid persona
        new_rows.append(row_copy)
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(f"Rows after persona normalization: {len(result)}")
    if invalid_personas:
        logger.info(f"Invalid personas found: {invalid_personas}")
    if result.empty:
        logger.warning(f"No valid personas found. Valid personas: {valid_personas}")
    return result

def prepare_features(behavioral_df, plan_df):
    try:
        # Normalize personas
        behavioral_df = normalize_persona(behavioral_df)
        
        # Check if behavioral_df is empty
        if behavioral_df.empty:
            logger.warning("Behavioral_df is empty after normalization. Using plan_df only.")
            training_df = plan_df.copy()
            training_df['persona'] = np.nan
        else:
            # Merge data
            training_df = behavioral_df.merge(
                plan_df.rename(columns={'StateCode': 'state'}),
                how='left', on=['zip', 'plan_id']
            ).reset_index(drop=True)
            logger.info(f"Rows after merge: {len(training_df)}")
        
        # Define features
        plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp', 'ma_drug_coverage', 'ma_provider_network']
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks',
            'time_csnp_pages', 'time_drug_pages', 'time_vision_pages', 'accordion_csnp',
            'accordion_dental', 'accordion_drug', 'accordion_provider', 'accordion_vision', 'accordion_dsnp'
        ]
        
        # Temporal features
        if 'start_time' in training_df.columns:  # Use start_time instead of timestamp
            training_df['recency'] = (pd.to_datetime('2025-04-25') - pd.to_datetime(training_df['start_time'])).dt.days.fillna(30)
            training_df['visit_frequency'] = training_df.groupby('userid')['start_time'].transform('count').fillna(1) / 30
            training_df['time_of_day'] = pd.to_datetime(training_df['start_time']).dt.hour.fillna(12) // 6
        else:
            training_df['recency'] = 30
            training_df['visit_frequency'] = 1
            training_df['time_of_day'] = 2
        
        # Clustering feature
        cluster_features = ['num_pages_viewed', 'total_session_time', 'num_clicks']
        if all(col in training_df.columns for col in cluster_features):
            kmeans = KMeans(n_clusters=5, random_state=42)
            training_df['user_cluster'] = kmeans.fit_predict(training_df[cluster_features].fillna(0))
        else:
            training_df['user_cluster'] = 0
        
        # Sequence feature
        training_df['nav_path'] = 'unknown'  # Simplified due to missing 'page' column
        
        # Robust aggregates
        training_df['dental_time_ratio'] = training_df.get('time_dental_pages', 0) / (training_df.get('total_session_time', 1) + 1e-5)
        training_df['click_ratio'] = training_df.get('num_clicks', 0) / (training_df.get('num_pages_viewed', 1) + 1e-5)
        
        # Plan ID embeddings
        if 'plan_id' in training_df.columns:
            plan_sentences = training_df.groupby('userid')['plan_id'].apply(list).tolist()
            w2v_model = Word2Vec(sentences=plan_sentences, vector_size=10, window=5, min_count=1, workers=4)
            plan_embeddings = training_df['plan_id'].apply(
                lambda x: w2v_model.wv[x] if x in w2v_model.wv else np.zeros(10)
            )
            embedding_cols = [f'plan_emb_{i}' for i in range(10)]
            training_df[embedding_cols] = pd.DataFrame(plan_embeddings.tolist(), index=training_df.index)
        else:
            embedding_cols = [f'plan_emb_{i}' for i in range(10)]
            training_df[embedding_cols] = 0
        
        # Aggregate features
        query_cols = [c for c in behavioral_features if c.startswith('query_') and c in training_df.columns]
        filter_cols = [c for c in behavioral_features if c.startswith('filter_') and c in training_df.columns]
        training_df['query_count'] = training_df[query_cols].sum(axis=1) if query_cols else pd.Series(0, index=training_df.index)
        training_df['filter_count'] = training_df[filter_cols].sum(axis=1) if filter_cols else pd.Series(0, index=training_df.index)
        
        # Initialize missing columns
        for col in behavioral_features + plan_features:
            if col not in training_df.columns:
                training_df[col] = pd.Series(0, index=training_df.index)
            else:
                training_df[col] = training_df[col].fillna(0)
        
        # Persona weights
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        # Domain-specific features
        additional_features = []
        training_df['csnp_interaction'] = training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
        ) * 2.5
        additional_features.append('csnp_interaction')

        training_df['csnp_type_flag'] = training_df.get('csnp_type', 'N').map({'Y': 1, 'N': 0}).fillna(0).astype(int)
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

        training_df['vision_signal'] = (
            training_df['query_vision'].fillna(0) +
            training_df['filter_vision'].fillna(0) +
            training_df['time_vision_pages'].fillna(0).clip(upper=5)
        ) * 2.0
        additional_features.append('vision_signal')

        training_df['dental_signal'] = (
            training_df['query_dental'].fillna(0) +
            training_df['filter_dental'].fillna(0) +
            training_df['time_dental_pages'].fillna(0).clip(upper=5)
        ) * 2.0
        additional_features.append('dental_signal')

        training_df['csnp_specific_signal'] = (
            training_df['query_csnp'].fillna(0) +
            training_df['filter_csnp'].fillna(0) +
            training_df['csnp_drug_interaction'].fillna(0) +
            training_df['csnp_doctor_interaction'].fillna(0)
        ).clip(upper=5) * 3.0
        additional_features.append('csnp_specific_signal')
        
        # Compile features
        feature_columns = behavioral_features + plan_features + additional_features + [
            'recency', 'visit_frequency', 'time_of_day', 'user_cluster', 
            'dental_time_ratio', 'click_ratio'
        ] + embedding_cols + [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        
        training_df = training_df[training_df['persona'].notna()].reset_index(drop=True)
        logger.info(f"Rows after filtering: {len(training_df)}")
        logger.info(f"Final persona distribution:\n{training_df['persona'].value_counts(dropna=False).to_string()}")
        
        if training_df.empty:
            logger.error("No rows with valid personas after filtering")
            raise ValueError("No rows with valid personas after filtering")
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X, y = smote.fit_resample(X, y)
        logger.info(f"Rows after SMOTE: {len(X)}")
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y, scaler
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def pseudo_labeling(X_labeled, y_labeled, X_unlabeled, model):
    try:
        model.fit(X_labeled, y_labeled)
        y_unlabeled_pred = model.predict(X_unlabeled)
        confidence = model.predict_proba(X_unlabeled).max(axis=1)
        high_conf_mask = confidence > 0.9
        X_pseudo = X_unlabeled[high_conf_mask]
        y_pseudo = y_unlabeled_pred[high_conf_mask]
        return X_pseudo, y_pseudo
    except Exception as e:
        logger.warning(f"Pseudo-labeling failed: {e}")
        return pd.DataFrame(), np.array([])

def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    return per_persona_accuracy

def main():
    # Load data
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Prepare features
    try:
        X, y, scaler = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return
    
    # Semi-supervised learning
    labeled_mask = y.notna()
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    X_unlabeled = X[~labeled_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train CatBoost with pseudo-labeling and k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    feature_importances = []
    
    class_counts = pd.Series(y_train_encoded).value_counts()
    class_weights = {i: max(1.0 / class_counts[i], 1e-3) for i in class_counts.index}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train_encoded[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train_encoded[val_idx]
        
        model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=5,
            loss_function='MultiClass',
            class_weights=class_weights,
            random_seed=42,
            verbose=0
        )
        X_pseudo, y_pseudo = pseudo_labeling(X_fold_train, y_fold_train, X_unlabeled, model)
        if not X_pseudo.empty:
            X_fold_train = pd.concat([X_fold_train, X_pseudo])
            y_fold_train = np.concatenate([y_fold_train, le.transform(y_pseudo)])
        
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=50
        )
        models.append(model)
        feature_importances.append(model.get_feature_importance())
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions
    y_pred_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    y_pred = np.argmax(y_pred_probas, axis=1)
    
    # Log feature importances
    avg_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': avg_importances
    }).sort_values(by='Importance', ascending=False)
    logger.info("Feature Importances:\n" + importance_df.to_string())
    
    # Evaluate
    overall_accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    logger.info(f"Overall Accuracy on Test Set: {overall_accuracy * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
    if overall_accuracy < 0.8:
        logger.warning(f"Accuracy {overall_accuracy * 100:.2f}% is below target of 80%.")
    
    per_persona_accuracy = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    logger.info("Per-Persona Accuracy (%):")
    for persona, acc in per_persona_accuracy.items():
        logger.info(f"  {persona}: {acc:.2f}%")
    
    logger.info("Classification Report:\n" + classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    # Save model
    model = models[0]
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved model, label encoder, and scaler to disk.")

if __name__ == "__main__":
    main()
