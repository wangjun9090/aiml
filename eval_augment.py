
# eval_augment_boosted.py

import pandas as pd
import numpy as np
import pickle
import logging
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()

PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']
PERSONA_CLASS_WEIGHT = {
    'drug': 5.5,
    'dental': 10.0,
    'doctor': 9.0,
    'dsnp': 4.5,
    'vision': 6.0,
    'csnp': 5.0
}
PERSONA_THRESHOLD = {
    'drug': 0.20,
    'dental': 0.05,
    'doctor': 0.05,
    'dsnp': 0.20,
    'vision': 0.20,
    'csnp': 0.20
}

MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.1.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
TRANSFORMER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/power_transformer.pkl'

BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'

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
    logger.info("Starting boosted model evaluation...")

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        le = pickle.load(f)
    with open(TRANSFORMER_FILE, 'rb') as f:
        transformer = pickle.load(f)

    behavioral_df = pd.read_csv(BEHAVIORAL_FILE)
    plan_df = pd.read_csv(PLAN_FILE)

    X_test = behavioral_df.drop(columns=['persona'])
    y_test = behavioral_df['persona']

    X_test = pd.DataFrame(transformer.transform(X_test), columns=X_test.columns)
    y_test_encoded = le.transform(y_test)

    y_pred_proba = model.predict_proba(X_test)

    binary_classifiers = {}
    for persona in PERSONAS:
        binary_model_path = MODEL_FILE.replace('.pkl', f'_{persona}_binary.pkl')
        with open(binary_model_path, 'rb') as f:
            binary_classifiers[persona] = pickle.load(f)

    binary_probas = {persona: binary_classifiers[persona].predict_proba(X_test)[:, 1] for persona in PERSONAS}

    for i, persona in enumerate(le.classes_):
        if persona in binary_probas:
            if persona in ['dental', 'doctor']:
                blend_ratio = 0.05
            elif persona in ['drug', 'csnp']:
                blend_ratio = 0.7
            else:
                blend_ratio = 0.5
            y_pred_proba[:, i] = blend_ratio * y_pred_proba[:, i] + (1-blend_ratio) * binary_probas[persona]
        y_pred_proba[:, i] *= PERSONA_CLASS_WEIGHT.get(persona, 1.0)

    y_pred_proba /= y_pred_proba.sum(axis=1, keepdims=True)

    y_pred = np.zeros(y_pred_proba.shape[0], dtype=int)
    for i in range(y_pred_proba.shape[0]):
        if binary_probas['dental'][i] > 0.15:
            y_pred[i] = np.where(le.classes_ == 'dental')[0][0]
        else:
            max_prob = -1
            max_idx = 0
            for j, persona in enumerate(le.classes_):
                prob = y_pred_proba[i, j]
                threshold = PERSONA_THRESHOLD.get(persona, 0.5)
                if prob > threshold and prob > max_prob:
                    max_prob = prob
                    max_idx = j
            y_pred[i] = max_idx

    overall_acc = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')

    logger.info(f"Boosted Overall Accuracy: {overall_acc * 100:.2f}%")
    logger.info(f"Boosted Macro F1: {macro_f1:.2f}")

    per_persona_acc = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
    for persona in PERSONAS:
        logger.info(f"{persona}: {per_persona_acc.get(persona, 0):.2f}%")

    logger.info("Done!")

if __name__ == "__main__":
    main()
