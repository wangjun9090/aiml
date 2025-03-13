# Recap of Model Logic: Calculating `final_training_dataset.csv`

## 1. Data Collection

### Behavioral Data
- **Source**: `behavioral_features_0901_2024_0228_2025.csv`
- **Purpose**: Captures user interactions (Sept 1, 2024 - Feb 28, 2025)
- **Key Features** (45 total):
  - Identifiers: `userid`, `zip`, `plan_id`, `compared_plan_ids`
  - Engagement: `num_pages_viewed`, `num_plans_compared`, `num_plans_selected`, `total_session_time`, `submitted_application`
  - Intent Signals: 
    - 8 Queries: `query_dental`, `query_vision`, ..., `query_dsnp`
    - 8 Filters: `filter_dental`, `filter_vision`, ..., `filter_dsnp`
    - 8 Accordions: `accordion_dental`, ..., `accordion_dsnp`
  - Time Metrics: 
    - 8 Absolute: `time_dental_pages`, ..., `time_dsnp_pages`
    - 8 Relative: `rel_time_dental_pages`, ..., `rel_time_dsnp_pages`
- **Loading**: `behavioral_df = pd.read_csv(behavioral_file)`

### Plan Derivation Data
- **Source**: `plan_derivation_by_zip.csv`
- **Purpose**: Provides plan-specific persona weights by ZIP code
- **Key Features** (8 used):
  - Identifiers: `zip`, `plan_id`, `StateCode` (renamed to `state`)
  - Persona Weights: `ma_otc`, `ma_transportation`, `ma_dental_benefit`, `ma_vision`, `csnp`, `dsnp`, `ma_provider_network`, `ma_drug_coverage`
- **Loading**: `plan_df = pd.read_csv(plan_file)` with `StateCode` renamed to `state`

## 2. Data Structure and Merging
- **Merging Process**:
  - **Method**: Left merge on `zip` and `plan_id`
    - `training_df = behavioral_df.merge(plan_df, how='left', on=['zip', 'plan_id'])`
  - **Logic**: Links user behavior to plan attributes (or `NaN` if no match)
  - **State Resolution**: 
    - Combine `state_beh` and `state_plan` into `state`
    - Rule: Use `state_beh` if present, else `state_plan`, drop extras
  - **Result**: `training_df` with 45 behavioral + 8 plan features + identifiers

- **Target Variable**:
  - **Source**: `persona` column in `training_df`
  - **Calculation**: Extract first persona if comma-separated
    - `training_df['target_persona'] = training_df['persona'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) and ',' in x else x)`
  - **Purpose**: Defines ground truth persona (e.g., `"doctor"`)

## 3. Weight Calculation Logic
- **Objective**: Compute weights (`w_doctor`, `w_vision`, etc.) reflecting persona likelihood
- **Structure**:
  - **Personas**: 8 (`doctor`, `drug`, `vision`, `dental`, `otc`, `transportation`, `csnp`, `dsnp`)
  - **Mapping**: Each tied to plan column (e.g., `ma_vision`), query (e.g., `query_vision`), filter (e.g., `filter_vision`)
- **Formula**:
  - **Base Weight**: `base_weight = min(ma_X, 0.5)`
    - Caps plan influence
    - Fallback: Average `ma_X` from `compared_plan_ids` if `plan_id` missing
  - **Behavioral Score**: `k3 * query_value + k4 * filter_value + k1 * min(pages_viewed, 3)`
    - Constants: `k3 = 0.3` (query), `k4 = 0.2` (filter), `k1 = 0.1` (pages)
  - **Adjusted Weight**: `adjusted_weight = base_weight + behavioral_score`
  - **Target Adjustment**: If persona is `target_persona`:
    - `w_X = max(adjusted_weight, max(other_weights) + 0.1)`
  - **Cap**: `min(adjusted_weight, 1.0)`

- **Example (Row 3108)**:
  - **Inputs**: 
    - `ma_vision = 0.357143`, `ma_provider_network = 0.127844`
    - `num_pages_viewed = 1`, all queries/filters = 0
    - `target_persona = "doctor"`
  - **Calculation**:
    - Behavioral score: `0.3 * 0 + 0.2 * 0 + 0.1 * 1 = 0.1`
    - `w_vision = min(0.357143, 0.5) + 0.1 = 0.457143`
    - `w_doctor = min(0.127844, 0.5) + 0.1 = 0.227844`
      - Target boost: `max(0.227844, 0.457143 + 0.1) = 0.557143`
  - **Output**: `w_doctor = 0.557143 > w_vision = 0.457143`

## 4. Final Training Data Assembly
- **Structure**:
  - **Columns**: 66 total
    - Identifiers: `userid`, `zip`, `plan_id`, `state` (4)
    - Behavioral Features: 45 (e.g., `query_dental`, `num_pages_viewed`)
    - Plan Features: 8 (e.g., `ma_vision`)
    - Weights: 8 (e.g., `w_doctor`)
    - Target: `target_persona` (1)
- **Processing**:
  - Fill `NaN` in features: `final_training_df[feature_columns].fillna(0)`
  - Deduplicate `state` per `userid`: First non-null value
- **Output**: `final_training_dataset.csv`

## 5. Validation
- **Checks**:
  - Alignment: `highest_w_persona` (max `w_X`) matches `target_persona`
  - Match Rate: `% of rows where highest_w_persona == target_persona`
- **Row 3108 Result**:
  - `w_doctor = 0.557143`, `w_vision = 0.457143`
  - `highest_w_persona = "doctor"`, matches `target_persona`

## Key Points
- **Data Integration**: Merges behavioral and plan data for a unified view
- **Weight Design**: Balances plan strength (`ma_X`) with behavior, prioritizes `target_persona`
- **Outcome**: 66-column dataset for persona prediction model
