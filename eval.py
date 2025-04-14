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

    # Compute or ensure quality_level
    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]
    if 'quality_level' not in training_df.columns:
        print("Warning: 'quality_level' not found in training_df. Computing it.")
        training_df['quality_level'] = training_df.apply(
            lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
        )
    else:
        null_quality = training_df['quality_level'].isna().sum()
        if null_quality > 0:
            print(f"Warning: {null_quality} rows with null 'quality_level'. Recomputing.")
            training_df['quality_level'] = training_df.apply(
                lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
            )

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

    # Ensure all raw_plan_features and csnp_type exist
    for col in raw_plan_features + ['csnp_type']:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    # Normalize continuous features (NEW)
    continuous_features = [
        'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
        'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
        'total_session_time', 'num_pages_viewed', 'num_plans_selected', 'num_plans_compared',
        'dce_click_count', 'pro_click_count'
    ]
    scaler = StandardScaler()
    training_df[continuous_features] = scaler.fit_transform(training_df[continuous_features].fillna(0))

    additional_features = []
    training_df['csnp_interaction'] = training_df['csnp'] * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 3.0
    additional_features.append('csnp_interaction')

    training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 3.0
    additional_features.append('csnp_signal_strength')

    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * training_df['ma_dental_benefit'] * 2.0
    additional_features.append('dental_interaction')

    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * training_df['ma_vision'] * 2.0
    additional_features.append('vision_interaction')

    training_df['csnp_drug_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.5 - training_df['ma_drug_coverage'] * (
            training_df.get('query_drug', 0).fillna(0) + training_df.get('filter_drug', 0).fillna(0) + 
            training_df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 3.0
    additional_features.append('csnp_drug_interaction')

    training_df['csnp_doctor_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0)
        ) * 2.0 - training_df['ma_provider_network'] * (
            training_df.get('query_provider', 0).fillna(0) + training_df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 2.0
    additional_features.append('csnp_doctor_interaction')

    # New interaction features (NEW)
    training_df['dsnp_interaction'] = training_df['dsnp'] * (
        training_df.get('query_dsnp', 0).fillna(0) + training_df.get('filter_dsnp', 0).fillna(0) + 
        training_df.get('time_dsnp_pages', 0).fillna(0) + training_df.get('accordion_dsnp', 0).fillna(0)
    ) * 3.0
    additional_features.append('dsnp_interaction')

    training_df['vision_signal_strength'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0) + 
        training_df.get('accordion_vision', 0).fillna(0) + training_df.get('time_vision_pages', 0).fillna(0)
    ).clip(upper=5) * 2.0
    additional_features.append('vision_signal_strength')

    # Debug: Check csnp features
    high_quality_csnp = training_df[(training_df['quality_level'] == 'High') & (training_df['persona'] == 'csnp')]
    print(f"Eval: High-quality csnp samples: {len(high_quality_csnp)}")
    print(f"Eval: Non-zero csnp_interaction: {sum(high_quality_csnp['csnp_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_drug_interaction: {sum(high_quality_csnp['csnp_drug_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_doctor_interaction: {sum(high_quality_csnp['csnp_doctor_interaction'] > 0)}")

    feature_columns = all_behavioral_features + raw_plan_features + additional_features

    print(f"Feature columns expected: {feature_columns}")

    missing_features = [col for col in feature_columns if col not in training_df.columns]
    if missing_features:
        print(f"Warning: Missing features in training_df: {missing_features}")
        for col in missing_features:
            training_df[col] = 0

    print(f"Columns in training_df after filling: {training_df.columns.tolist()}")

    # Ensure metadata includes quality_level
    metadata_columns = ['userid', 'zip', 'plan_id', 'persona', 'quality_level'] + feature_columns
    missing_metadata_cols = [col for col in metadata_columns if col not in training_df.columns]
    if missing_metadata_cols:
        print(f"Warning: Missing metadata columns: {missing_metadata_cols}")
        for col in missing_metadata_cols:
            training_df[col] = 0

    metadata = training_df[metadata_columns]

    # Filter out fitness and hearing
    valid_mask = (
        training_df['persona'].notna() & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    training_df = training_df[valid_mask]
    metadata = metadata[valid_mask]
    print(f"Rows after filtering fitness/hearing: {len(training_df)}")

    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona'] if 'persona' in training_df.columns else None

    if y_true is not None:
        print(f"Unique personas before filtering: {y_true.unique().tolist()}")

    return X, y_true, metadata
