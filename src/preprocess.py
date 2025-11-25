import pandas as pd

def map_labels(df):
    """
    Maps the original trend_label into trending / likely / low buckets.
    """
    mapping = {
        'rising': 'trending',
        'seasonal': 'likely',
        'stable': 'likely',
        'declining': 'low'
    }
    df['trend_bucket'] = df['trend_label'].map(mapping)
    return df


def clean_features(df, feature_cols):
    """
    Removes rows with missing labels and fills ANY NaNs in features with 0.
    This protects the model from crashing.
    """
    # Ensure trend_bucket exists
    df = df.dropna(subset=['trend_bucket'])

    # Fill all missing numeric values with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


def get_feature_matrix(df, feature_cols):
    """
    Returns X, y ready to pass into train_test_split.
    """
    X = df[feature_cols]
    y = df['trend_bucket']
    return X, y
