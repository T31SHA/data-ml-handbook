def add_feats(X):
    """Row-wise domain features (no statistics learned => no leakage)."""
    X = X.copy()
    X["rooms_per_person"] = X.AveRooms / X.AveOccup
    X["bedroom_ratio"] = X.AveBedrms / X.AveRooms
    return X
