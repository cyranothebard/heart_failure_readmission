import re

def clean_feature_names(X):
    """
    Clean feature names to be compatible with LightGBM
    """
    clean_columns = {}
    for col in X.columns:
        # Replace any special characters with underscore
        clean_col = re.sub(r'[^\w_]', '_', str(col))
        clean_columns[col] = clean_col
    
    # Rename columns
    X = X.rename(columns=clean_columns)
    return X
