import pandas as pd
import numpy as np

def detect_target_column(df):
    """
    Heuristically detects the classification target column from a DataFrame.
    Returns the name of the detected target column.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    scores = {}
    n_rows = len(df)
    
    # Common target keyword sets
    primary_keywords = {'target', 'label', 'class', 'churn', 'y', 'default', 'output', 'decision', 'survived', 'survive'}
    secondary_keywords = {'status', 'response', 'outcome', 'category', 'result'}
    
    for idx, col in enumerate(df.columns):
        col_lower = str(col).lower()
        score = 0.0
        
        # 1. Cardinality check
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            scores[col] = -float('inf')
            continue
            
        # Do not rule out as ID if it's a float column (regression targets are unique floats)
        if n_unique == n_rows and not pd.api.types.is_float_dtype(df[col]):
            scores[col] = -float('inf')
            continue
            
        # Classifiers typically predict binary or multi-class (e.g. 2 to 15 classes)
        if 2 <= n_unique <= 15:
            score += 10.0
        elif 16 <= n_unique <= 50:
            score += 2.0
        else:
            # High cardinality might indicate continuous values or high-cardinality categorical
            score -= 5.0
            
        # 2. Keyword check
        if any(kw == col_lower for kw in primary_keywords):
            score += 20.0
        elif any(kw in col_lower for kw in primary_keywords):
            score += 10.0
        elif any(kw in col_lower for kw in secondary_keywords):
            score += 5.0
            
        # 3. Position check (target is often the last column)
        if idx == len(df.columns) - 1:
            score += 3.0
            
        # 4. Data type check
        if df[col].dtype == 'bool' or df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            # Categorical/bool columns are common classification targets
            score += 2.0
            
        scores[col] = score
        
    if not scores:
        raise ValueError("No columns available to detect target.")
        
    # Get column with the highest score
    best_col = max(scores, key=scores.get)
    return best_col
