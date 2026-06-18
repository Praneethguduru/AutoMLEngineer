def analyze_dataset(df):

    report = {
        'rows' : df.shape[0],
        'columns' : df.shape[1],
        'missing values' : df.isnull().sum(),
        'duplicate rows' : df.duplicated().sum(),
        'statistical summary' : df.describe(),
        'issues' : []
    }
    return report