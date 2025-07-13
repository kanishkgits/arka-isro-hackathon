import joblib
from Scripts.train_cme_classifier import create_dataframe_from_numpy_dict, label_cme_occurrences

def test(data, cme_times):
    
    clf = joblib.load('trained_cme_classifier.pkl')     # RandomForestClassifier
    
    df = create_dataframe_from_numpy_dict(data)
    df = label_cme_occurrences(df, cme_times)

    df['delta_speed'] = df['proton_bulk_speed'].diff().fillna(0)
    df['rolling_thermal_mean'] = df['proton_thermal'].rolling(window=5).mean().bfill()
    df['rolling_density_mean'] = df['proton_density'].rolling(window=5).mean().bfill()

    feature_cols = ['delta_speed', 'rolling_thermal_mean', 'rolling_density_mean']
    X = df[feature_cols].values

    df['cme_proba'] = clf.predict_proba(X)[:, 1]  # probability of CME
    threshold = 0.8
    df['cme_pred'] = (df['cme_proba'] >= threshold).astype(int)
    df['weighted_signal'] = df['cme_proba'].rolling(window=10, center=True).mean().fillna(0)

    return df