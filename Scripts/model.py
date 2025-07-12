import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
from imblearn.under_sampling import RandomUnderSampler

def create_dataframe_from_numpy_dict(data_dict):
    df = pd.DataFrame({key: val for key, val in data_dict.items() if key != 'time'})
    df['time'] = pd.to_datetime(data_dict['time'])
    return df


def label_cme_occurrences(df, cme_times):
    df['cme_label'] = 0  # 0: no CME, 1: CME (within ±30 min)

    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')

    cme_times = [
        pd.Timestamp(t).tz_localize('UTC') if pd.Timestamp(t).tzinfo is None
        else pd.Timestamp(t).tz_convert('UTC')
        for t in cme_times
    ]

    df = df.sort_values('time')

    for t in cme_times:
        window_start = t - timedelta(minutes=30)
        window_end = t + timedelta(minutes=30)
        df.loc[(df['time'] >= window_start) & (df['time'] <= window_end), 'cme_label'] = 1

    return df


def train_cme_prediction_model(data_dict, cme_times, test_size=0.2, random_state=42):
    # Step 1: Prepare data
    df = create_dataframe_from_numpy_dict(data_dict)
    df = label_cme_occurrences(df, cme_times)

    df['delta_density'] = df['proton_density'].diff().fillna(0)
    df['rolling_speed_mean'] = df['proton_bulk_speed'].rolling(window=15).mean().bfill()


    feature_cols = ['delta_density', 'rolling_speed_mean']
    X = df[feature_cols].values
    y = df['cme_label'].values

    # Step 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Step 3: Under-sampling
    rus = RandomUnderSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Step 4: Train classifier (use class_weight to focus more on CME)
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=10,
        class_weight={0: 1, 1: 3},  # CME weighted more
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train_resampled, y_train_resampled)

    # Step 5: Evaluation
    y_proba = clf.predict_proba(X_test)[:, 1]  # CME probability
    threshold = 0.75
    y_pred = (y_proba >= threshold).astype(int)

    # Step 6: Predict for full data
    df['cme_pred'] = clf.predict(X)
    df['cme_proba'] = clf.predict_proba(X)[:, 1]  # probability of CME
    df['weighted_signal'] = df['cme_proba'].rolling(window=10, center=True).mean().fillna(0)

    return df, clf, dict(zip(feature_cols, clf.feature_importances_))