import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek


def create_dataframe_from_numpy_dict(data_dict):
    df = pd.DataFrame({key: val for key, val in data_dict.items() if key != 'time'})
    df['time'] = pd.to_datetime(data_dict['time'])
    return df


def label_cme_occurrences(df, cme_times):
    df['cme_label'] = 0  # 0: none, 1: fast, 2: slow

    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')

    cme_times = [
        pd.Timestamp(t).tz_localize('UTC') if pd.Timestamp(t).tzinfo is None
        else pd.Timestamp(t).tz_convert('UTC')
        for t in cme_times
    ]

    df = df.sort_values('time')

    for t in cme_times:
        short_start = t + timedelta(minutes=10)
        short_end = t + timedelta(minutes=40)
        long_start = t + timedelta(days=3)
        long_end = t + timedelta(days=4)

        df.loc[(df['time'] >= short_start) & (df['time'] <= short_end), 'cme_label'] = 1
        df.loc[(df['time'] >= long_start) & (df['time'] <= long_end), 'cme_label'] = 2

    return df


def train_cme_prediction_model(data_dict, cme_times, test_size=0.2, random_state=42):
    # Step 1: Prepare data
    df = create_dataframe_from_numpy_dict(data_dict)
    df = label_cme_occurrences(df, cme_times)

    feature_cols = [col for col in df.columns if col not in ['time', 'cme_label']]
    X = df[feature_cols].values
    y = df['cme_label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Step 2: Pipeline with SMOTETomek and Random Forest
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTETomek(random_state=random_state)),
        ('clf',RandomForestClassifier(class_weight='balanced_subsample', random_state=random_state))
    ])

    param_grid = {
    'clf__n_estimators': [100],           # Just one reasonable default
    'clf__max_depth': [10, None]          # Shallow vs deep
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Step 3: Evaluation
    y_pred = best_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=['None', 'Fast', 'Slow']
    ))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Weighted F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    # Step 4: Predict on full dataset
    X_scaled = best_model.named_steps['scaler'].transform(X)
    df['cme_pred'] = best_model.named_steps['clf'].predict(X_scaled)

    return df, best_model.named_steps['clf'], dict(zip(feature_cols, best_model.named_steps['clf'].feature_importances_))
