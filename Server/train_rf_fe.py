import pandas as pd
import numpy as np
import os
import glob
import sys
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

def load_feature_data(feature_csv):
    """
    Loads the preprocessed feature CSV (which should have columns like
    emgX_..., ... , pose).
    """
    if not os.path.exists(feature_csv):
        print(f"No feature file found: {feature_csv}")
        return pd.DataFrame()
    
    return pd.read_csv(feature_csv)

def prepare_features_labels(df):
    """
    Identify all columns except 'pose' as features.
    'pose' is the label.
    """
    if 'pose' not in df.columns:
        raise ValueError("DataFrame must contain a 'pose' column for labels.")
    
    # All columns except 'pose' are features
    feature_cols = [c for c in df.columns if c != 'pose']
    X = df[feature_cols].values
    
    y_str = df['pose'].values  # e.g. 'fist', 'pinch', ...
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    
    return X, y, label_encoder, feature_cols

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_rf_fe.py <feature_csv> <model_output_path>")
        sys.exit(1)
    
    feature_csv = sys.argv[1]
    model_path = sys.argv[2]
    
    # 1) Load the extracted features CSV
    df = load_feature_data(feature_csv)
    if df.empty:
        print("No data found in feature CSV. Exiting.")
        sys.exit(1)

    # Check if we have enough samples
    if len(df) < 10:
        print(f"Not enough data samples ({len(df)}). Need at least 10 samples.")
        sys.exit(1)

    # 2) Prepare X, y, label encoder
    X, y, label_encoder, feature_cols = prepare_features_labels(df)

    # Check if we have multiple classes
    unique_classes = len(label_encoder.classes_)
    if unique_classes < 2:
        print(f"Not enough classes ({unique_classes}). Need at least 2 classes for classification.")
        sys.exit(1)

    # 3) Determine estimator size up front so CV + final model stay consistent
    n_estimators = min(100, max(10, len(df) // 2))

    def make_classifier() -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # 4) Run stratified 5-fold CV when each class has enough samples
    class_counts = np.bincount(y)
    min_class_samples = int(class_counts.min()) if class_counts.size else 0
    target_folds = 5
    cv_folds = min(target_folds, min_class_samples)
    accuracy = None
    cv_scores = None
    report_dict = None
    evaluation_note = ""

    if cv_folds >= 2:
        print(f"Running {cv_folds}-fold stratified cross-validation ({len(df)} samples)...")
        cv_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", make_classifier()),
        ])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        for idx, score in enumerate(cv_scores, start=1):
            print(f"Fold {idx}: {score:.3f}")
        print(f"Mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        y_pred_cv = cross_val_predict(cv_pipeline, X, y, cv=cv, n_jobs=-1)
        target_names = list(label_encoder.classes_)
        label_indices = list(range(len(target_names)))
        print("Cross-validated classification report:")
        print(
            classification_report(
                y,
                y_pred_cv,
                labels=label_indices,
                target_names=target_names,
                zero_division=0,
            )
        )
        report_dict = classification_report(
            y,
            y_pred_cv,
            labels=label_indices,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        accuracy = float(cv_scores.mean())
        evaluation_note = f"{cv_folds}-fold cross-validation"
    else:
        print("Not enough samples per class for 5-fold CV; falling back to hold-out split.")
        test_size = min(0.2, 0.5) if len(df) < 50 else 0.2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        scaler_eval = StandardScaler()
        X_train_scaled = scaler_eval.fit_transform(X_train)
        X_test_scaled = scaler_eval.transform(X_test)
        clf_eval = make_classifier()
        clf_eval.fit(X_train_scaled, y_train)
        y_pred = clf_eval.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        label_indices = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [label_encoder.classes_[i] for i in label_indices]
        print("Classification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=label_indices,
                target_names=target_names,
                zero_division=0,
            )
        )
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=label_indices,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        evaluation_note = "hold-out split"

    # 5) Fit final scaler + classifier on the full dataset for export
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)
    clf = make_classifier()
    clf.fit(X_scaled_full, y)

    # 6) Save Model
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_cols': feature_cols,
        'accuracy': accuracy,
        'evaluation_method': evaluation_note,
        'classification_report': report_dict,
        'n_samples': len(df),
        'n_classes': unique_classes
    }
    if cv_scores is not None:
        model_data['cv_fold_scores'] = cv_scores.tolist()
    
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")
    print(f"Model stats: {len(df)} samples, {unique_classes} classes, {accuracy:.1%} accuracy ({evaluation_note})")

if __name__ == '__main__':
    main()
