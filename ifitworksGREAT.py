import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer

print("🚀 UPDATED SKLEARN-BASED ML PIPELINE WITH THRESHOLD TUNING FOR GENE CLASSIFICATION")
print("=" * 60)

# ---- CONFIGURATION ---- #
expr_path = r"H:\\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\\sem3\\bio\\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 1_000_000
max_chunks = 5  # Number of chunks to process

# ---- LOAD CGC ---- #
print("📥 Loading CGC genes...")
cgc = pd.read_csv(cgc_path, sep="\t")
driver_genes_set = set(cgc['Gene Symbol'].values)
print(f"Loaded {len(driver_genes_set)} CGC driver genes.")

# ---- COLLECT DATA WITH BALANCE ---- #
print("\n📦 Collecting data with better class balance...")

driver_samples = []
non_driver_samples = []

reader = pd.read_csv(expr_path, sep="\t", chunksize=chunksize)

for i, chunk in enumerate(reader):
    if i >= max_chunks:
        break
    print(f"Processing chunk {i+1}/{max_chunks}...")

    chunk['is_driver'] = chunk['GENE_SYMBOL'].isin(driver_genes_set).astype(int)
    chunk = chunk[['Z_SCORE', 'REGULATION', 'is_driver']].ffill().bfill()

    drivers = chunk[chunk['is_driver'] == 1]
    non_drivers = chunk[chunk['is_driver'] == 0]

    if not drivers.empty:
        driver_samples.append(drivers)

    if len(non_drivers) > 15000:  # downsample non-drivers
        non_drivers = non_drivers.sample(n=15000, random_state=42)
    non_driver_samples.append(non_drivers)

    print(f"  Drivers: {len(drivers)}, Non-drivers: {len(non_drivers)}")

# ---- COMBINE DATA ---- #
print("\n🔄 Combining data...")
all_drivers = pd.concat(driver_samples, ignore_index=True) if driver_samples else pd.DataFrame()
all_non_drivers = pd.concat(non_driver_samples, ignore_index=True)

print(f"Total drivers: {len(all_drivers)}")
print(f"Total non-drivers: {len(all_non_drivers)}")

if all_drivers.empty:
    print("❌ No drivers found in data. Exiting.")
    exit(1)

final_data = pd.concat([all_drivers, all_non_drivers], ignore_index=True)
print(f"Final dataset shape: {final_data.shape}")
print("Class distribution:")
class_counts = final_data['is_driver'].value_counts()
print(class_counts)
print(f"Class imbalance ratio (non-driver:driver): {class_counts[0] / class_counts[1]:.2f}:1")

# ---- PREPARE FEATURES ---- #
print("\n⚡ Preparing features...")

X = final_data[['Z_SCORE', 'REGULATION']]
y = final_data['is_driver'].values

numeric_features = ['Z_SCORE']
categorical_features = ['REGULATION']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

print(f"Feature matrix shape: {X_processed.shape}")
print(f"Class distribution in dataset: {Counter(y)}")

# ---- COMPUTE CLASS WEIGHTS ---- #
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Computed class weights: {class_weight_dict}")

# ---- SPLIT TRAIN-TEST ---- #
print("\n✂️ Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training label distribution: {Counter(y_train)}")

# ---- OVERSAMPLING USING SMOTE ---- #
print("\n🧬 Applying SMOTE oversampling on training data to handle imbalance...")
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Post-SMOTE class distribution: {Counter(y_train_res)}")

# ---- DEFINE MODELS ---- #
models_to_test = {
    "LogReg_Balanced": LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced', n_jobs=-1
    ),
    "RandomForest_Balanced": RandomForestClassifier(
        random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42, n_estimators=100
    ),
    "XGBoost": xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=class_weight_dict[0] / class_weight_dict[1],
        n_jobs=-1
    ),
    "LogReg_SMOTE": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    "RandomForest_SMOTE": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
}

# ---- HYPERPARAMETER TUNING FOR RANDOM FOREST ---- #
print("\n🎯 Hyperparameter tuning RandomForest_Balanced with RandomizedSearchCV...")
param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [10, 20, 40, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}
rfc = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

rscv = RandomizedSearchCV(
    rfc, param_dist, n_iter=20, cv=StratifiedKFold(5),
    scoring='f1', random_state=42, n_jobs=-1, verbose=1
)
rscv.fit(X_train, y_train)

print(f"Best params for RandomForest_Balanced: {rscv.best_params_}")
print(f"Best CV F1 score: {rscv.best_score_:.4f}")

models_to_test["RandomForest_Balanced"] = rscv.best_estimator_

# ---- FUNCTION FOR THRESHOLD TUNING ---- #
def tune_threshold(y_true, y_prob):
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
    return best_threshold, best_f1

# ---- EVALUATE MODELS WITH THRESHOLD TUNING ---- #
print("\n🔍 Evaluating models with threshold tuning...")
print("=" * 80)
results = {}

for name, model in models_to_test.items():
    print(f"\n--- {name} ---")
    try:
        if "SMOTE" in name:
            X_tr, y_tr = X_train_res, y_train_res
        else:
            X_tr, y_tr = X_train, y_train

        print("Training model...")
        model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # scale to [0,1]
        else:
            y_prob = model.predict(X_test)

        threshold, best_f1 = tune_threshold(y_test, y_prob)
        y_pred_thresh = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if hasattr(model, "predict_proba") or hasattr(model, "decision_function") else 0.5

        print(f"Best threshold: {threshold:.2f} with F1-score: {best_f1:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"AUC:       {auc:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_thresh))

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": best_f1, "auc": auc, "threshold": threshold
        }

    except Exception as e:
        print(f"Error training {name}: {e}")

# ---- SELECT BEST MODEL ---- #
print("\n🏆 Selecting best model based on tuned F1-score...")
if results:
    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"Best Model: {best_name} with F1-score: {results[best_name]['f1']:.4f}")

    best_model = models_to_test[best_name]
    best_threshold = results[best_name]['threshold']

    if "SMOTE" in best_name:
        X_final, y_final = X_train_res, y_train_res
    else:
        X_final, y_final = X_train, y_train

    print(f"Training best model ({best_name}) on full training data...")
    best_model.fit(X_final, y_final)

    print(f"Saving best model, preprocessor, and threshold...")
    joblib.dump(best_model, f"best_model_{best_name}.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(best_threshold, "best_threshold.pkl")

    print("✅ All done!")
else:
    print("❌ No models successfully trained.")

