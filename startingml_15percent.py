import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter
import matplotlib.pyplot as plt

print("🚀 SKLEARN ML PIPELINE WITH ADJUSTABLE THRESHOLD FOR 15% CANCER DETECTION")
print("=" * 70)

# ---- CONFIGURATION ---- #
expr_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 1_000_000
max_chunks = 5  # Number of chunks to process
TARGET_CANCER_PERCENTAGE = 0.15  

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
    chunk = chunk[['Z_SCORE', 'REGULATION', 'is_driver']].fillna(0)
    
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

X_numeric = final_data[['Z_SCORE']].values
X_categorical = final_data[['REGULATION']].values
y = final_data['is_driver'].values

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_numeric_scaled = scaler.fit_transform(X_numeric)
X_categorical_encoded = encoder.fit_transform(X_categorical)

X = np.hstack([X_numeric_scaled, X_categorical_encoded])

print(f"Feature matrix shape: {X.shape}")
print(f"Class distribution in dataset: {Counter(y)}")

# ---- COMPUTE CLASS WEIGHTS ---- #
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Computed class weights: {class_weight_dict}")

# ---- SPLIT TRAIN-TEST ---- #
print("\n✂️ Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training label distribution: {Counter(y_train)}")

# ---- OVERSAMPLING HELPER FUNCTION ---- #
def simple_oversample(X, y, target_ratio=0.5):
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    
    target_minority_count = int(majority_count * target_ratio / (1 - target_ratio))
    samples_to_generate = target_minority_count - minority_count
    if samples_to_generate <= 0:
        return X, y
    minority_indices = np.where(y == minority_class)[0]
    new_samples = []
    new_labels = []
    for _ in range(samples_to_generate):
        idx = np.random.choice(minority_indices)
        noise = np.random.normal(0, 0.01, X[idx].shape)
        new_sample = X[idx] + noise
        new_samples.append(new_sample)
        new_labels.append(minority_class)
    X_new = np.vstack([X, np.array(new_samples)])
    y_new = np.hstack([y, np.array(new_labels)])
    return X_new, y_new

# ---- THRESHOLD OPTIMIZATION FUNCTION ---- #
def find_optimal_threshold(model, X_test, y_test, target_percentage=0.15):
    """
    Find optimal threshold to achieve target percentage of positive predictions
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        return 0.5  # Default threshold
    
    # Sort probabilities and find threshold for target percentage
    sorted_probs = np.sort(y_prob)[::-1]  # Descending order
    threshold_idx = int(len(sorted_probs) * target_percentage)
    optimal_threshold = sorted_probs[threshold_idx]
    
    return optimal_threshold

def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model with custom threshold
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
    
    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate percentage of positive predictions
    positive_percentage = np.mean(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
    
    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'positive_percentage': positive_percentage,
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }

# ---- MODELS ---- #
models_to_test = {
    "LogReg_Balanced": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "RandomForest_Balanced": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "LogReg_Oversampled": LogisticRegression(random_state=42, max_iter=1000),
}

# ---- EVALUATE MODELS ---- #
print("\n🔍 Evaluating models with standard and optimized thresholds...")
print("=" * 80)
results = {}
threshold_results = {}

for name, model in models_to_test.items():
    print(f"\n--- {name} ---")
    try:
        if "Oversampled" in name:
            print("Applying oversampling on training data...")
            X_train_res, y_train_res = simple_oversample(X_train, y_train, target_ratio=0.3)
            print(f"Post-oversampling class distribution: {Counter(y_train_res)}")
        else:
            X_train_res, y_train_res = X_train, y_train
        
        print("Training model...")
        model.fit(X_train_res, y_train_res)
        
        # Standard evaluation (default threshold)
        standard_eval = evaluate_with_threshold(model, X_test, y_test, threshold=0.5)
        
        # Find optimal threshold for 15% detection
        optimal_threshold = find_optimal_threshold(model, X_test, y_test, TARGET_CANCER_PERCENTAGE)
        threshold_eval = evaluate_with_threshold(model, X_test, y_test, optimal_threshold)
        
        print(f"\n=== STANDARD THRESHOLD (0.5) ===")
        print(f"Positive predictions: {standard_eval['positive_percentage']:.1%}")
        print(f"Accuracy:  {standard_eval['accuracy']:.4f}")
        print(f"Precision: {standard_eval['precision']:.4f}")
        print(f"Recall:    {standard_eval['recall']:.4f}")
        print(f"F1-score:  {standard_eval['f1']:.4f}")
        print(f"AUC:       {standard_eval['auc']:.4f}")
        
        print(f"\n=== OPTIMIZED THRESHOLD ({optimal_threshold:.4f}) FOR 15% DETECTION ===")
        print(f"Positive predictions: {threshold_eval['positive_percentage']:.1%}")
        print(f"Accuracy:  {threshold_eval['accuracy']:.4f}")
        print(f"Precision: {threshold_eval['precision']:.4f}")
        print(f"Recall:    {threshold_eval['recall']:.4f}")
        print(f"F1-score:  {threshold_eval['f1']:.4f}")
        print(f"AUC:       {threshold_eval['auc']:.4f}")
        
        print("Confusion Matrix (15% threshold):")
        cm = confusion_matrix(y_test, threshold_eval['predictions'])
        print(cm)
        
        results[name] = standard_eval
        threshold_results[name] = threshold_eval
        
    except Exception as e:
        print(f"Error training {name}: {e}")

# ---- SELECT BEST MODEL FOR 15% DETECTION ---- #
print("\n🏆 Selecting best model for 15% cancer detection...")
if threshold_results:
    # Select based on AUC for threshold-optimized models
    best_name = max(threshold_results, key=lambda k: threshold_results[k]['auc'])
    best_result = threshold_results[best_name]
    
    print(f"Best Model: {best_name}")
    print(f"Optimal threshold: {best_result['threshold']:.4f}")
    print(f"Cancer detection rate: {best_result['positive_percentage']:.1%}")
    print(f"AUC: {best_result['auc']:.4f}")
    print(f"F1-score: {best_result['f1']:.4f}")
    
    # Save best model and threshold
    best_model = models_to_test[best_name]
    
    # Retrain on full training data
    if "Oversampled" in best_name:
        X_final, y_final = simple_oversample(X_train, y_train, target_ratio=0.3)
    else:
        X_final, y_final = X_train, y_train
    
    print(f"Retraining best model ({best_name}) on full training data...")
    best_model.fit(X_final, y_final)
    
    print(f"Saving best model and configurations...")
    joblib.dump(best_model, f"best_model_{best_name}_15percent.pkl")
    joblib.dump(scaler, "scaler_15percent.pkl")
    joblib.dump(encoder, "encoder_15percent.pkl")
    joblib.dump(best_result['threshold'], "optimal_threshold_15percent.pkl")
    
    print("✅ Done!")
    print(f"\n🎯 SUMMARY:")
    print(f"- Model trained to detect ~15% of genes as potentially cancerous")
    print(f"- Optimal threshold: {best_result['threshold']:.4f}")
    print(f"- True positive rate (sensitivity): {best_result['recall']:.1%}")
    print(f"- Precision: {best_result['precision']:.1%}")
    print(f"- This captures {int(best_result['recall'] * len(all_drivers))} out of {len(all_drivers)} known cancer genes")
    
else:
    print("❌ No models successfully trained.")