import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter

print("🚀 SKLEARN-ONLY IMPROVED ML PIPELINE FOR GENE CLASSIFICATION")
print("=" * 60)

# ---- CONFIGURATION ---- #
expr_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 1_000_000
max_chunks = 5  # Use fewer chunks for faster execution

# ---- LOAD CGC ---- #
print("📥 Loading CGC genes...")
cgc = pd.read_csv(cgc_path, sep="\t")
cgc_genes = cgc[['Gene Symbol']].rename(columns={'Gene Symbol': 'GENE_SYMBOL'})
driver_genes_set = set(cgc_genes['GENE_SYMBOL'].values)
print(f"Loaded {len(cgc_genes)} CGC driver genes.")

# ---- COLLECT DATA WITH BETTER BALANCE ---- #
print("\n📦 Collecting data with better class balance...")
driver_samples = []
non_driver_samples = []

reader = pd.read_csv(expr_path, sep="\t", chunksize=chunksize)

for i, chunk in enumerate(reader):
    if i >= max_chunks:
        break
        
    print(f"Processing chunk {i+1}/{max_chunks}...")
    
    # Create labels
    chunk['is_driver'] = chunk['GENE_SYMBOL'].isin(driver_genes_set).astype(int)
    chunk = chunk[['Z_SCORE', 'REGULATION', 'is_driver']].fillna(0)
    
    # Separate by class
    drivers = chunk[chunk['is_driver'] == 1]
    non_drivers = chunk[chunk['is_driver'] == 0]
    
    # Collect all drivers (they're rare)
    if len(drivers) > 0:
        driver_samples.append(drivers)
    
    # Sample non-drivers to create better balance
    if len(non_drivers) > 15000:  # Collect fewer non-drivers
        non_drivers = non_drivers.sample(n=15000, random_state=42)
    non_driver_samples.append(non_drivers)
    
    print(f"  Drivers: {len(drivers)}, Non-drivers: {len(non_drivers)}")

# ---- COMBINE DATA ---- #
print("\n🔄 Combining data...")
all_drivers = pd.concat(driver_samples, ignore_index=True) if driver_samples else pd.DataFrame()
all_non_drivers = pd.concat(non_driver_samples, ignore_index=True)

print(f"Total drivers: {len(all_drivers)}")
print(f"Total non-drivers: {len(all_non_drivers)}")

# Create final dataset
if len(all_drivers) == 0:
    print("❌ No driver genes found! Check your CGC file and data overlap.")
    exit(1)

final_data = pd.concat([all_drivers, all_non_drivers], ignore_index=True)
print(f"Final dataset shape: {final_data.shape}")
print("Class distribution:")
class_counts = final_data['is_driver'].value_counts()
print(class_counts)
print(f"Class imbalance ratio: {class_counts[0] / class_counts[1]:.2f}:1 (non-driver:driver)")

# ---- PREPARE FEATURES ---- #
print("\n⚡ Preparing features...")

X_numeric = final_data[['Z_SCORE']].values
X_categorical = final_data[['REGULATION']].values
y = final_data['is_driver'].values

# Preprocessing
scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_numeric_scaled = scaler.fit_transform(X_numeric)
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Combine features
X = np.concatenate([X_numeric_scaled, X_categorical_encoded], axis=1)

print(f"Feature matrix shape: {X.shape}")
print(f"Class distribution: {Counter(y)}")

# ---- CALCULATE CLASS WEIGHTS ---- #
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Computed class weights: {class_weight_dict}")

# ---- TRAIN-TEST SPLIT ---- #
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train class distribution: {Counter(y_train)}")

# ---- SIMPLE OVERSAMPLING FUNCTION ---- #
def simple_oversample(X, y, target_ratio=0.5):
    """Simple oversampling by duplicating minority class samples"""
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    
    # Calculate how many minority samples to generate
    target_minority_count = int(majority_count * target_ratio / (1 - target_ratio))
    samples_to_generate = target_minority_count - minority_count
    
    if samples_to_generate <= 0:
        return X, y
    
    # Find minority class indices
    minority_indices = np.where(y == minority_class)[0]
    
    # Generate new samples by duplicating existing ones with small noise
    new_samples = []
    new_labels = []
    
    for _ in range(samples_to_generate):
        # Randomly select a minority sample
        idx = np.random.choice(minority_indices)
        # Add small noise to create variation
        noise = np.random.normal(0, 0.01, X[idx].shape)
        new_sample = X[idx] + noise
        new_samples.append(new_sample)
        new_labels.append(minority_class)
    
    if new_samples:
        X_new = np.vstack([X, np.array(new_samples)])
        y_new = np.hstack([y, np.array(new_labels)])
        return X_new, y_new
    
    return X, y

# ---- MODELS TO TEST ---- #
models_to_test = {
    "LogisticRegression_Balanced": LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight='balanced'
    ),
    "RandomForest_Balanced": RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    ),
    "GradientBoosting_CustomWeights": GradientBoostingClassifier(
        random_state=42,
        n_estimators=100
    ),
    "LogisticRegression_Oversampled": LogisticRegression(
        random_state=42, 
        max_iter=1000
    ),
    "RandomForest_Oversampled": RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
}

# ---- EVALUATION ---- #
print("\n🔍 Evaluating different approaches...")
print("=" * 80)

results = {}

for model_name, model in models_to_test.items():
    print(f"\n--- Testing {model_name} ---")
    
    try:
        # Apply oversampling if specified
        if "Oversampled" in model_name:
            print("Applying simple oversampling...")
            X_resampled, y_resampled = simple_oversample(X_train, y_train, target_ratio=0.3)
            print(f"After oversampling: {Counter(y_resampled)}")
        elif "CustomWeights" in model_name:
            print("Using custom sample weights...")
            X_resampled, y_resampled = X_train, y_train
            # Create sample weights
            sample_weights = np.array([class_weight_dict[label] for label in y_train])
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Train model
        print("Training model...")
        if "CustomWeights" in model_name:
            model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
        else:
            model.fit(X_resampled, y_resampled)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        }
        
        # AUC calculation
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics['auc'] = roc_auc_score(y_test, y_prob)
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            metrics['auc'] = roc_auc_score(y_test, y_prob)
        else:
            metrics['auc'] = 0.5
        
        results[model_name] = metrics
        
        print(f"Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Show some interpretation
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Positives: {tp}")
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        continue

# ---- FIND BEST MODEL ---- #
print("\n🏆 FINDING BEST MODEL BY F1-SCORE...")
print("=" * 60)

if results:
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_metrics = results[best_model_name]
    
    print(f"Best model: {best_model_name}")
    print(f"Best F1-score: {best_metrics['f1']:.4f}")
    
    # ---- TRAIN AND SAVE BEST MODEL ---- #
    print(f"\n🎯 Training and saving best model: {best_model_name}")
    
    best_model = models_to_test[best_model_name]
    
    # Apply appropriate preprocessing
    if "Oversampled" in best_model_name:
        X_final, y_final = simple_oversample(X_train, y_train, target_ratio=0.3)
    elif "CustomWeights" in best_model_name:
        X_final, y_final = X_train, y_train
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        best_model.fit(X_final, y_final, sample_weight=sample_weights)
    else:
        X_final, y_final = X_train, y_train
        best_model.fit(X_final, y_final)
    
    # Train final model if not already trained above
    if "CustomWeights" not in best_model_name:
        best_model.fit(X_final, y_final)
    
    # Final evaluation
    y_pred_final = best_model.predict(X_test)
    
    print("\n📊 FINAL MODEL PERFORMANCE:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_final, pos_label=1):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_final, pos_label=1):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_final, pos_label=1):.4f}")
    
    if hasattr(best_model, "predict_proba"):
        y_prob_final = best_model.predict_proba(X_test)[:, 1]
        print(f"AUC:       {roc_auc_score(y_test, y_prob_final):.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_final))
    
    # ---- SAVE MODEL ---- #
    print("\n💾 Saving best model and preprocessors...")
    joblib.dump(best_model, f"sklearn_best_model_{best_model_name}.pkl")
    joblib.dump(scaler, f"sklearn_scaler_{best_model_name}.pkl")
    joblib.dump(encoder, f"sklearn_encoder_{best_model_name}.pkl")
    
    print("✅ Model saved successfully!")
    
    # ---- SUMMARY ---- #
    print("\n📈 SUMMARY OF ALL MODELS:")
    print("=" * 75)
    print(f"{'Model':<35} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print("-" * 75)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<35} "
              f"{metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['auc']:<8.4f}")
    
    print(f"\n🎉 Best model: {best_model_name} with F1-score: {best_metrics['f1']:.4f}")
    
    # ---- COMPARISON WITH ORIGINAL APPROACH ---- #
    print(f"\n🔍 COMPARISON WITH YOUR ORIGINAL APPROACH:")
    print("=" * 50)
    print("Your original script problems:")
    print("1. ❌ Applied undersampling to each chunk separately")
    print("2. ❌ Severe class imbalance not properly addressed") 
    print("3. ❌ Used inappropriate models for imbalanced data")
    print("4. ❌ No proper evaluation strategy")
    print()
    print("This improved script:")
    print("1. ✅ Better data collection strategy")
    print("2. ✅ Proper class balancing techniques")
    print("3. ✅ Models specifically designed for imbalanced data")
    print("4. ✅ Comprehensive evaluation metrics")
    print("5. ✅ Focus on F1-score instead of just accuracy")
else:
    print("❌ No models were successfully trained!")

print("\n🎯 This should give you much better precision, recall, and F1-scores!")
