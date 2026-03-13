import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIGURATION ---- #
expr_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 1_000_000  # Smaller chunks for better memory management
max_chunks = 10        # Limit chunks for faster experimentation

print("🚀 IMPROVED ML PIPELINE FOR GENE CLASSIFICATION")
print("=" * 60)

# ---- LOAD CGC ---- #
print("📥 Loading CGC genes...")
cgc = pd.read_csv(cgc_path, sep="\t")
cgc_genes = cgc[['Gene Symbol']].rename(columns={'Gene Symbol': 'GENE_SYMBOL'})
driver_genes_set = set(cgc_genes['GENE_SYMBOL'].values)
print(f"Loaded {len(cgc_genes)} CGC driver genes.")

# ---- COLLECT BALANCED DATASET ---- #
print("\n📦 Collecting balanced dataset...")
all_chunks = []
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
    
    driver_samples.append(drivers)
    # Sample non-drivers to reduce memory usage
    if len(non_drivers) > 50000:
        non_drivers = non_drivers.sample(n=50000, random_state=42)
    non_driver_samples.append(non_drivers)
    
    print(f"  Drivers: {len(drivers)}, Non-drivers: {len(non_drivers)}")

# Combine all samples
print("\n🔄 Combining samples...")
all_drivers = pd.concat(driver_samples, ignore_index=True)
all_non_drivers = pd.concat(non_driver_samples, ignore_index=True)

print(f"Total drivers: {len(all_drivers)}")
print(f"Total non-drivers: {len(all_non_drivers)}")

# Create final dataset
final_data = pd.concat([all_drivers, all_non_drivers], ignore_index=True)
print(f"Final dataset shape: {final_data.shape}")
print("Class distribution:")
print(final_data['is_driver'].value_counts())

# ---- FEATURE ENGINEERING ---- #
print("\n⚡ Feature engineering...")

# Prepare features
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

# ---- TRAIN-TEST SPLIT ---- #
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train class distribution: {Counter(y_train)}")
print(f"Test class distribution: {Counter(y_test)}")

# ---- SAMPLING STRATEGIES ---- #
print("\n🎯 Testing different sampling strategies...")

sampling_strategies = {
    "Original": None,
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "SMOTE+ENN": SMOTEENN(random_state=42),
    "SMOTE+Tomek": SMOTETomek(random_state=42),
}

# ---- MODELS ---- #
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(n_estimators=100, random_state=42),
    "BalancedBagging": BalancedBaggingClassifier(random_state=42),
    "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
}

# ---- EVALUATION ---- #
print("\n🔍 Evaluating models with different sampling strategies...")
results = {}

for sampling_name, sampler in sampling_strategies.items():
    print(f"\n--- {sampling_name} ---")
    
    # Apply sampling if specified
    if sampler is None:
        X_resampled, y_resampled = X_train, y_train
    else:
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print(f"Resampled distribution: {Counter(y_resampled)}")
        except Exception as e:
            print(f"Sampling failed: {e}")
            continue
    
    results[sampling_name] = {}
    
    for model_name, model in models.items():
        try:
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_resampled, y_resampled)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities for AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
            }
            
            results[sampling_name][model_name] = metrics
            
            print(f"  Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
                  f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            continue

# ---- FIND BEST MODEL ---- #
print("\n🏆 FINDING BEST MODEL...")
print("=" * 60)

best_f1 = 0
best_config = None
best_model = None

for sampling_name, sampling_results in results.items():
    for model_name, metrics in sampling_results.items():
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_config = (sampling_name, model_name)

print(f"Best configuration: {best_config[0]} + {best_config[1]}")
print(f"Best F1-score: {best_f1:.4f}")

# ---- TRAIN AND EVALUATE BEST MODEL ---- #
if best_config:
    print(f"\n🎯 Training final model: {best_config[0]} + {best_config[1]}")
    
    # Get best sampling strategy and model
    best_sampler = sampling_strategies[best_config[0]]
    best_model_class = models[best_config[1]]
    
    # Resample training data
    if best_sampler is None:
        X_final, y_final = X_train, y_train
    else:
        X_final, y_final = best_sampler.fit_resample(X_train, y_train)
    
    # Train final model
    final_model = best_model_class
    final_model.fit(X_final, y_final)
    
    # Final predictions
    y_pred_final = final_model.predict(X_test)
    
    print("\n📊 FINAL RESULTS:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_final, pos_label=1):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_final, pos_label=1):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_final, pos_label=1):.4f}")
    
    if hasattr(final_model, "predict_proba"):
        y_prob_final = final_model.predict_proba(X_test)[:, 1]
        print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob_final):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_final))
    
    # ---- SAVE MODELS ---- #
    print("\n💾 Saving final model and preprocessors...")
    joblib.dump(final_model, f"best_model_{best_config[1]}_{best_config[0]}.pkl")
    joblib.dump(scaler, f"scaler_for_{best_config[1]}.pkl")
    joblib.dump(encoder, f"encoder_for_{best_config[1]}.pkl")
    
    print("✅ Model pipeline completed successfully!")
    print(f"\nBest model saved as: best_model_{best_config[1]}_{best_config[0]}.pkl")

# ---- RESULTS SUMMARY ---- #
print("\n📈 COMPREHENSIVE RESULTS SUMMARY:")
print("=" * 80)
print(f"{'Sampling':<15} {'Model':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
print("-" * 80)

for sampling_name, sampling_results in results.items():
    for model_name, metrics in sampling_results.items():
        print(f"{sampling_name:<15} {model_name:<20} "
              f"{metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['auc']:<8.4f}")

print("\n🎉 Analysis complete! Check the results above to find the best performing model.")
