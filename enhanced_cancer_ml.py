import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("🚀 ENHANCED CANCER GENE CLASSIFICATION - TARGET: 90%+ ACCURACY")
print("=" * 65)

# ---- CONFIGURATION ---- #
expr_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 1_000_000
max_chunks = 8  # Increased for more data
TARGET_ACCURACY = 0.90

# ---- LOAD CGC ---- #
print("📥 Loading CGC genes...")
cgc = pd.read_csv(cgc_path, sep="\t")
driver_genes_set = set(cgc['Gene Symbol'].values)
print(f"Loaded {len(driver_genes_set)} CGC driver genes.")

# ---- ENHANCED DATA COLLECTION ---- #
print("\n📦 Enhanced data collection with better sampling...")

driver_samples = []
non_driver_samples = []

reader = pd.read_csv(expr_path, sep="\t", chunksize=chunksize)

for i, chunk in enumerate(reader):
    if i >= max_chunks:
        break
    print(f"Processing chunk {i+1}/{max_chunks}...")
    
    chunk['is_driver'] = chunk['GENE_SYMBOL'].isin(driver_genes_set).astype(int)
    chunk = chunk[['Z_SCORE', 'REGULATION', 'is_driver']].fillna(0)
    
    # Quality filtering for better signal
    chunk = chunk[
        (np.abs(chunk['Z_SCORE']) < 15) &  # Remove extreme outliers
        (chunk['Z_SCORE'].notna()) &
        (chunk['REGULATION'].isin(['normal', 'over', 'under']))
    ]
    
    drivers = chunk[chunk['is_driver'] == 1]
    non_drivers = chunk[chunk['is_driver'] == 0]
    
    if not drivers.empty:
        driver_samples.append(drivers)
    
    # Balanced sampling for non-drivers
    if len(non_drivers) > 20000:  # Increased sample size
        non_drivers = non_drivers.sample(n=20000, random_state=42)
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

# Balance the dataset for better performance
min_size = min(len(all_drivers), len(all_non_drivers))
if len(all_drivers) > min_size * 1.5:
    all_drivers = all_drivers.sample(n=int(min_size * 1.2), random_state=42)
if len(all_non_drivers) > min_size * 1.5:
    all_non_drivers = all_non_drivers.sample(n=int(min_size * 1.2), random_state=42)

final_data = pd.concat([all_drivers, all_non_drivers], ignore_index=True)
print(f"Balanced dataset shape: {final_data.shape}")
class_counts = final_data['is_driver'].value_counts()
print(f"Final class distribution: {dict(class_counts)}")

# ---- ADVANCED FEATURE ENGINEERING ---- #
print("\n⚡ Advanced feature engineering...")

def create_enhanced_features(data):
    """Create comprehensive feature set for cancer gene detection"""
    
    # Base features
    z_scores = data['Z_SCORE'].values
    regulation_values = data['REGULATION'].values
    
    # Mathematical transformations
    features = []
    feature_names = []
    
    # 1. Basic Z-score features
    features.extend([
        z_scores,                                    # Original Z-score
        np.abs(z_scores),                           # Absolute Z-score
        z_scores ** 2,                              # Squared Z-score
        z_scores ** 3,                              # Cubed Z-score
        np.sign(z_scores),                          # Sign of Z-score
    ])
    feature_names.extend(['z_score', 'abs_z', 'z_squared', 'z_cubed', 'z_sign'])
    
    # 2. Statistical transformations
    features.extend([
        np.log1p(np.abs(z_scores)),                 # Log(1+|z|)
        np.sqrt(np.abs(z_scores)),                  # Square root
        np.tanh(z_scores),                          # Tanh transformation
        1 / (1 + np.exp(-z_scores)),                # Sigmoid
        np.exp(-0.5 * z_scores**2),                 # Gaussian kernel
    ])
    feature_names.extend(['log_abs_z', 'sqrt_abs_z', 'tanh_z', 'sigmoid_z', 'gaussian_z'])
    
    # 3. Threshold-based features
    features.extend([
        (np.abs(z_scores) > 1).astype(int),         # |Z| > 1
        (np.abs(z_scores) > 2).astype(int),         # |Z| > 2
        (np.abs(z_scores) > 3).astype(int),         # |Z| > 3
        (z_scores > 2).astype(int),                 # Z > 2 (over-expression)
        (z_scores < -2).astype(int),                # Z < -2 (under-expression)
    ])
    feature_names.extend(['abs_z_gt1', 'abs_z_gt2', 'abs_z_gt3', 'z_gt2', 'z_lt_neg2'])
    
    # 4. Regulation encoding
    regulation_map = {'normal': 0, 'over': 1, 'under': 2}
    regulation_encoded = np.array([regulation_map.get(reg, 0) for reg in regulation_values])
    features.append(regulation_encoded)
    feature_names.append('regulation_encoded')
    
    # 5. One-hot encoding for regulation
    for reg_type in ['normal', 'over', 'under']:
        features.append((regulation_values == reg_type).astype(int))
        feature_names.append(f'reg_{reg_type}')
    
    # 6. Target encoding for regulation
    target_means = {}
    for reg in ['normal', 'over', 'under']:
        mask = regulation_values == reg
        if mask.sum() > 0:
            target_means[reg] = data.loc[mask, 'is_driver'].mean()
        else:
            target_means[reg] = data['is_driver'].mean()
    
    regulation_target = np.array([target_means.get(reg, 0) for reg in regulation_values])
    features.append(regulation_target)
    feature_names.append('regulation_target_encoding')
    
    # 7. Interaction features
    interaction_features = []
    interaction_names = []
    
    # Z-score × regulation interactions
    for i, reg_type in enumerate(['normal', 'over', 'under']):
        reg_mask = (regulation_values == reg_type).astype(float)
        interaction_features.extend([
            z_scores * reg_mask,                    # Z × regulation
            np.abs(z_scores) * reg_mask,           # |Z| × regulation
            (z_scores ** 2) * reg_mask,            # Z² × regulation
        ])
        interaction_names.extend([f'z_x_{reg_type}', f'abs_z_x_{reg_type}', f'z2_x_{reg_type}'])
    
    features.extend(interaction_features)
    feature_names.extend(interaction_names)
    
    # Combine all features
    X = np.column_stack(features)
    
    print(f"Created {X.shape[1]} engineered features")
    return X, feature_names

# Create enhanced features
X, feature_names = create_enhanced_features(final_data)
y = final_data['is_driver'].values

print(f"Final feature matrix shape: {X.shape}")
print(f"Class distribution: {Counter(y)}")

# ---- FEATURE SCALING ---- #
print("\n🔧 Feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- FEATURE SELECTION ---- #
print("\n🎯 Feature selection...")
selector = SelectKBest(f_classif, k=25)  # Select top 25 features
X_selected = selector.fit_transform(X_scaled, y)
selected_features = np.array(feature_names)[selector.get_support()]
print(f"Selected {X_selected.shape[1]} most informative features")
print(f"Top features: {list(selected_features[:5])}")

# ---- TRAIN-TEST SPLIT ---- #
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ---- ADVANCED MODELS ---- #
print("\n🤖 Training advanced models...")

models = {
    'LogisticRegression': LogisticRegression(
        random_state=42, max_iter=2000, class_weight='balanced', C=1.0
    ),
    'RandomForest': RandomForestClassifier(
        random_state=42, n_estimators=200, max_depth=15, 
        class_weight='balanced', min_samples_split=5
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42, n_estimators=200, max_depth=8,
        learning_rate=0.1, subsample=0.8
    ),
    'SVM': SVC(
        random_state=42, kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True
    ),
    'NeuralNetwork': MLPClassifier(
        random_state=42, hidden_layer_sizes=(100, 50),
        max_iter=1000, alpha=0.001, learning_rate='adaptive'
    )
}

# ---- MODEL EVALUATION ---- #
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Train model
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 
        'f1': f1, 'auc': auc, 'model': model
    }

# ---- ENSEMBLE MODEL ---- #
print("\n🎪 Creating ensemble model...")

# Select top 3 models based on accuracy
top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
print(f"Top models for ensemble: {[name for name, _ in top_models]}")

ensemble_models = [(name, results[name]['model']) for name, _ in top_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')

print("Training ensemble...")
ensemble.fit(X_train, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test)
y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]

acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
prec_ensemble = precision_score(y_test, y_pred_ensemble, zero_division=0)
rec_ensemble = recall_score(y_test, y_pred_ensemble, zero_division=0)
f1_ensemble = f1_score(y_test, y_pred_ensemble, zero_division=0)
auc_ensemble = roc_auc_score(y_test, y_prob_ensemble)

print(f"\n🏆 ENSEMBLE RESULTS:")
print(f"Accuracy:  {acc_ensemble:.4f}")
print(f"Precision: {prec_ensemble:.4f}")
print(f"Recall:    {rec_ensemble:.4f}")
print(f"F1-score:  {f1_ensemble:.4f}")
print(f"AUC:       {auc_ensemble:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(cm)

# ---- SELECT BEST MODEL ---- #
print("\n🎯 Selecting best model...")

# Add ensemble to results
results['Ensemble'] = {
    'accuracy': acc_ensemble, 'precision': prec_ensemble, 
    'recall': rec_ensemble, 'f1': f1_ensemble, 'auc': auc_ensemble,
    'model': ensemble
}

# Find best model by accuracy
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
best_acc = results[best_name]['accuracy']

print(f"Best Model: {best_name}")
print(f"Best Accuracy: {best_acc:.4f}")

if best_acc >= TARGET_ACCURACY:
    print(f"🎉 TARGET ACHIEVED! Accuracy {best_acc:.4f} >= {TARGET_ACCURACY}")
else:
    print(f"⚠️ Target not reached. Best: {best_acc:.4f}, Target: {TARGET_ACCURACY}")

# ---- SAVE MODELS ---- #
print("\n💾 Saving best model and components...")

# Retrain on full dataset
print("Retraining best model on full dataset...")
best_model.fit(X_selected, y)

# Save everything
joblib.dump(best_model, f"enhanced_model_{best_name}.pkl")
joblib.dump(scaler, "enhanced_scaler.pkl")
joblib.dump(selector, "enhanced_feature_selector.pkl")
joblib.dump(feature_names, "enhanced_feature_names.pkl")

# Save results summary
results_summary = {
    'best_model': best_name,
    'best_accuracy': best_acc,
    'all_results': {name: {k: v for k, v in res.items() if k != 'model'} 
                    for name, res in results.items()},
    'feature_names': list(selected_features),
    'target_achieved': best_acc >= TARGET_ACCURACY
}

joblib.dump(results_summary, "enhanced_results_summary.pkl")

print("✅ Enhanced cancer gene classification complete!")
print(f"\n📊 FINAL SUMMARY:")
print(f"🏆 Best Model: {best_name}")
print(f"🎯 Accuracy: {best_acc:.1%}")
print(f"🔬 Features: {X_selected.shape[1]} selected features")
print(f"📈 F1-Score: {results[best_name]['f1']:.3f}")
print(f"🎪 Models Trained: {len(results)}")

if best_acc >= TARGET_ACCURACY:
    print("🎉 SUCCESS: 90%+ accuracy target achieved!")
else:
    print(f"📈 Progress: {best_acc:.1%} accuracy (target: {TARGET_ACCURACY:.1%})")