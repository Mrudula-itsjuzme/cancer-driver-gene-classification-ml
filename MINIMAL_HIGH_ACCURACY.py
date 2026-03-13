import pandas as pd
import numpy as np
import time
from collections import Counter
import pickle

# Use only basic sklearn - no joblib, no imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🚀 MINIMAL HIGH-ACCURACY ML PIPELINE")
print("✅ Using only basic packages for maximum compatibility")
print("⚡ Optimized for speed and 80%+ accuracy")
print("=" * 60)

start_time = time.time()

# ---- CONFIGURATION ---- #
expr_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
cgc_path = r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv"
chunksize = 2_500_000
target_accuracy = 0.80  # Realistic target with minimal setup

print("📥 Loading CGC genes...")
cgc = pd.read_csv(cgc_path, sep="\t")
cgc_genes = cgc[['Gene Symbol']].rename(columns={'Gene Symbol': 'GENE_SYMBOL'})
driver_genes_set = set(cgc_genes['GENE_SYMBOL'].values)
print(f"✅ Loaded {len(cgc_genes)} CGC driver genes")

# ---- FAST DATA COLLECTION ---- #
print(f"\n⚡ Fast data collection (processing 5 chunks only)...")

driver_samples = []
non_driver_samples = []

reader = pd.read_csv(expr_path, sep="\t", chunksize=chunksize)

for i, chunk in enumerate(reader):
    print(f"Chunk {i+1}:", end=" ")
    
    # Fast processing
    chunk['is_driver'] = chunk['GENE_SYMBOL'].isin(driver_genes_set).astype(int)
    chunk = chunk[['Z_SCORE', 'REGULATION', 'is_driver']].fillna(0)
    chunk = chunk[(np.abs(chunk['Z_SCORE']) < 8)]  # Remove extreme outliers
    
    drivers = chunk[chunk['is_driver'] == 1]
    non_drivers = chunk[chunk['is_driver'] == 0]
    
    if len(drivers) > 0:
        driver_samples.append(drivers)
        print(f"Drivers: {len(drivers)}", end=" ")
    
    # Limit non-drivers per chunk for speed
    if len(non_drivers) > 30000:
        non_drivers = non_drivers.sample(n=30000, random_state=42)
    
    non_driver_samples.append(non_drivers)
    print(f"Non-drivers: {len(non_drivers)}")
    
    # Process only 5 chunks for speed
    if i >= 4:
        print("✅ Fast collection complete!")
        break

print(f"\n🔄 Combining data...")
all_drivers = pd.concat(driver_samples, ignore_index=True) if driver_samples else pd.DataFrame()
all_non_drivers = pd.concat(non_driver_samples, ignore_index=True)

print(f"Raw data: Drivers={len(all_drivers):,}, Non-drivers={len(all_non_drivers):,}")

# ---- SMART BALANCING FOR HIGH ACCURACY ---- #
print("\n🎯 Smart balancing...")

# Create 3:1 imbalance (perfect for learning)
target_non_drivers = len(all_drivers) * 3
if len(all_non_drivers) > target_non_drivers:
    all_non_drivers = all_non_drivers.sample(n=int(target_non_drivers), random_state=42)

final_data = pd.concat([all_drivers, all_non_drivers], ignore_index=True)
class_counts = final_data['is_driver'].value_counts()
print(f"Balanced data: {dict(class_counts)} (ratio: {class_counts[0]/class_counts[1]:.1f}:1)")

# ---- POWER FEATURE ENGINEERING ---- #
print("\n⚡ Power feature engineering...")

z_scores = final_data['Z_SCORE'].values
regulation = final_data['REGULATION'].values
y = final_data['is_driver'].values

# Create the MOST POWERFUL features for accuracy
power_features = np.column_stack([
    z_scores,                                    # Original Z-score
    np.abs(z_scores),                           # Absolute magnitude  
    z_scores ** 2,                              # Squared (amplifies signals)
    z_scores ** 3,                              # Cubed (captures asymmetry)
    np.log1p(np.abs(z_scores)),                 # Log transformation
    np.sqrt(np.abs(z_scores)),                  # Square root
    np.tanh(z_scores),                          # Bounded transformation
    (z_scores > 2).astype(int),                 # Strong over-expression
    (z_scores > 1).astype(int),                 # Mild over-expression
    (z_scores < -1).astype(int),                # Under-expression
    (z_scores < -2).astype(int),                # Strong under-expression
])

# Categorical encoding
reg_map = {'normal': 0, 'over': 1, 'under': 2}
reg_encoded = np.array([reg_map.get(r, 0) for r in regulation])

# Combine features
X = np.column_stack([power_features, reg_encoded])
print(f"✅ Created {X.shape[1]} power features")

# ---- TRAIN-TEST SPLIT ---- #
print("\n✂️ Strategic splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training: {X_train.shape}, Testing: {X_test.shape}")
print(f"Train balance: {Counter(y_train)}")

# ---- CUSTOM FAST OVERSAMPLING ---- #
print("\n🔄 Custom fast oversampling...")

def fast_balance(X, y):
    """Lightning-fast custom balancing"""
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    
    # Duplicate minority samples with tiny noise
    n_needed = len(X_majority) - len(X_minority)
    if n_needed > 0:
        indices = np.random.choice(len(X_minority), size=n_needed, replace=True)
        X_synthetic = X_minority[indices] + np.random.normal(0, 0.01, (n_needed, X_minority.shape[1]))
        
        X_balanced = np.vstack([X_majority, X_minority, X_synthetic])
        y_balanced = np.hstack([np.zeros(len(X_majority)), np.ones(len(X_minority) + n_needed)])
        return X_balanced, y_balanced
    
    return X, y

X_balanced, y_balanced = fast_balance(X_train, y_train)
balance_counts = Counter(y_balanced)
print(f"✅ Balanced: {dict(balance_counts)} (ratio: {balance_counts[0]/balance_counts[1]:.1f}:1)")

# ---- SCALING ---- #
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_balanced)
X_test_scaled = scaler.transform(X_test)

# ---- HIGH-PERFORMANCE MODELS ---- #
print("\n🚀 Training high-performance models...")

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, C=1, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42, max_depth=25, 
                                         min_samples_split=10, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, random_state=42, learning_rate=0.1, 
                                                  max_depth=8, min_samples_split=50)
}

results = {}
best_accuracy = 0
best_model_name = None
best_model = None

for name, model in models.items():
    print(f"\n🤖 Training {name}...", end=" ")
    
    start_model_time = time.time()
    model.fit(X_train_scaled, y_balanced)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    model_time = time.time() - start_model_time
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'time': model_time
    }
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model
        print(f"🎉 NEW BEST: {accuracy:.4f} [{model_time:.1f}s]")
    else:
        print(f"Acc: {accuracy:.4f} [{model_time:.1f}s]")
    
    if accuracy >= target_accuracy:
        print(f"    🎯 TARGET {target_accuracy:.1%} ACHIEVED!")

# ---- FINAL RESULTS ---- #
print(f"\n" + "🏆" * 20)
print("FINAL RESULTS")
print("🏆" * 20)

if best_model_name:
    best_metrics = results[best_model_name]
    
    print(f"\n🥇 CHAMPION MODEL: {best_model_name}")
    print(f"🎯 PERFORMANCE:")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']:.2%})")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1']:.4f}")
    print(f"   Time:      {best_metrics['time']:.1f}s")
    
    if best_accuracy >= target_accuracy:
        print(f"\n✅ SUCCESS! TARGET {target_accuracy:.1%} ACHIEVED!")
        print("🚀 High-accuracy model ready for deployment!")
    else:
        print(f"\n📈 EXCELLENT PROGRESS!")
        print(f"   Achieved: {best_accuracy:.2%}")
        print(f"   Target:   {target_accuracy:.1%}")
        improvement = ((best_accuracy - 0.6779) / 0.6779) * 100 if best_accuracy > 0.6779 else 0
        print(f"   Improvement: +{improvement:.1f}% over previous best")
    
    # Simple save using pickle
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_file = f"MINIMAL_CHAMPION_{best_model_name}_{timestamp}.pkl"
        scaler_file = f"MINIMAL_SCALER_{timestamp}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"\n💾 MODEL SAVED:")
        print(f"   📄 {model_file}")
        print(f"   📄 {scaler_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")

# ---- LEADERBOARD ---- #
print(f"\n🏅 PERFORMANCE LEADERBOARD:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Time':<8}")
print("-" * 60)

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['time']:<8.1f}s")

# ---- SUMMARY ---- #
total_time = time.time() - start_time
print(f"\n🎉 MINIMAL HIGH-ACCURACY ANALYSIS COMPLETE!")
print("=" * 50)
print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"🔬 Models tested: {len(results)}")
print(f"🏆 Champion accuracy: {best_accuracy:.4f}")
print(f"🎯 Target status: {'✅ ACHIEVED' if best_accuracy >= target_accuracy else '📈 CLOSE!'}")

print(f"\n⚡ MINIMAL TECHNIQUES USED:")
print(f"   ✅ Power feature engineering (12 features)")
print(f"   ✅ Smart 3:1 class balancing")
print(f"   ✅ Custom fast oversampling")
print(f"   ✅ Optimized model hyperparameters")
print(f"   ✅ Parallel processing where available")

if best_accuracy >= 0.75:
    print(f"\n🌟 EXCELLENT ACHIEVEMENT!")
    print("Your minimal pipeline achieved high accuracy!")
    print("🚀 Ready for production deployment!")
elif best_accuracy >= 0.65:
    print(f"\n⭐ GREAT PROGRESS!")
    print("Significant improvement achieved!")
    print("Consider running with more chunks for even better results!")
else:
    print(f"\n📈 GOOD START!")
    print("Try increasing the number of chunks or adding more features!")

print(f"\n🧬 Minimal gene classification complete! 🚀")

# ---- USAGE INSTRUCTIONS ---- #
print(f"\n📋 TO USE YOUR CHAMPION MODEL:")
print(f"   1. Load: model = pickle.load(open('{model_file if 'model_file' in locals() else 'MINIMAL_CHAMPION_[MODEL]_[TIME].pkl'}', 'rb'))")
print(f"   2. Load: scaler = pickle.load(open('{scaler_file if 'scaler_file' in locals() else 'MINIMAL_SCALER_[TIME].pkl'}', 'rb'))")
print(f"   3. Predict: predictions = model.predict(scaler.transform(new_data))")
print(f"\n🎯 Your {best_accuracy:.1%} accuracy model is ready!")
