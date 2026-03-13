import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# File path
file_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"

# Load first 1000 rows to peek
df_iter = pd.read_csv(file_path, sep="\t", chunksize=1000)
df_sample = next(df_iter)

# Show structure
print("📊 Data Preview:")
print(df_sample.head())
print("\n🔍 Info:")
print(df_sample.info())
print("\nMissing values per column:")
print(df_sample.isnull().sum())

# --- Numeric exploration (Z-scores only) ---
df_numeric = df_sample.select_dtypes(include=["float64", "int64"])

print("\nSummary Stats (numeric):")
print(df_numeric.describe().T)

# --- Check duplicates ---
print("\nDuplicate rows:", df_sample.duplicated().sum())

# --- Histograms of Z-scores ---
plt.figure(figsize=(10, 6))
df_numeric["Z_SCORE"].hist(bins=50)
plt.title("Distribution of Gene Expression Z-scores (Sample Chunk)")
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.show()

# --- Correlation heatmap (optional) ---
if "Z_SCORE" in df_numeric.columns:
    corr_matrix = df_numeric.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

plt.figure(figsize=(8,6))
sns.boxplot(x="REGULATION", y="Z_SCORE", data=df_sample)
plt.title("Z-score Distribution by Regulation Category")
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x="REGULATION", y="Z_SCORE", data=df_sample)
plt.title("Violin Plot of Z-scores by Regulation")
plt.show()

print(df_sample.groupby("REGULATION")["Z_SCORE"].describe())