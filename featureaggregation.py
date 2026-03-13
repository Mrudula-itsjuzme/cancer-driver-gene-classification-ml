import pandas as pd

# Load data (chunking still for huge file, but here let’s demo on 100k rows maybe)
file_path = r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv"
chunksize = 2500000  # adjust as needed

agg_list = []
for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize):
    # Ensure Z_SCORE is numeric
    chunk["Z_SCORE"] = pd.to_numeric(chunk["Z_SCORE"], errors="coerce")
    print(f"Processing chunk {len(agg_list)}")
    
    # Group by gene within the chunk
    grouped = chunk.groupby("GENE_SYMBOL").agg(
        mean_z=("Z_SCORE", "mean"),
        std_z=("Z_SCORE", "std"),
        over_frac=("REGULATION", lambda x: (x=="over").mean()),
        under_frac=("REGULATION", lambda x: (x=="under").mean()),
        normal_frac=("REGULATION", lambda x: (x=="normal").mean()),
        sample_count=("Z_SCORE", "count")
    )
    agg_list.append(grouped)

# Concatenate and re-aggregate across all chunks
gene_features = pd.concat(agg_list).groupby(level=0).agg(
    mean_z=("mean_z", "mean"),
    std_z=("std_z", "mean"),  # mean of stds across chunks
    over_frac=("over_frac", "mean"),
    under_frac=("under_frac", "mean"),
    normal_frac=("normal_frac", "mean"),
    sample_count=("sample_count", "sum")
)

print(f"Aggregated {len(gene_features)} genes")

# Preview
print("\nGene Features:")
print(gene_features.head(20))

##ranking
# Top 20 over-expressed candidates
print("\nTop 20 Over-expressed Genes:")
print(gene_features.sort_values("over_frac", ascending=False).head(20))

# Top 20 under-expressed candidates
print("\nTop 20 Under-expressed Genes:")
print(gene_features.sort_values("under_frac", ascending=False).head(20))

import matplotlib.pyplot as plt

# Scatterplot: mean Z vs fraction over-expressed
print("\nScatterplot: mean Z vs fraction over-expressed")
plt.figure(figsize=(8,6))
plt.scatter(
    gene_features["mean_z"], 
    gene_features["over_frac"], 
    alpha=0.6, s=20
)
plt.xlabel("Mean Z-score per Gene")
plt.ylabel("Fraction Over-expressed")
plt.title("Gene-level Expression Features")
plt.show()

