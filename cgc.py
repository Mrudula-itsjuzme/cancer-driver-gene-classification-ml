import pandas as pd

# Load your gene expression data in chunks
expr_iter = pd.read_csv(
    r"H:\Cosmic_CompleteGeneExpression_v102_GRCh37.tsv",
    sep="\t",
    chunksize=2_500_000
)

# Load CGC gene list
cgc = pd.read_csv(r"H:\sem3\bio\Census_symbolTue Aug 19 04_30_59 2025.tsv", sep="\t")
print("CGC Columns:", cgc.columns)

# Keep only Gene Symbol
cgc_genes = cgc[['Gene Symbol']].rename(columns={'Gene Symbol':'GENE_SYMBOL'})

# Collect processed chunks here
chunks = []

for i, chunk in enumerate(expr_iter):
    print(f"Processing chunk {i}")
    chunk['is_driver'] = chunk['GENE_SYMBOL'].isin(cgc_genes['GENE_SYMBOL']).astype(int)
    chunks.append(chunk[['GENE_SYMBOL','Z_SCORE','REGULATION','is_driver']])  # keep only needed cols

# Concatenate everything into one final DataFrame
df_full = pd.concat(chunks, ignore_index=True)

print("✅ Final merged dataset shape:", df_full.shape)
print(df_full['is_driver'].value_counts())
