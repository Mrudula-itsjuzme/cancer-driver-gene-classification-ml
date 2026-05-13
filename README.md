# Cancer Driver Gene Classification with Machine Learning

A bioinformatics machine-learning project for classifying cancer driver genes using feature engineering, curated gene datasets, and supervised ML pipelines.

The goal is to explore how computational models can help separate likely cancer-associated driver genes from non-driver or less relevant genes using structured biological features.

---

## Project links and evidence

| Item | Link / Note |
|---|---|
| Repository | https://github.com/Mrudula-itsjuzme/cancer-driver-gene-classification-ml |
| Paper / reference | No external paper link attached yet; methodology is documented in project reports |
| Demo video | Not uploaded yet |
| Deployment | Not applicable; research/ML pipeline |
| Dataset note | Uses gene-symbol/cancer-gene census style data included or referenced in the repo; provenance should be expanded in a future update |
| Result screenshots / reports | Check `Technical_Report_Cancer_Gene_Classification.md`, `ML_Analysis_Summary.md`, and `VERIFICATION_REPORT.md` |

---

## Problem statement

Cancer driver genes play an important role in tumor development and progression. Manually identifying these genes is slow and requires biological expertise, curated datasets, and repeated validation.

This repository experiments with machine-learning pipelines that learn patterns from gene-level data and classify genes based on cancer-driver relevance.

---

## What this project includes

- gene-symbol and cancer-gene dataset preparation
- feature aggregation and preprocessing
- model-training scripts
- high-accuracy minimal pipelines
- technical reports and verification notes
- experiment summaries for reproducibility

---

## Pipeline

```text
Gene / Census Data
        ↓
Cleaning + Feature Engineering
        ↓
Training Dataset Construction
        ↓
ML Model Training
        ↓
Evaluation + Verification
        ↓
Classification Results + Reports
```

---

## Key files

| File | Purpose |
|---|---|
| `ULTRA_ENHANCED_ML_PIPELINE.py` | expanded ML experimentation pipeline |
| `MINIMAL_HIGH_ACCURACY.py` | compact high-performing training script |
| `Technical_Report_Cancer_Gene_Classification.md` | project report and methodology notes |
| `ML_Analysis_Summary.md` | summary of model experiments and findings |
| `VERIFICATION_REPORT.md` | validation and verification notes |

---

## Quick start

```bash
git clone https://github.com/Mrudula-itsjuzme/cancer-driver-gene-classification-ml.git
cd cancer-driver-gene-classification-ml

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn
```

Run one of the main pipelines:

```bash
python MINIMAL_HIGH_ACCURACY.py
```

or:

```bash
python ULTRA_ENHANCED_ML_PIPELINE.py
```

---

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Bioinformatics datasets
- Supervised machine learning

---

## Why this project matters

The project connects machine learning with biomedical discovery. It shows how structured biological data can be turned into a classification pipeline and how model outputs can support gene-prioritization workflows.

---

## Current status

Research/prototype implementation. The next improvement would be adding a single `requirements.txt`, a CLI runner, and clearly documented dataset provenance so the experiments are easier to reproduce end-to-end.

---

## Author

Built by [Pedamallu Sai Mrudula](https://github.com/Mrudula-itsjuzme) as part of an applied AI and biomedical machine-learning portfolio.
