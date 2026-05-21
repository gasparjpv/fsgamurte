# FSGA-MuRTe

Evolutionary Feature Selection in Multi-Representational Text Environments for Unsupervised Text Clustering.

Repository: [FSGA-MuRTe GitHub Repository](https://github.com/gasparjpv/FSGAMuTRe)


---

# Overview

This repository contains the codebase developed for the Master's Dissertation in Technology at the Faculty of Technology (FT) of the UNICAMP. 

The project proposes an evolutionary framework for unsupervised feature selection in multi-representational text clustering environments using Genetic Algorithms (GA).

The framework combines heterogeneous textual representations, including:

- Bag-of-Words (BoW)
- Word2Vec
- FastText
- Doc2Vec
- BERT/SBERT embeddings
- Part-of-Speech (POS) representations
- LIWC-based psycholinguistic features
- MRC Psycholinguistic Database features

All representations are individually L2-normalized and concatenated into a unified feature space through an early-fusion strategy. A Genetic Algorithm then performs feature selection using the Dunn Index as the fitness function, followed by clustering with K-Means.

The experiments evaluate clustering quality before and after feature selection using:

- Dunn Index (DI)
- Calinski–Harabasz Index (CH)
- Davies–Bouldin Index (DB)

The proposed approach demonstrated improvements in clustering compactness and separation across multiple datasets and textual domains.

---

# Requirements

- Python 3.11
- Poetry
- Linux/macOS recommended
- CUDA-compatible GPU optional

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/gasparjpv/fsgamurte.git
cd fsgamurte
```

---

## 2. Install Poetry

Official documentation:

```text
https://python-poetry.org/docs/
```

Linux/macOS example:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:

```bash
poetry --version
```

---

## 3. Create the Virtual Environment

Configure Poetry to create the `.venv` inside the project:

```bash
poetry config virtualenvs.in-project true
```

Install dependencies:

```bash
poetry install
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

---

# Execution Pipeline

The scripts must be executed sequentially in the following order to reproduce the experiments and statistical analyses.

---

## Step 1 — Dataset Processing and Representation Generation

```bash
poetry run python process_arxiv.py
```

This stage performs:

- dataset preprocessing
- text normalization
- feature extraction
- representation generation
- vector storage

---

## Step 2 — Evolutionary Feature Selection (FSGA-MuRTe)

```bash
poetry run python run_fsga_murte.py
```

This stage performs:

- early fusion of representations
- Genetic Algorithm optimization
- feature subset selection
- clustering before and after feature selection
- metric computation

---

## Step 3 — Statistical Aggregation and Metrics

```bash
poetry run python run_statistics.py
```

This stage computes:

- aggregated experimental statistics
- mean and standard deviation
- clustering performance summaries
- comparative analyses

---

## Step 4 — Wilcoxon Statistical Analysis

```bash
poetry run python wilcoxon_analysis.py
```

This stage performs:

- Wilcoxon signed-rank statistical tests
- significance validation
- robustness analysis of clustering improvements

---

# Important Configuration Notes

The repository contains multiple datasets organized by domain:

```text
datasets/
├── Abstracts/
│   ├── arXiv/
│   ├── Klinger/
│   └── NLP_Research/
│
├── News/
│   ├── CovidArticles/
│   ├── NewsPopularity/
│   └── WELFake/
│
├── Reviews/
│   ├── DrugReviews/
│   ├── Pitchfork/
│   └── Sentiment_Labelled/
│
└── Short_Texts/
    ├── SMS/
    └── Truth_Seeker/
```

The default execution examples provided in this repository use the `arXiv` dataset as the reference configuration.

To execute experiments with other datasets, it is necessary to manually update the dataset paths inside the scripts.

Examples of parameters that may require modification:

```python
DATASET_PATH = "datasets/Abstracts/arXiv/cleansed_data.csv"
```

or

```python
BASE_PATH = "datasets/News/WELFake/"
```

Users should replace these paths according to the dataset they intend to process.

Example:

```python
DATASET_PATH = "datasets/Reviews/DrugReviews/cleansed_data.csv"
```

This adjustment must be performed before executing the experimental pipeline for a different dataset.

---

# Datasets

Some datasets used in this project may be distributed under specific licensing restrictions and therefore cannot be publicly redistributed.

The datasets may be made available upon request, depending on licensing conditions and permissions.

---

# External Resources

This repository depends on external linguistic and psycholinguistic resources.

## LIWC

LIWC must be obtained directly from the official distributors.

Official website:

```text
https://www.liwc.app/
```

---

## MRC Psycholinguistic Database

The MRC database must be obtained directly from its maintainers.

Official website:

```text
https://websites.psychology.uwa.edu.au/school/MRCDatabase/uwa_mrc.htm
```

---

# Notes

- Large intermediate files, embeddings, caches, and generated vectors are excluded from version control.
- GPU acceleration may significantly improve execution time for embedding generation and optimization stages.
- Some experiments may require substantial RAM and storage resources depending on dataset size and representation dimensionality.

---

# Citation

If you use this repository in academic work, please cite the corresponding dissertation or future publication.

```bibtex
@mastersthesis{gaspar2026fsgamurte,
  title  = {An Evolutionary Feature Selection Approach in Multi-Representational Text Environments},
  author = {João Gaspar},
  school = {University of Campinas (UNICAMP)},
  year   = {2026}
}
```

---

# License

```text
MIT License
```
