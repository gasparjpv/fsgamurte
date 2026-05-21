# === CÉLULA 8: processamento de FastText ===

import os
import glob
import gc

import numpy as np
import pandas as pd

from gafsmre.representations.fasttext import FastText

# =========================================================
# CONFIGURAÇÃO
# =========================================================

col_texto = "abstracts"

data_dir = "vectors/abstracts/arXiv/database"

out_dir = "vectors/abstracts/arXiv/FastText"

os.makedirs(out_dir, exist_ok=True)

# =========================================================
# CPU WORKERS
# =========================================================

workers = min(
    8,
    max(1, os.cpu_count() - 1)
)

print(f"Workers FastText: {workers}\n")

# =========================================================
# CORPUS
# =========================================================

corpus_pattern = os.path.join(
    data_dir,
    "[0-9]_df_corpus_5000.parquet"
)

corpus_files = sorted(
    glob.glob(corpus_pattern)
)

print("Corpora encontrados:")
print(corpus_files)
print()

# =========================================================
# PROCESSAMENTO
# =========================================================

for idx_loop, corpus_path in enumerate(corpus_files):

    print(f"Processando corpus: {corpus_path}")

    # =====================================================
    # LEITURA
    # =====================================================

    df = pd.read_parquet(
        corpus_path,
        engine="pyarrow"
    )

    corpus = df[col_texto].astype(str).tolist()

    # =====================================================
    # TREINAMENTO FASTTEXT
    # =====================================================

    ft = FastText(
        train_corpus=corpus,
        min_count=1,
        vector_size=300,
        workers=workers,
        stop_word_removal_enabled=False,
        train_algorithm="skip-gram",
    )

    # =====================================================
    # REPRESENTAÇÃO
    # =====================================================

    _, X_ft = ft.generate_representation(
        corpus
    )

    print("Shape FastText:", X_ft.shape)

    # =====================================================
    # REDUZIR MEMÓRIA
    # =====================================================

    X_ft = X_ft.astype(np.float32)

    # =====================================================
    # SAVE
    # =====================================================

    idx = os.path.basename(
        corpus_path
    ).split("_")[0]

    out_path = os.path.join(
        out_dir,
        f"{idx}_FastText.npy"
    )

    np.save(out_path, X_ft)

    print(
        f"Vetores FastText salvos em: "
        f"{out_path}\n"
    )

    # =====================================================
    # CLEANUP
    # =====================================================

    del df
    del corpus
    del X_ft
    del ft

    if idx_loop % 2 == 0:
        gc.collect()

print("Processamento FastText concluído.")