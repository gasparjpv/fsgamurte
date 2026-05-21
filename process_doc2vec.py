# === Doc2Vec - treinamento direto no corpus ===

import os
import glob
import gc

import numpy as np
import pandas as pd

from gafsmre.representations.doc2vec import Doc2Vec

# =========================================================
# CONFIGURAÇÃO
# =========================================================

col_texto = "abstracts"

data_dir = "vectors/abstracts/arXiv/database"

out_dir = "vectors/abstracts/arXiv/Doc2Vec"

os.makedirs(out_dir, exist_ok=True)

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
    # TREINAMENTO DOC2VEC
    # =====================================================

    model = Doc2Vec(
        train_corpus=corpus,
        vector_size=100,
        train_algorithm="PV-DBOW",
        stop_word_removal_enabled=False,
    )

    # =====================================================
    # INFERÊNCIA
    # =====================================================

    _, X_doc2vec = (
        model.generate_representation(
            corpus
        )
    )

    print("Shape Doc2Vec:", X_doc2vec.shape)

    # =====================================================
    # FLOAT32
    # =====================================================

    X_doc2vec = X_doc2vec.astype(np.float32)

    # =====================================================
    # SAVE
    # =====================================================

    idx = os.path.basename(
        corpus_path
    ).split("_")[0]

    out_path = os.path.join(
        out_dir,
        f"{idx}_Doc2Vec.npy"
    )

    np.save(out_path, X_doc2vec)

    print(
        f"Vetores Doc2Vec salvos em: "
        f"{out_path}\n"
    )

    # =====================================================
    # LIMPEZA
    # =====================================================

    del df
    del corpus
    del X_doc2vec
    del model

    if idx_loop % 2 == 0:
        gc.collect()

print("Processamento Doc2Vec concluído.")