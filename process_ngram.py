# === CÉLULA 4: processamento de NGram ===

import os
import glob
import gc
import json

import numpy as np
import pandas as pd

from scipy.sparse import issparse, save_npz

from gafsmre.representations.ngram import NGram

# =========================================================
# CONFIGURAÇÃO
# =========================================================

# Nome da coluna de texto
col_texto = "abstracts"

# Diretórios
data_dir = "vectors/abstracts/arXiv/database"
out_dir = "vectors/abstracts/arXiv/NGram"

os.makedirs(out_dir, exist_ok=True)

# =========================================================
# MODELO NGRAM
# =========================================================

# Instanciar apenas 1 vez
ngram_model = NGram(
    min_ngram_group=1,
    max_ngram_group=1,
    model_type="tf-idf",
)

# =========================================================
# ARQUIVOS
# =========================================================

pattern = os.path.join(
    data_dir,
    "[0-9]_df_corpus_5000.parquet"
)

parquet_files = sorted(glob.glob(pattern))

print("Arquivos encontrados:")
print(parquet_files)
print(f"Total: {len(parquet_files)}\n")

# =========================================================
# FEATURES
# =========================================================

features_saved = False

# =========================================================
# PROCESSAMENTO
# =========================================================

for idx, parquet_path in enumerate(parquet_files):

    print(f"Processando: {parquet_path}")

    # =====================================================
    # LEITURA PARQUET
    # =====================================================

    df = pd.read_parquet(
        parquet_path,
        engine="pyarrow"
    )

    # =====================================================
    # CORPUS
    # =====================================================

    corpus = df[col_texto].astype(str).tolist()

    # =====================================================
    # REPRESENTAÇÃO NGRAM
    # =====================================================

    features, X_ngram = ngram_model.generate_representation(
        corpus
    )

    print("Shape NGram:", X_ngram.shape)
    print("Tipo:", type(X_ngram))

    # =====================================================
    # REDUZIR MEMÓRIA
    # =====================================================

    # Apenas para matrizes densas
    if not issparse(X_ngram):
        X_ngram = X_ngram.astype(np.float32)

    # =====================================================
    # SALVAR FEATURES
    # =====================================================

    if not features_saved:

        if isinstance(features, np.ndarray):
            features_to_save = features.tolist()

        else:
            try:
                features_to_save = list(features)

            except TypeError:
                features_to_save = [str(x) for x in features]

        features_path = os.path.join(
            out_dir,
            "features_ngram.json"
        )

        with open(
            features_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                features_to_save,
                f,
                ensure_ascii=False,
                indent=2
            )

        features_saved = True

        print(
            f"features_ngram.json salvo em: "
            f"{features_path}\n"
        )

    # =====================================================
    # NOME OUTPUT
    # =====================================================

    fname = os.path.splitext(
        os.path.basename(parquet_path)
    )[0]

    # =====================================================
    # SAVE SPARSE OU DENSO
    # =====================================================

    if issparse(X_ngram):

        out_path = os.path.join(
            out_dir,
            f"{fname}_NGram.npz"
        )

        save_npz(out_path, X_ngram)

    else:

        out_path = os.path.join(
            out_dir,
            f"{fname}_NGram.npy"
        )

        np.save(out_path, X_ngram)

    print(f"Vetores NGram salvos em: {out_path}\n")

    # =====================================================
    # LIMPEZA
    # =====================================================

    del df
    del corpus
    del X_ngram
    del features

    # GC menos agressivo
    if idx % 3 == 0:
        gc.collect()

print("Processamento NGram concluído.")