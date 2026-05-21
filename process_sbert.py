# === CÉLULA 2: processamento do SBert ===

import os
import glob
import gc
import numpy as np
import pandas as pd

from gafsmre.representations.sbert import SBert
from sentence_transformers import SentenceTransformer


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Nome da coluna de texto
col_texto = "abstracts"

# Diretórios
data_dir = "vectors/abstracts/arXiv/database"
out_dir = "vectors/abstracts/arXiv/Sbert"

os.makedirs(out_dir, exist_ok=True)

# =========================================================
# SBERT
# =========================================================

# Instanciar modelo apenas 1 vez
sbert = SBert(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=128,
    device="cuda",
    stop_word_removal_enabled=False
)
# =========================================================
# LEITURA DOS PARQUETS
# =========================================================

# Agora processa os 10 arquivos
pattern = os.path.join(data_dir, "[0-9]_df_corpus_5000.parquet")

parquet_files = sorted(glob.glob(pattern))

print("Arquivos encontrados:")
for f in parquet_files:
    print(f)

print(f"\nTotal encontrado: {len(parquet_files)}")

# =========================================================
# PROCESSAMENTO
# =========================================================

for parquet_path in parquet_files:

    print(f"\n=================================================")
    print(f"Processando: {parquet_path}")

    # -----------------------------------------------------
    # 1) Ler parquet
    # -----------------------------------------------------

    df = pd.read_parquet(parquet_path)

    print(f"Total de registros carregados: {len(df)}")

    # Confirma integridade
    if len(df) != 2000:
        raise ValueError(
            f"Arquivo {parquet_path} possui "
            f"{len(df)} registros ao invés de 5000."
        )

    # -----------------------------------------------------
    # 2) Gerar corpus
    # -----------------------------------------------------

    corpus = df[col_texto].astype(str).tolist()

    print(f"Corpus criado com {len(corpus)} textos.")

    # -----------------------------------------------------
    # 3) Gerar embeddings SBERT
    # -----------------------------------------------------

    features, X_sbert = sbert.generate_representation(corpus)

    print("Shape SBert:", X_sbert.shape)

    # -----------------------------------------------------
    # 4) Nome de saída
    # -----------------------------------------------------

    fname = os.path.splitext(
        os.path.basename(parquet_path)
    )[0]

    out_path = os.path.join(
        out_dir,
        f"{fname}_SBert.npy"
    )

    # -----------------------------------------------------
    # 5) Salvar vetores
    # -----------------------------------------------------

    np.save(out_path, X_sbert)

    print(f"Vetores salvos em:")
    print(out_path)

    # -----------------------------------------------------
    # 6) Limpeza de memória
    # -----------------------------------------------------

    del df
    del corpus
    del X_sbert
    del features

    gc.collect()

print("\nProcessamento concluído.")

# =========================================================
# LIBERAR MEMÓRIA
# =========================================================

del sbert

gc.collect()