# === CÉLULA 7: processamento de Word2Vec (treino direto no corpus, sem train separado) ===

import os
import glob
import gc
import numpy as np
import pandas as pd

from gafsmre.representations.word2vec import Word2Vec

# Nome da coluna de texto
col_texto = "abstracts"

# Diretórios
data_dir = "vectors/abstracts/arXiv/database"   # onde estão os parquet
out_dir  = "vectors/abstracts/arXiv/Word2Vec"   # onde salvar vetores Word2Vec
os.makedirs(out_dir, exist_ok=True)

# Padrão do corpus principal (sem train separado)
corpus_pattern = os.path.join(data_dir, "[0-9]_df_corpus_5000.parquet")
corpus_files = sorted(glob.glob(corpus_pattern))

print("Corpora encontrados:", corpus_files)
print()

for corpus_path in corpus_files:
    print(f"Processando corpus: {corpus_path}")

    # === 1) Ler corpus ===
    df = pd.read_parquet(corpus_path)
    corpus = df[col_texto].astype(str).tolist()

    # === 2) Treinar Word2Vec diretamente no corpus ===
    w2v = Word2Vec(
        train_corpus=corpus,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,                 # 4 vCPUs da tua instância
        stop_word_removal_enabled=False,
    )

    # === 3) Gerar representação para o próprio corpus ===
    _, X_w2v = w2v.generate_representation(corpus)
    print("Shape Word2Vec:", X_w2v.shape)

    # === 4) Salvar matriz de vetores Word2Vec ===
    idx = os.path.basename(corpus_path).split("_")[0]   # "0" de "0_df_corpus_5000"
    out_path = os.path.join(out_dir, f"{idx}_Word2Vec.npy")

    np.save(out_path, X_w2v)
    print(f"Vetores Word2Vec salvos em: {out_path}\n")

    # === 5) Limpeza de memória ===
    del df, corpus, X_w2v, w2v
    gc.collect()

print("Processamento Word2Vec concluído.")
