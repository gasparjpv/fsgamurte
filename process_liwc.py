# === CÉLULA 3: processamento de LIWC ===

import os
import glob
import gc
import json
import numpy as np
import pandas as pd

from pathlib import Path
from multiprocessing import Pool, cpu_count

from gafsmre.representations.liwc import LIWC

# =========================================================
# CONFIG
# =========================================================

col_texto = "abstracts"

data_dir = "vectors/abstracts/arXiv/database"
out_dir = "vectors/abstracts/arXiv/LIWC"

os.makedirs(out_dir, exist_ok=True)

liwc_path = Path("shared/dictionaries/liwc/LIWC2015.dic")

pattern = os.path.join(data_dir, "[0-9]_df_corpus_5000.parquet")
parquet_files = sorted(glob.glob(pattern))

print("Arquivos encontrados:")
print(parquet_files)
print(f"Total: {len(parquet_files)}\n")

# =========================================================
# WORKER
# =========================================================

def process_liwc(parquet_path):

    print(f"Processando: {parquet_path}")

    # Instancia LIWC dentro do processo
    model = LIWC(dic_filepath=liwc_path)

    # =====================================================
    # LEITURA
    # =====================================================

    df = pd.read_parquet(parquet_path)

    corpus = df[col_texto].astype(str).tolist()

    # =====================================================
    # REPRESENTAÇÃO
    # =====================================================

    features, X_liwc = model.generate_representation(corpus)

    print(f"Shape LIWC {parquet_path}: {X_liwc.shape}")

    # =====================================================
    # SALVAR FEATURES
    # =====================================================

    features_path = os.path.join(out_dir, "features_liwc.json")

    if not os.path.exists(features_path):

        if isinstance(features, np.ndarray):
            features_to_save = features.tolist()

        else:
            try:
                features_to_save = list(features)
            except TypeError:
                features_to_save = [str(f) for f in features]

        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(
                features_to_save,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"features_liwc.json salvo em: {features_path}")

    # =====================================================
    # SALVAR MATRIZ
    # =====================================================

    fname = os.path.splitext(
        os.path.basename(parquet_path)
    )[0]

    out_path = os.path.join(
        out_dir,
        f"{fname}_LIWC.npy"
    )

    np.save(out_path, X_liwc)

    print(f"Vetores LIWC salvos em: {out_path}\n")

    # =====================================================
    # CLEANUP
    # =====================================================

    del df
    del corpus
    del X_liwc
    del features

    gc.collect()

# =========================================================
# EXECUÇÃO PARALELA
# =========================================================

if __name__ == "__main__":

    workers = max(1, cpu_count() - 1)

    print(f"Workers: {workers}\n")

    with Pool(processes=workers) as pool:
        pool.map(process_liwc, parquet_files)

    print("Processamento LIWC concluído.")