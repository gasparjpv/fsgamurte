#!/usr/bin/env python3

# =========================================================
# MRC2 PARALLEL PIPELINE (.py)
# =========================================================

import gc
import glob
import json
import multiprocessing as mp
import os
import sys
import time

# =========================================================
# ROOT / SRC
# =========================================================

BASE = os.path.dirname(os.path.abspath(__file__))

SRC_PATH = os.path.join(BASE, "src")

if SRC_PATH not in sys.path:

    sys.path.insert(0, SRC_PATH)

# =========================================================
# CONTROLE THREADS
# =========================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CPU ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =========================================================
# IMPORTS
# =========================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from gafsmre.representations.mrc2 import MRC2

# =========================================================
# CONFIGURAÇÃO
# =========================================================

col_texto = "abstracts"

data_dir = "vectors/abstracts/arXiv/database"

out_dir = "vectors/abstracts/arXiv/MRC2"

os.makedirs(
    out_dir,
    exist_ok=True,
)

# =========================================================
# DICIONÁRIO MRC2
# =========================================================

dic_path = Path("shared/dictionaries/mrc/mrc2.dct")

# =========================================================
# ARQUIVOS
# =========================================================

pattern = os.path.join(data_dir, "*_df_corpus_5000.parquet")

parquet_files = sorted(glob.glob(pattern))

print("\nArquivos encontrados:\n")

for p in parquet_files:

    print(p)

print(f"\nTotal: {len(parquet_files)}\n")

# =========================================================
# WORKER
# =========================================================


def process_mrc2(parquet_path):

    start = time.time()

    try:

        print("\n================================================")
        print(f"PROCESSANDO: {parquet_path}")
        print("================================================\n")

        # =================================================
        # MODELO LOCAL
        # =================================================

        model = MRC2(dic_filepath=dic_path)

        # =================================================
        # LEITURA
        # =================================================

        df = pd.read_parquet(parquet_path, engine="pyarrow")

        print(f"Linhas carregadas: {len(df)}")

        # =================================================
        # CORPUS
        # =================================================

        corpus = df[col_texto].fillna("").astype(str).tolist()

        # =================================================
        # REPRESENTAÇÃO
        # =================================================

        features_mrc2, X_mrc2 = model.generate_representation(
            corpus, as_dataframe=False
        )

        print(f"Shape MRC2: {X_mrc2.shape}")

        # =================================================
        # REDUÇÃO MEMÓRIA
        # =================================================

        X_mrc2 = X_mrc2.astype(np.float32)

        # =================================================
        # FEATURES
        # =================================================

        features_path = os.path.join(out_dir, "features_mrc2.json")

        if not os.path.exists(features_path):

            if isinstance(features_mrc2, np.ndarray):

                features_to_save = features_mrc2.tolist()

            else:

                try:

                    features_to_save = list(features_mrc2)

                except Exception:

                    features_to_save = [str(x) for x in features_mrc2]

            with open(features_path, "w", encoding="utf-8") as f:

                json.dump(features_to_save, f, ensure_ascii=False, indent=2)

            print("\nFeatures salvas:")
            print(features_path)

        # =================================================
        # OUTPUT
        # =================================================

        fname = os.path.splitext(os.path.basename(parquet_path))[0]

        out_path = os.path.join(out_dir, f"{fname}_MRC2.npy")

        np.save(out_path, X_mrc2)

        print("\nVetores salvos:")
        print(out_path)

        # =================================================
        # CLEANUP
        # =================================================

        del df
        del corpus
        del X_mrc2
        del features_mrc2
        del model

        gc.collect()

        elapsed = time.time() - start

        print(f"\nTempo: {elapsed:.2f}s")

        return True

    except Exception as e:

        print("\n================================")
        print(f"ERRO EM: {parquet_path}")
        print("================================\n")

        print(str(e))

        gc.collect()

        return False


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    workers = min(4, max(1, os.cpu_count() - 1))

    print(f"Workers utilizados: {workers}\n")

    results = []

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:

        futures = [
            executor.submit(process_mrc2, parquet_path)
            for parquet_path in parquet_files
        ]

        for future in as_completed(futures):

            try:

                result = future.result()

                results.append(result)

            except Exception as e:

                print("\n================================")
                print("ERRO NO WORKER")
                print("================================\n")

                print(str(e))

    success = sum(results)

    failed = len(results) - success

    print("\n===================================")
    print("PROCESSAMENTO FINALIZADO")
    print("===================================")

    print(f"Sucesso : {success}")
    print(f"Falhas  : {failed}")
