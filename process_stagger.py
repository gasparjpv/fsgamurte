#!/usr/bin/env python3

# =========================================================
# STAGGER PARALLEL PIPELINE (.py)
# =========================================================

import gc
import glob
import json
import multiprocessing as mp
import os
import time

# =========================================================
# CONTROLE THREADS INTERNAS
# =========================================================

# Evita oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Evita conflitos tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Força CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

from gafsmre.representations.stagger import STagger

# =========================================================
# CONFIG
# =========================================================

col_texto = "abstracts"

data_dir = "vectors/abstracts/arXiv/database"

out_dir = "vectors/abstracts/arXiv/STagger"

os.makedirs(out_dir, exist_ok=True)

# =========================================================
# ARQUIVOS
# =========================================================

# pega arquivos de 0 até 9
pattern = os.path.join(data_dir, "*_df_corpus_5000.parquet")

parquet_files = sorted(glob.glob(pattern))

print("\nArquivos encontrados:\n")

for p in parquet_files:
    print(p)

print(f"\nTotal: {len(parquet_files)}\n")

# =========================================================
# WORKER
# =========================================================


def process_stagger(parquet_path):

    start = time.time()

    try:

        print("\n================================================")
        print(f"PROCESSANDO: {parquet_path}")
        print("================================================\n")

        # =================================================
        # MODELO LOCAL
        # =================================================

        # Worker isolado
        s = STagger()

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

        features_st, X_STagger = s.generate_representation(corpus, as_dataframe=False)

        print(f"Shape STagger: {X_STagger.shape}")

        # =================================================
        # REDUÇÃO MEMÓRIA
        # =================================================

        X_STagger = X_STagger.astype(np.float32)

        # =================================================
        # FEATURES
        # =================================================

        features_path = os.path.join(out_dir, "features_stagger.json")

        if not os.path.exists(features_path):

            if isinstance(features_st, np.ndarray):

                features_to_save = features_st.tolist()

            else:

                try:
                    features_to_save = list(features_st)

                except TypeError:

                    features_to_save = [str(x) for x in features_st]

            with open(features_path, "w", encoding="utf-8") as f:

                json.dump(features_to_save, f, ensure_ascii=False, indent=2)

            print("\nFeatures salvas:")
            print(features_path)

        # =================================================
        # OUTPUT
        # =================================================

        fname = os.path.splitext(os.path.basename(parquet_path))[0]

        out_path = os.path.join(out_dir, f"{fname}_STagger.npy")

        np.save(out_path, X_STagger)

        print("\nVetores salvos:")
        print(out_path)

        # =================================================
        # CLEANUP
        # =================================================

        del df
        del corpus
        del X_STagger
        del features_st
        del s

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

    # -----------------------------------------------------
    # spawn evita problemas multiprocessing + spaCy
    # -----------------------------------------------------

    mp.set_start_method("spawn", force=True)

    # =====================================================
    # WORKERS
    # =====================================================

    workers = min(4, max(1, os.cpu_count() - 1))

    print(f"Workers utilizados: {workers}\n")

    # =====================================================
    # EXECUÇÃO PARALELA
    # =====================================================

    # maxtasksperchild=1:
    # cada worker processa 1 parquet
    # depois MORRE
    #
    # resolve leaks/cache do spaCy
    # mantendo paralelismo
    # =====================================================

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:

        results = pool.map(process_stagger, parquet_files)

    # =====================================================
    # RESULTADOS
    # =====================================================

    success = sum(results)

    failed = len(results) - success

    print("\n===================================")
    print("PROCESSAMENTO FINALIZADO")
    print("===================================")

    print(f"Sucesso : {success}")
    print(f"Falhas  : {failed}")

    if failed == 0:

        print("\nTodos os arquivos processados.")
