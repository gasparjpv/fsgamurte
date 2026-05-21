# === CÉLULA 9: concatenação das visualizações ===

import os
import gc

import numpy as np

from multiprocessing import Pool, cpu_count

from scipy import sparse
from scipy.sparse import load_npz, issparse

from sklearn.preprocessing import normalize

# =========================================================
# DIRETÓRIOS
# =========================================================

base_dir = "vectors/abstracts/arXiv"

sbert_dir   = os.path.join(base_dir, "Sbert")
liwc_dir    = os.path.join(base_dir, "LIWC")
ngram_dir   = os.path.join(base_dir, "NGram")
stagger_dir = os.path.join(base_dir, "STagger")
mrc2_dir    = os.path.join(base_dir, "MRC2")
doc2vec_dir = os.path.join(base_dir, "Doc2Vec")
w2v_dir     = os.path.join(base_dir, "Word2Vec")
ft_dir      = os.path.join(base_dir, "FastText")

concat_dir  = os.path.join(base_dir, "concat")

os.makedirs(concat_dir, exist_ok=True)

# =========================================================
# ÍNDICES
# =========================================================

indices = list(range(10))

# =========================================================
# LOADER INTELIGENTE
# =========================================================

def load_representation(path_without_ext):

    npy_path = f"{path_without_ext}.npy"
    npz_path = f"{path_without_ext}.npz"

    if os.path.exists(npz_path):

        X = load_npz(npz_path)

    elif os.path.exists(npy_path):

        X = np.load(npy_path)

    else:

        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{path_without_ext}"
        )

    return X

# =========================================================
# NORMALIZAÇÃO EFICIENTE
# =========================================================

def normalize_to_csr(X):

    # -----------------------------------------------------
    # SPARSE
    # -----------------------------------------------------

    if issparse(X):

        X = X.astype(np.float32)

        X = normalize(
            X,
            norm="l2",
            axis=1,
            copy=False
        )

        return X.tocsr()

    # -----------------------------------------------------
    # DENSO
    # -----------------------------------------------------

    X = X.astype(np.float32)

    X = normalize(
        X,
        norm="l2",
        axis=1,
        copy=False
    )

    return sparse.csr_matrix(X)

# =========================================================
# WORKER
# =========================================================

def process_concat(idx):

    print(f"Processando índice {idx}...")

    fname_corpus = f"{idx}_df_corpus_5000"

    # =====================================================
    # LOAD
    # =====================================================

    X_sbert = load_representation(
        os.path.join(
            sbert_dir,
            f"{fname_corpus}_SBert"
        )
    )

    X_liwc = load_representation(
        os.path.join(
            liwc_dir,
            f"{fname_corpus}_LIWC"
        )
    )

    X_ngram = load_representation(
        os.path.join(
            ngram_dir,
            f"{fname_corpus}_NGram"
        )
    )

    X_stagger = load_representation(
        os.path.join(
            stagger_dir,
            f"{fname_corpus}_STagger"
        )
    )

    X_mrc2 = load_representation(
        os.path.join(
            mrc2_dir,
            f"{fname_corpus}_MRC2"
        )
    )

    X_doc2vec = load_representation(
        os.path.join(
            doc2vec_dir,
            f"{idx}_Doc2Vec"
        )
    )

    X_w2v = load_representation(
        os.path.join(
            w2v_dir,
            f"{idx}_Word2Vec"
        )
    )

    X_ft = load_representation(
        os.path.join(
            ft_dir,
            f"{idx}_FastText"
        )
    )

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    X_sbert_csr   = normalize_to_csr(X_sbert)
    X_liwc_csr    = normalize_to_csr(X_liwc)
    X_ngram_csr   = normalize_to_csr(X_ngram)
    X_stagger_csr = normalize_to_csr(X_stagger)
    X_mrc2_csr    = normalize_to_csr(X_mrc2)
    X_doc2vec_csr = normalize_to_csr(X_doc2vec)
    X_w2v_csr     = normalize_to_csr(X_w2v)
    X_ft_csr      = normalize_to_csr(X_ft)

    # =====================================================
    # CONCAT
    # =====================================================

    X_concat_csr = sparse.hstack(
        [
            X_sbert_csr,
            X_liwc_csr,
            X_ngram_csr,
            X_stagger_csr,
            X_mrc2_csr,
            X_doc2vec_csr,
            X_w2v_csr,
            X_ft_csr,
        ],
        format="csr",
        dtype=np.float32
    )

    print(
        f"Shape Concat-CSR (idx {idx}): "
        f"{X_concat_csr.shape}"
    )

    # =====================================================
    # SAVE
    # =====================================================

    out_path = os.path.join(
        concat_dir,
        f"{idx}_X_concat_csr.npz"
    )

    sparse.save_npz(
        out_path,
        X_concat_csr,
        compressed=True
    )

    print(f"Concat salvo em: {out_path}\n")

    # =====================================================
    # CLEANUP
    # =====================================================

    del X_sbert
    del X_liwc
    del X_ngram
    del X_stagger
    del X_mrc2
    del X_doc2vec
    del X_w2v
    del X_ft

    del X_sbert_csr
    del X_liwc_csr
    del X_ngram_csr
    del X_stagger_csr
    del X_mrc2_csr
    del X_doc2vec_csr
    del X_w2v_csr
    del X_ft_csr

    del X_concat_csr

    gc.collect()

# =========================================================
# EXECUÇÃO PARALELA
# =========================================================

if __name__ == "__main__":

    workers = min(
        4,
        max(1, cpu_count() - 1)
    )

    print(f"Workers concatenação: {workers}\n")

    with Pool(processes=workers) as pool:

        pool.map(
            process_concat,
            indices
        )

print("Concatenação concluída.")