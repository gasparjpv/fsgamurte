import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gafsmre.representations.liwc import LIWC
from gafsmre.representations.mrc2 import MRC2
from gafsmre.representations.stagger import STagger
from gafsmre.representations.sbert import SBert

# =========================
# Paths / Dictionaries
# =========================
BASE_PATH = Path("/mnt/d/GaFsMre")
base_dir = BASE_PATH / "vectors" / "preprocessed"

liwc_dir = base_dir / "liwc"
mrc_dir = base_dir / "mrc2"
stagger_dir = base_dir / "stagger"
sbert_dir = base_dir / "sbert"

DICT_DIR = BASE_PATH / "shared" / "dictionaries"
LIWC_DIC = DICT_DIR / "liwc" / "LIWC2015.dic"
MRC2_DIC = DICT_DIR / "mrc" / "mrc2.dct"

# Evita que libs tentem GPU automaticamente (especialmente em ambientes com CUDA "meio instalado")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =========================
# Cache de modelos (1 vez)
# =========================
@lru_cache(maxsize=1)
def get_liwc() -> LIWC:
    return LIWC(dic_filepath=LIWC_DIC)


@lru_cache(maxsize=1)
def get_mrc2() -> MRC2:
    return MRC2(dic_filepath=MRC2_DIC)


@lru_cache(maxsize=1)
def get_stagger() -> STagger:
    # Se sua implementação de STagger expõe algum parâmetro "use_gpu", force False aqui.
    # Caso não exponha, ao menos cachear evita reload/reinit a cada clique.
    return STagger()


@lru_cache(maxsize=1)
def get_sbert() -> SBert:
    """
    Força SBERT em CPU e evita re-instanciação em cada rerun do Streamlit.
    Se sua classe SBert não aceitar 'device', ela deve (idealmente) ser atualizada.
    Tentamos com device, e se falhar, caímos para a inicialização padrão.
    """
    try:
        return SBert(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=32,
            stop_word_removal_enabled=False,
            spacy_model_name="pt_core_news_sm",
            device="cpu",  # <<< REQUER suporte na classe SBert
        )
    except TypeError:
        # fallback: sua classe SBert não tem parâmetro device
        # (ainda assim cache ajuda; mas o ideal é ajustar SBert para suportar device="cpu")
        return SBert(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=32,
            stop_word_removal_enabled=False,
            spacy_model_name="pt_core_news_sm",
        )


# =========================
# Dataset vectors (offline)
# =========================
def load_dataset_vectors_from_disk():
    """
    Carrega vetores já gerados no pipeline offline.
    Espera estes arquivos:
      - vectors_liwc.npy
      - vectors_mrc2.npy
      - vectors_stagger.npy
      - vectors_sbert.npy
      - labels_liwc.npy  (labels)
    """
    X_liwc = np.load(liwc_dir / "vectors_liwc.npy", allow_pickle=False)
    X_mrc = np.load(mrc_dir / "vectors_mrc2.npy", allow_pickle=False)
    X_st = np.load(stagger_dir / "vectors_stagger.npy", allow_pickle=False)
    X_sbert = np.load(sbert_dir / "vectors_sbert.npy", allow_pickle=False)

    y = np.load(liwc_dir / "labels_liwc.npy", allow_pickle=False).astype(str)

    # L2 (garantia)
    X_liwc = normalize(X_liwc, norm="l2")
    X_mrc = normalize(X_mrc, norm="l2")
    X_st = normalize(X_st, norm="l2")
    X_sbert = normalize(X_sbert, norm="l2")

    X = np.hstack([X_liwc, X_mrc, X_st, X_sbert])
    return X, y


def train_kmeans(X, labels, k=2):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(X)

    mapping = {}
    for c in np.unique(clusters):
        true_labels = labels[clusters == c]
        vals, counts = np.unique(true_labels, return_counts=True)
        mapping[int(c)] = str(vals[int(np.argmax(counts))])

    return kmeans, clusters, mapping


# =========================
# Online vectorization (1 text)
# =========================
def vectorize_new_text(text: str):
    """
    Vetoriza 1 texto novo no MESMO formato (LIWC/MRC2/STagger/SBERT),
    normaliza L2 e concatena na ordem: LIWC, MRC2, STagger, SBERT.
    """
    corpus = [text]

    liwc = get_liwc()
    mrc2 = get_mrc2()
    stagger = get_stagger()
    sbert = get_sbert()

    _, X_liwc = liwc.generate_representation(corpus)
    _, X_mrc = mrc2.generate_representation(corpus)
    _, X_st = stagger.generate_representation(corpus, as_dataframe=False)
    _, X_sbert = sbert.generate_representation(corpus)

    X_liwc = normalize(np.asarray(X_liwc), norm="l2")
    X_mrc = normalize(np.asarray(X_mrc), norm="l2")
    X_st = normalize(np.asarray(X_st), norm="l2")
    X_sbert = normalize(np.asarray(X_sbert), norm="l2")

    X_new = np.hstack([X_liwc, X_mrc, X_st, X_sbert])
    return X_new


# =========================
# 2D projection for plotting
# =========================
def fit_projection(X):
    svd = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd.fit_transform(X)
    return svd, X2
