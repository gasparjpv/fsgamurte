#!/usr/bin/env python
# coding: utf-8

# In[1]:


# === CÉLULA 1: Configurações e caminhos ===

import os

import numpy as np

from scipy.sparse import load_npz

# =========================================================
# DIRETÓRIOS
# =========================================================

base_dir = "vectors/abstracts/arXiv"

concat_dir = os.path.join(base_dir, "concat")
database_dir = os.path.join(base_dir, "database")
estat_dir = os.path.join(base_dir, "estatisticas")

# Representações
sbert_dir = os.path.join(base_dir, "Sbert")
liwc_dir = os.path.join(base_dir, "LIWC")
ngram_dir = os.path.join(base_dir, "NGram")
stagger_dir = os.path.join(base_dir, "STagger")
mrc2_dir = os.path.join(base_dir, "MRC2")
doc2vec_dir = os.path.join(base_dir, "Doc2Vec")
w2v_dir = os.path.join(base_dir, "Word2Vec")
ft_dir = os.path.join(base_dir, "FastText")

os.makedirs(estat_dir, exist_ok=True)

# =========================================================
# ÍNDICES
# =========================================================

indices = list(range(10))

# =========================================================
# LOADER INTELIGENTE
# =========================================================


def get_shape(path_without_ext):

    npy_path = f"{path_without_ext}.npy"
    npz_path = f"{path_without_ext}.npz"

    # -----------------------------------------------------
    # SPARSE NPZ
    # -----------------------------------------------------

    if os.path.exists(npz_path):

        X = load_npz(npz_path)

        return X.shape

    # -----------------------------------------------------
    # DENSO NPY
    # -----------------------------------------------------

    elif os.path.exists(npy_path):

        X = np.load(npy_path, mmap_mode="r")

        return X.shape

    # -----------------------------------------------------
    # ERRO
    # -----------------------------------------------------

    else:

        raise FileNotFoundError(f"Arquivo não encontrado:\n" f"{path_without_ext}")


# =========================================================
# FATIAS DAS VISUALIZAÇÕES
# =========================================================


def get_view_slices(idx: int):
    """
    Retorna:
        {view: (start, end)}

    Ordem:
        [
            SBert,
            LIWC,
            NGram,
            STagger,
            MRC2,
            Doc2Vec,
            Word2Vec,
            FastText
        ]
    """

    fname_corpus = f"{idx}_df_corpus_5000"

    # =====================================================
    # SHAPES
    # =====================================================

    lens = {}

    lens["SBert"] = get_shape(os.path.join(sbert_dir, f"{fname_corpus}_SBert"))[1]

    lens["LIWC"] = get_shape(os.path.join(liwc_dir, f"{fname_corpus}_LIWC"))[1]

    lens["NGram"] = get_shape(os.path.join(ngram_dir, f"{fname_corpus}_NGram"))[1]

    lens["STagger"] = get_shape(os.path.join(stagger_dir, f"{fname_corpus}_STagger"))[1]

    lens["MRC2"] = get_shape(os.path.join(mrc2_dir, f"{fname_corpus}_MRC2"))[1]

    lens["Doc2Vec"] = get_shape(os.path.join(doc2vec_dir, f"{idx}_Doc2Vec"))[1]

    lens["Word2Vec"] = get_shape(os.path.join(w2v_dir, f"{idx}_Word2Vec"))[1]

    lens["FastText"] = get_shape(os.path.join(ft_dir, f"{idx}_FastText"))[1]

    # =====================================================
    # ORDEM CONCATENAÇÃO
    # =====================================================

    ordered_views = [
        "SBert",
        "LIWC",
        "NGram",
        "STagger",
        "MRC2",
        "Doc2Vec",
        "Word2Vec",
        "FastText",
    ]

    # =====================================================
    # FATIAS
    # =====================================================

    slices = {}

    current = 0

    for view in ordered_views:

        n = lens[view]

        slices[view] = (current, current + n)

        current += n

    return slices


# =========================================================
# DEBUG
# =========================================================

print("Configuração carregada.")
print("Indices:", indices)
print("Concat dir:", concat_dir)
print("Database dir:", database_dir)
print("Estat dir:", estat_dir)


# In[2]:


# === CÉLULA 2: GA binário (via geneticalgorithm) + utilitários + métricas internas ===
import numpy as np
from numpy.random import default_rng
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
)
from scipy import sparse
from geneticalgorithm import geneticalgorithm as ga


# ACC via Hungarian
def clustering_accuracy(y_true, y_pred) -> float:
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = {c: i for i, c in enumerate(np.unique(y_true))}
    clusters = {c: i for i, c in enumerate(np.unique(y_pred))}

    C = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        C[clusters[yp], classes[yt]] += 1

    row_ind, col_ind = linear_sum_assignment(C.max() - C)
    return C[row_ind, col_ind].sum() / len(y_true)


def _slice_cols(X, cols_mask):
    """
    Fatiamento de colunas que funciona para denso (numpy) e esparso (CSR/CSC).
    cols_mask pode ser booleano (len = n_feats) ou array de índices.
    """
    if isinstance(cols_mask, np.ndarray) and cols_mask.dtype == bool:
        if sparse.issparse(X):
            idx = np.arange(X.shape[1])[cols_mask]
            return X[:, idx]
        else:
            return X[:, cols_mask]
    else:
        # lista/array de índices
        return X[:, cols_mask]


def dunn_index_centroid(X, labels, metric: str = "cosine") -> float:
    """
    Approx. Dunn Index via centróides:
    - intra: maior distância doc–centróide dentro de um cluster
    - inter: menor distância entre centróides de clusters diferentes
    Dunn = inter_min / intra_max

    Funciona para X denso ou esparso.
    """
    labels = np.asarray(labels)
    unique = np.unique(labels)

    if len(unique) < 2:
        return 0.0

    centroids = []
    intra_max = 0.0

    for c in unique:
        mask = labels == c
        Xc = X[mask]
        if Xc.shape[0] <= 1:
            continue

        if sparse.issparse(Xc):
            centroid = Xc.mean(axis=0)
            centroid = np.asarray(centroid).ravel()
        else:
            centroid = Xc.mean(axis=0)

        centroids.append(centroid)

        dists = pairwise_distances(Xc, centroid.reshape(1, -1), metric=metric)
        intra_max = max(intra_max, float(dists.max()))

    centroids = np.vstack(centroids)
    inter = pairwise_distances(centroids, metric=metric)
    np.fill_diagonal(inter, np.inf)
    inter_min = float(inter.min())

    if intra_max == 0.0:
        return 0.0
    return inter_min / intra_max


def rodar_ga_fs(
    X_full,
    y_true,
    K_CLUST,
    *,
    random_state=42,
    pop_size=30,
    generations=20,
    p_init_keep=0.30,
    mut_rate=0.02,
    cxpb=0.80,
    elitism=2,
    min_feats_ratio=0.05,
    min_feats_abs=10,
    alpha_sparsity=0.01,  # 0.0 => fitness puramente por NMI
    ninit_fast=10,
    ninit_final=20,
    tag="[FS GA]",
):
    """
    Executa GA de seleção de features sobre X_full e retorna:
      result (dict com métricas externas) + best_mask (np.bool_).

    Agora usando a biblioteca `geneticalgorithm` (GA elitista padrão).
    Funciona com X_full denso ou CSR.
    """
    rng = default_rng(random_state)
    np.random.seed(random_state)  # para o geneticalgorithm

    y_true = np.asarray(y_true)
    n_docs, n_feats = X_full.shape

    min_feats = max(min_feats_abs, int(min_feats_ratio * n_feats))

    # --- função objetivo para o geneticalgorithm ---
    def objective(x):
        mask = np.array(x, dtype=bool)
        n_on = mask.sum()

        # Penalização para soluções com poucas features
        if n_on < min_feats:
            return 1e3 + (min_feats - n_on)

        X_sel = _slice_cols(X_full, mask)

        # KMeans para avaliar a solução
        km = KMeans(
            n_clusters=K_CLUST,
            n_init=ninit_fast,
            random_state=random_state,
        )
        labels_pred = km.fit_predict(X_sel)

        # Dunn Index (métrica interna)
        dunn = dunn_index_centroid(X_sel, labels_pred, metric="cosine")

        # Penalização por sparsity (opcional)
        penalty = alpha_sparsity * (n_on / n_feats)

        # GA minimiza → queremos maximizar Dunn
        return -(dunn - penalty)

    # Parâmetros do GA (mapeando os que você já usava)
    elit_ratio = max(float(elitism) / float(pop_size), 0.0)

    algorithm_param = {
        "max_num_iteration": generations,
        "population_size": pop_size,
        "mutation_probability": mut_rate,
        "elit_ratio": elit_ratio,
        "crossover_probability": cxpb,
        "parents_portion": 0.3,
        "crossover_type": "uniform",
        "max_iteration_without_improv": None,
    }

    # Cria e roda o GA da biblioteca
    model = ga(
        function=objective,
        dimension=n_feats,
        variable_type="bool",
        function_timeout=600,  # segundos (ajuste se precisar)
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        progress_bar=False,
    )

    model.run()

    # Recupera melhor solução encontrada
    out = model.output_dict
    best_vector = np.array(out["variable"])
    best_mask = best_vector.astype(bool)

    # Só por segurança, garante mínimo de features
    if best_mask.sum() < min_feats:
        # liga features aleatórias até atingir min_feats
        off = np.where(~best_mask)[0]
        need = min_feats - best_mask.sum()
        if need > 0 and off.size >= need:
            turn_on = rng.choice(off, size=need, replace=False)
            best_mask[turn_on] = True

    # Métricas externas finais (com n_init "lento")
    X_sel = _slice_cols(X_full, best_mask)
    km_final = KMeans(
        n_clusters=K_CLUST,
        n_init=ninit_final,
        random_state=random_state,
    )
    pred = km_final.fit_predict(X_sel)

    nmi = normalized_mutual_info_score(y_true, pred)
    ari = adjusted_rand_score(y_true, pred)
    acc = clustering_accuracy(y_true, pred)

    result = {
        "tag": tag,
        "n_features_init": n_feats,
        "n_features_final": int(best_mask.sum()),
        "NMI": nmi,
        "ARI": ari,
        "ACC": acc,
    }

    return result, best_mask


print("Configuração do GA (geneticalgorithm) e métricas internas carregada.")


# In[ ]:


# === CÉLULA 3: Rodar KMeans antes, GA FS + KMeans depois ===

import os
import gc
import time

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.cluster import KMeans

from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from threadpoolctl import threadpool_limits

# =========================================================
# CONFIGURAÇÃO
# =========================================================

RANDOM_STATE = 42

CPU_THREADS = min(8, max(1, os.cpu_count() - 1))

print(f"Threads sklearn/OpenMP: {CPU_THREADS}")

# =========================================================
# DIRETÓRIOS
# =========================================================

BASE_DIR = "vectors/abstracts/arXiv"

database_dir = os.path.join(BASE_DIR, "database")

concat_dir = os.path.join(BASE_DIR, "concat")

estat_dir = os.path.join(BASE_DIR, "estatisticas")

os.makedirs(estat_dir, exist_ok=True)

print(f"database_dir: {database_dir}")
print(f"concat_dir:   {concat_dir}")
print(f"estat_dir:    {estat_dir}")

# =========================================================
# RESULTADOS PARCIAIS
# =========================================================

results_path = os.path.join(estat_dir, "resultados_parciais.csv")

if os.path.exists(results_path):

    resultados_df_existente = pd.read_csv(results_path)

    resultados_existentes = resultados_df_existente["idx"].astype(int).tolist()

    resultados = resultados_df_existente.to_dict("records")

    print(f"Resultados existentes: " f"{resultados_existentes}")

else:

    resultados_existentes = []

    resultados = []

# =========================================================
# VIEW SLICES
# =========================================================


def get_view_slices(idx):
    """
    Retorna os slices das views concatenadas.
    """

    metadata_path = os.path.join(concat_dir, f"{idx}_view_slices.json")

    if not os.path.exists(metadata_path):

        raise FileNotFoundError(f"Arquivo não encontrado:\n" f"{metadata_path}")

    metadata = pd.read_json(metadata_path, typ="series")

    return {key: tuple(value) for key, value in metadata.items()}


# =========================================================
# VIEW SLICES DINÂMICO
# =========================================================


def get_view_slices(idx):
    """
    Reconstrói os slices das views
    a partir das matrizes salvas.
    """

    views_dir = {
        "BoW": "BoW",
        "LIWC": "LIWC",
        "MRC2": "MRC2",
        "POS": "POS",
        "Word2Vec": "Word2Vec",
        "FastText": "FastText",
        "Doc2Vec": "Doc2Vec",
        "SBERT": "SBERT",
    }

    slices = {}

    start = 0

    for view_name, folder in views_dir.items():

        # =================================================
        # NPZ
        # =================================================

        npz_path = os.path.join(BASE_DIR, folder, f"{idx}_{folder}.npz")

        # =================================================
        # NPY
        # =================================================

        npy_path = os.path.join(BASE_DIR, folder, f"{idx}_{folder}.npy")

        # =================================================
        # LOAD MATRIX
        # =================================================

        if os.path.exists(npz_path):

            X = sparse.load_npz(npz_path)

            feat_count = X.shape[1]

            del X

        elif os.path.exists(npy_path):

            X = np.load(npy_path, mmap_mode="r")

            feat_count = X.shape[1]

            del X

        else:

            print(f"[WARNING] View não encontrada: " f"{view_name}")

            continue

        end = start + feat_count

        slices[view_name] = (start, end)

        start = end

    print(f"Slices reconstruídos idx={idx}:")

    print(slices)

    return slices


# =========================================================
# LOOP PRINCIPAL
# =========================================================

for idx in range(10):

    # =====================================================
    # SKIP TOTAL
    # =====================================================

    if idx in resultados_existentes:

        print(f"\n[SKIP] Índice {idx} já processado.")

        continue

    print("\n====================")
    print(f" Rodando índice {idx}")
    print("====================")

    t_idx_start = time.perf_counter()

    # =====================================================
    # PATHS
    # =====================================================

    mask_path = os.path.join(estat_dir, f"{idx}_best_mask.npy")

    labels_before_path = os.path.join(estat_dir, f"{idx}_labels_before.npy")

    labels_after_path = os.path.join(estat_dir, f"{idx}_labels_after.npy")

    # =====================================================
    # LABELS
    # =====================================================

    df_sample = pd.read_csv(os.path.join(database_dir, f"{idx}_df_sample_5000.csv"))

    y_true = df_sample["major_class"].astype(str).to_numpy()

    K_CLUST = df_sample["major_class"].nunique()

    # =====================================================
    # MATRIZ CONCAT
    # =====================================================

    X_concat_csr = sparse.load_npz(
        os.path.join(concat_dir, f"{idx}_X_concat_csr.npz")
    ).astype(np.float32)

    assert X_concat_csr.shape[0] == len(y_true)

    n_docs, n_feats = X_concat_csr.shape

    view_slices = get_view_slices(idx)

    # =====================================================
    # KMEANS BEFORE
    # =====================================================

    if os.path.exists(labels_before_path):

        print(f"KMeans BEFORE já existe " f"(idx={idx})")

        y_pred_full = np.load(labels_before_path)

        time_kmeans_before = 0.0

    else:

        print(f"Executando KMeans BEFORE " f"(idx={idx})")

        t0 = time.perf_counter()

        km_full = KMeans(
            n_clusters=K_CLUST,
            n_init=20,
            random_state=RANDOM_STATE,
            algorithm="lloyd",
        )

        with threadpool_limits(limits=CPU_THREADS):

            y_pred_full = km_full.fit_predict(X_concat_csr)

        time_kmeans_before = time.perf_counter() - t0

        np.save(labels_before_path, y_pred_full)

    # =====================================================
    # MÉTRICAS BEFORE
    # =====================================================

    nmi_before = normalized_mutual_info_score(y_true, y_pred_full)

    ari_before = adjusted_rand_score(y_true, y_pred_full)

    acc_before = clustering_accuracy(y_true, y_pred_full)

    X_dense_before = X_concat_csr.toarray().astype(np.float32)

    ch_before = calinski_harabasz_score(X_dense_before, y_pred_full)

    db_before = davies_bouldin_score(X_dense_before, y_pred_full)

    del X_dense_before

    gc.collect()

    dunn_before = dunn_index_centroid(X_concat_csr, y_pred_full, metric="cosine")

    # =====================================================
    # GA FS
    # =====================================================

    if os.path.exists(mask_path):

        print(f"Máscara GA já existe " f"(idx={idx})")

        best_mask = np.load(mask_path).astype(bool)

        time_ga_fs = 0.0

    else:

        print(f"Executando GA " f"(idx={idx})")

        t0 = time.perf_counter()

        _, mask = rodar_ga_fs(
            X_full=X_concat_csr,
            y_true=y_true,
            K_CLUST=K_CLUST,
            random_state=RANDOM_STATE,
            pop_size=30,
            generations=20,
            p_init_keep=0.30,
            mut_rate=0.02,
            cxpb=0.80,
            elitism=1,
            min_feats_ratio=0.10,
            min_feats_abs=10,
            alpha_sparsity=0.01,
            ninit_fast=10,
            ninit_final=20,
            tag=f"[Concat-ALL+FS GA | idx={idx}]",
        )

        time_ga_fs = time.perf_counter() - t0

        best_mask = mask.astype(bool)

        np.save(mask_path, best_mask)

    # =====================================================
    # KMEANS AFTER
    # =====================================================

    X_sel = _slice_cols(X_concat_csr, best_mask)

    if os.path.exists(labels_after_path):

        print(f"KMeans AFTER já existe " f"(idx={idx})")

        y_pred_fs = np.load(labels_after_path)

        time_kmeans_after = 0.0

    else:

        print(f"Executando KMeans AFTER " f"(idx={idx})")

        t0 = time.perf_counter()

        km_fs = KMeans(
            n_clusters=K_CLUST,
            n_init=20,
            random_state=RANDOM_STATE,
            algorithm="lloyd",
        )

        with threadpool_limits(limits=CPU_THREADS):

            y_pred_fs = km_fs.fit_predict(X_sel)

        time_kmeans_after = time.perf_counter() - t0

        np.save(labels_after_path, y_pred_fs)

    # =====================================================
    # MÉTRICAS AFTER
    # =====================================================

    nmi_after = normalized_mutual_info_score(y_true, y_pred_fs)

    ari_after = adjusted_rand_score(y_true, y_pred_fs)

    acc_after = clustering_accuracy(y_true, y_pred_fs)

    X_dense_after = X_sel.toarray().astype(np.float32)

    ch_after = calinski_harabasz_score(X_dense_after, y_pred_fs)

    db_after = davies_bouldin_score(X_dense_after, y_pred_fs)

    del X_dense_after

    gc.collect()

    dunn_after = dunn_index_centroid(X_sel, y_pred_fs, metric="cosine")

    # =====================================================
    # TEMPO
    # =====================================================

    time_total_idx = time.perf_counter() - t_idx_start

    # =====================================================
    # RESULTADO
    # =====================================================

    row = {
        "idx": idx,
        "n_docs": int(n_docs),
        "n_clusters": int(K_CLUST),
        "n_features_init_total": int(n_feats),
        "n_features_final_total": int(best_mask.sum()),
        "NMI_before": float(nmi_before),
        "ARI_before": float(ari_before),
        "ACC_before": float(acc_before),
        "NMI_after": float(nmi_after),
        "ARI_after": float(ari_after),
        "ACC_after": float(acc_after),
        "CH_before": float(ch_before),
        "DB_before": float(db_before),
        "Dunn_before": float(dunn_before),
        "CH_after": float(ch_after),
        "DB_after": float(db_after),
        "Dunn_after": float(dunn_after),
        "time_kmeans_before": float(time_kmeans_before),
        "time_ga_fs": float(time_ga_fs),
        "time_kmeans_after": float(time_kmeans_after),
        "time_total_idx": float(time_total_idx),
    }

    # =====================================================
    # FEATURES POR VIEW
    # =====================================================

    for view, (start, end) in view_slices.items():

        row[f"{view}_init"] = int(end - start)

        row[f"{view}_final"] = int(best_mask[start:end].sum())

    resultados.append(row)

    # =====================================================
    # SAVE PARCIAL
    # =====================================================

    pd.DataFrame(resultados).to_csv(results_path, index=False)

    print(f"\nResultados salvos em:\n" f"{results_path}")

    print(f"Índice {idx} concluído em " f"{time_total_idx:.2f}s")

    # =====================================================
    # CLEANUP
    # =====================================================

    del df_sample
    del y_true
    del X_concat_csr
    del X_sel
    del y_pred_full
    del y_pred_fs
    del best_mask

    gc.collect()

# =========================================================
# FINAL
# =========================================================

gc.collect()

print("\nGA + KMeans finalizado.")


# In[ ]:


# === CÉLULA 4: salvar resultados e estatísticas ===
import pandas as pd
import numpy as np
import os

# resultados veio da CÉLULA 3
df_results = pd.DataFrame(resultados)

# Colunas de métricas internas para média e desvio padrão
metric_cols = [
    "calinski_harabasz_before",
    "calinski_harabasz_after",
    "davies_bouldin_before",
    "davies_bouldin_after",
    "dunn_before",
    "dunn_after",
    "time_kmeans_before",
    "time_ga_fs",
    "time_kmeans_after",
    "time_total_idx",
]


estatisticas = df_results[metric_cols].agg(["mean", "std"])

# Salvar CSVs
out_results_path = os.path.join(estat_dir, "resultados_ga_kmeans_internos.csv")
out_stats_path = os.path.join(estat_dir, "estatisticas_ga_kmeans_internos.csv")

df_results.to_csv(
    out_results_path, index=False, sep=";", decimal=",", float_format="%.9f"
)

estatisticas.to_csv(out_stats_path, sep=";", decimal=",", float_format="%.9f")

print("Salvo em:")
print(" -", out_results_path)
print(" -", out_stats_path)

df_results, estatisticas


# In[ ]:


df_results.head(11)


# In[ ]:


estatisticas.head(3)
