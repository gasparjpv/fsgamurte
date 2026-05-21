#!/usr/bin/env python
# coding: utf-8

"""
Wilcoxon Statistical Analysis
=====================================

Pipeline:
- Carrega resultados experimentais
- Executa Wilcoxon Signed-Rank Test
- Calcula redução percentual de features
- Calcula ganhos/perdas métricos
- Gera tabelas consolidadas
- Salva CSVs finais

Compatível com:
- resultados_ga_kmeans_internos.csv

Uso:
poetry run python analysis/wilcoxon_analysis.py
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURAÇÃO
# =========================================================

BASE_DIR = "vectors/abstracts/arXiv"

ESTAT_DIR = os.path.join(BASE_DIR, "estatisticas")

INPUT_RESULTS = os.path.join(ESTAT_DIR, "resultados_ga_kmeans_internos.csv")

OUTPUT_WILCOXON = os.path.join(ESTAT_DIR, "wilcoxon_results.csv")

OUTPUT_SUMMARY = os.path.join(ESTAT_DIR, "summary_statistics.csv")

OUTPUT_REDUCTION = os.path.join(ESTAT_DIR, "feature_reduction.csv")

OUTPUT_METRIC_GAIN = os.path.join(ESTAT_DIR, "metric_gain.csv")

os.makedirs(ESTAT_DIR, exist_ok=True)


# =========================================================
# UTILITÁRIOS
# =========================================================


def safe_wilcoxon(before, after):
    """
    Executa Wilcoxon com tratamento seguro de exceções.
    """

    before = np.asarray(before)
    after = np.asarray(after)

    mask = np.isfinite(before) & np.isfinite(after)

    before = before[mask]
    after = after[mask]

    if len(before) < 2:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "significant": False,
        }

    try:

        stat, p = wilcoxon(
            before,
            after,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )

        return {
            "statistic": float(stat),
            "pvalue": float(p),
            "significant": bool(p < 0.05),
        }

    except Exception:

        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "significant": False,
        }


def percentage_change(before, after):
    """
    Calcula mudança percentual.
    """

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = ((after - before) / before) * 100.0

    result[~np.isfinite(result)] = np.nan

    return result


def summarize_metric(before, after):
    """
    Resumo estatístico de métricas.
    """

    delta = after - before

    return {
        "before_mean": float(np.nanmean(before)),
        "after_mean": float(np.nanmean(after)),
        "before_std": float(np.nanstd(before)),
        "after_std": float(np.nanstd(after)),
        "delta_mean": float(np.nanmean(delta)),
        "delta_std": float(np.nanstd(delta)),
        "improved_count": int(np.sum(delta > 0)),
        "worsened_count": int(np.sum(delta < 0)),
        "equal_count": int(np.sum(delta == 0)),
    }


# =========================================================
# LOAD
# =========================================================

if not os.path.exists(INPUT_RESULTS):

    raise FileNotFoundError(f"Arquivo não encontrado:\n{INPUT_RESULTS}")

print(f"Carregando:\n{INPUT_RESULTS}")

df = pd.read_csv(INPUT_RESULTS, sep=";", decimal=",")

print("\nShape:")
print(df.shape)

print("\nColunas:")
print(df.columns.tolist())


# =========================================================
# MÉTRICAS
# =========================================================

metric_pairs = [
    ("NMI_before", "NMI_after", "NMI"),
    ("ARI_before", "ARI_after", "ARI"),
    ("ACC_before", "ACC_after", "ACC"),
    ("CH_before", "CH_after", "CH"),
    ("DB_before", "DB_after", "DB"),
    ("Dunn_before", "Dunn_after", "Dunn"),
]


# =========================================================
# WILCOXON
# =========================================================

wilcoxon_rows = []

summary_rows = []

print("\n========================================")
print(" WILCOXON TESTS")
print("========================================")

for before_col, after_col, metric_name in metric_pairs:

    if before_col not in df.columns or after_col not in df.columns:
        print(f"[WARNING] Métrica ausente: {metric_name}")
        continue

    before = df[before_col].to_numpy(dtype=np.float64)

    after = df[after_col].to_numpy(dtype=np.float64)

    # =====================================================
    # WILCOXON
    # =====================================================

    wilcox_result = safe_wilcoxon(before, after)

    row = {
        "metric": metric_name,
        "statistic": wilcox_result["statistic"],
        "pvalue": wilcox_result["pvalue"],
        "significant_0_05": wilcox_result["significant"],
    }

    wilcoxon_rows.append(row)

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = summarize_metric(before, after)

    summary["metric"] = metric_name

    summary_rows.append(summary)

    # =====================================================
    # PRINT
    # =====================================================

    print(f"\n[{metric_name}]")

    print(f"p-value: " f"{wilcox_result['pvalue']}")

    print(f"Significant: " f"{wilcox_result['significant']}")


# =========================================================
# FEATURE REDUCTION
# =========================================================

print("\n========================================")
print(" FEATURE REDUCTION")
print("========================================")

required_cols = [
    "n_features_init_total",
    "n_features_final_total",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:

    raise ValueError(f"Colunas ausentes:\n{missing}")

feature_before = df["n_features_init_total"].to_numpy(dtype=np.float64)

feature_after = df["n_features_final_total"].to_numpy(dtype=np.float64)

feature_reduction_pct = ((feature_before - feature_after) / feature_before) * 100.0

df_reduction = pd.DataFrame(
    {
        "idx": df["idx"],
        "features_before": feature_before,
        "features_after": feature_after,
        "reduction_pct": feature_reduction_pct,
    }
)

print(df_reduction.head())


# =========================================================
# MÉTRICA GAIN
# =========================================================

print("\n========================================")
print(" METRIC GAIN")
print("========================================")

metric_gain_rows = []

for before_col, after_col, metric_name in metric_pairs:

    if before_col not in df.columns or after_col not in df.columns:
        continue

    before = df[before_col].to_numpy(dtype=np.float64)

    after = df[after_col].to_numpy(dtype=np.float64)

    gain_pct = percentage_change(before, after)

    for idx, value in enumerate(gain_pct):

        metric_gain_rows.append(
            {
                "idx": idx,
                "metric": metric_name,
                "gain_pct": value,
            }
        )

df_metric_gain = pd.DataFrame(metric_gain_rows)

print(df_metric_gain.head())


# =========================================================
# DATAFRAMES FINAIS
# =========================================================

df_wilcoxon = pd.DataFrame(wilcoxon_rows)

df_summary = pd.DataFrame(summary_rows)


# =========================================================
# SAVE
# =========================================================

print("\n========================================")
print(" SAVING")
print("========================================")

df_wilcoxon.to_csv(
    OUTPUT_WILCOXON, index=False, sep=";", decimal=",", float_format="%.9f"
)

df_summary.to_csv(
    OUTPUT_SUMMARY, index=False, sep=";", decimal=",", float_format="%.9f"
)

df_reduction.to_csv(
    OUTPUT_REDUCTION, index=False, sep=";", decimal=",", float_format="%.9f"
)

df_metric_gain.to_csv(
    OUTPUT_METRIC_GAIN, index=False, sep=";", decimal=",", float_format="%.9f"
)

print("\nArquivos salvos:")

print(" -", OUTPUT_WILCOXON)
print(" -", OUTPUT_SUMMARY)
print(" -", OUTPUT_REDUCTION)
print(" -", OUTPUT_METRIC_GAIN)


# =========================================================
# PREVIEW
# =========================================================

print("\n========================================")
print(" WILCOXON RESULTS")
print("========================================")

print(df_wilcoxon)

print("\n========================================")
print(" SUMMARY")
print("========================================")

print(df_summary)

print("\n========================================")
print(" FEATURE REDUCTION")
print("========================================")

print(df_reduction.head())

print("\n========================================")
print(" METRIC GAIN")
print("========================================")

print(df_metric_gain.head())

print("\nWilcoxon analysis finalizado.")
