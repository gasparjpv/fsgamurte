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
