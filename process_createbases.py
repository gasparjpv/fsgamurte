# === CÉLULA 1: carregamento da base de dados ===

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import ast

# =========================================================
# CAMINHO
# =========================================================

BASE = Path().resolve()

path = (
    BASE
    / "shared"
    / "datasets"
    / "Abstracts"
    / "arXiv"
    / "cleansed_data.csv"
)

# =========================================================
# CARREGAR CSV
# =========================================================

df = pd.read_csv(path)

# nome da coluna de texto
col_texto = "abstracts"

# =========================================================
# TRATAMENTO DOS TERMS -> MAJOR_TERMS
# =========================================================

def parse_terms(x):

    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    try:
        return ast.literal_eval(x)
    except:
        return []

def extract_major_terms(terms):

    majors = sorted({
        term.split(".")[0].strip()
        for term in terms
        if isinstance(term, str) and "." in term
    })

    return majors

def build_major_class(majors):

    if len(majors) == 0:
        return None

    return "{" + ", ".join(majors) + "}"

# =========================================================
# GERAR major_terms E major_class
# =========================================================

COL_TERMS = "terms"

df[COL_TERMS] = df[COL_TERMS].apply(parse_terms)

df["major_terms"] = df[COL_TERMS].apply(extract_major_terms)

df["major_class"] = df["major_terms"].apply(build_major_class)

print("[OK] major_terms gerado")

# =========================================================
# FILTROS
# =========================================================

df["num_terms"] = df["major_terms"].apply(len)

df["num_words"] = (
    df[col_texto]
    .fillna("")
    .astype(str)
    .str.split()
    .apply(len)
)

df_filtrado = df[
    (df["major_terms"].apply(len) > 0) &
    (df["num_words"] > 100)
].copy()

print(f"[OK] filtros aplicados | total={len(df_filtrado)}")

# =========================================================
# CONFIG
# =========================================================

seeds = [42, 7, 99, 123, 2025, 314, 777, 888, 1024, 4096]

NUM_ARQUIVOS = len(seeds)

AMOSTRAS_POR_ARQUIVO = 2000

TOTAL_NECESSARIO = NUM_ARQUIVOS * AMOSTRAS_POR_ARQUIVO

# =========================================================
# VALIDAÇÃO
# =========================================================

if len(df_filtrado) < TOTAL_NECESSARIO:

    raise ValueError(
        f"Base insuficiente.\n"
        f"Necessário: {TOTAL_NECESSARIO}\n"
        f"Disponível: {len(df_filtrado)}"
    )

# =========================================================
# EMBARALHAMENTO GLOBAL
# =========================================================

df_embaralhado = df_filtrado.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("[OK] embaralhamento global realizado")

# =========================================================
# OUTPUT
# =========================================================

output_dir = Path("vectors/abstracts/arXiv/database")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# ESTATÍSTICAS
# =========================================================

estatisticas_samples = []

# =========================================================
# GERAÇÃO DOS ARQUIVOS
# =========================================================

for i, seed in enumerate(seeds):

    inicio = i * AMOSTRAS_POR_ARQUIVO

    fim = inicio + AMOSTRAS_POR_ARQUIVO

    # =====================================================
    # SAMPLE SEM SOBREPOSIÇÃO
    # =====================================================

    df_sample = df_embaralhado.iloc[
        inicio:fim
    ].copy()

    # =====================================================
    # ESTATÍSTICAS
    # =====================================================

    counts = (
        df_sample["major_class"]
        .value_counts()
        .to_dict()
    )

    estatisticas_samples.append({
        "seed": seed,
        "{cs}": counts.get("{cs}", 0),
        "{cs, stat}": counts.get("{cs, stat}", 0),
        "{cs, eess}": counts.get("{cs, eess}", 0),
    })

    # =====================================================
    # PARQUET
    # =====================================================

    parquet_path = (
        output_dir
        / f"{i}_df_corpus_5000.parquet"
    )

    df_sample.to_parquet(
        parquet_path,
        index=False
    )

    # =====================================================
    # CSV
    # =====================================================

    csv_path = (
        output_dir
        / f"{i}_df_sample_5000.csv"
    )

    df_sample.to_csv(
        csv_path,
        index=False
    )

    print(f"[OK] seed {seed} processada")

    del df_sample

# =========================================================
# RESUMO FINAL
# =========================================================

df_estatisticas = pd.DataFrame(
    estatisticas_samples
)

print("\n==============================")
print("RESUMO FINAL DAS AMOSTRAS")
print("==============================\n")

print(df_estatisticas)

print("\n==============================")
print("MÉDIA POR CLASSE")
print("==============================\n")

print(
    df_estatisticas[
        [
            "{cs}",
            "{cs, stat}",
            "{cs, eess}"
        ]
    ].mean()
)

print("\n[OK] finalizado")

# =========================================================
# CLEANUP
# =========================================================

del df_filtrado
del df_embaralhado
del df