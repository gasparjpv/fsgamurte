import subprocess
from pathlib import Path
from datetime import datetime

# =========================================================
# BASE
# =========================================================

BASE_PATH = Path().resolve()

SCRIPTS_DIR = (
    BASE_PATH
    / "scripts"
    / "abstracts"
    / "arXiv"
)

# =========================================================
# ORDEM DO PIPELINE
# =========================================================

SCRIPTS = [

    # =========================================
    # BASES
    # =========================================

    "process_createbases.py",

    # =========================================
    # REPRESENTAÇÕES
    # =========================================

    "process_ngram.py",
    "process_word2vec.py",
    "process_fasttext.py",
    "process_doc2vec.py",
    "process_sbert.py",
    "process_liwc.py",
    "process_mrc2.py",
    "process_stagger.py",

    # =========================================
    # CONCATENAÇÃO
    # =========================================

    "process_concatenation.py",
]

# =========================================================
# EXECUÇÃO
# =========================================================

total_scripts = len(SCRIPTS)

print("\n======================================")
print("INICIANDO PIPELINE arXiv")
print("======================================\n")

inicio_pipeline = datetime.now()

for idx, script in enumerate(SCRIPTS, start=1):

    script_path = SCRIPTS_DIR / script

    print("\n--------------------------------------")
    print(f"[{idx}/{total_scripts}] {script}")
    print("--------------------------------------")

    inicio = datetime.now()

    try:

        resultado = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                str(script_path)
            ],
            cwd=BASE_PATH,
            capture_output=True,
            text=True
        )

        fim = datetime.now()

        tempo = fim - inicio

        # =====================================
        # SUCESSO
        # =====================================

        if resultado.returncode == 0:

            print(f"[OK] {script}")
            print(f"Tempo: {tempo}")

        # =====================================
        # ERRO
        # =====================================

        else:

            print(f"[ERRO] {script}")

            print("\nSTDOUT:\n")
            print(resultado.stdout)

            print("\nSTDERR:\n")
            print(resultado.stderr)

            break

    except Exception as e:

        print(f"[EXCEPTION] {script}")
        print(str(e))
        break

# =========================================================
# FINAL
# =========================================================

fim_pipeline = datetime.now()

print("\n======================================")
print("PIPELINE FINALIZADO")
print("======================================")

print(f"\nInício : {inicio_pipeline}")
print(f"Fim    : {fim_pipeline}")
print(f"Duração: {fim_pipeline - inicio_pipeline}")